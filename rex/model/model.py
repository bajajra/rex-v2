"""RexModel: forward schedule (§5.4), checkpoint segments (§6.4), output head (§5.8),
auxiliary-loss plumbing (§5.7) and inference caches (§5.10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import RexConfig
from .attention import KVEntry
from .attnres import DepthMix
from .layer import AppContext, Layer
from .loop import GATE_OVERRIDES, GateMix
from .norm import RMSNorm, nrm


@dataclass
class RexOutput:
    hidden: torch.Tensor                 # hfinal [N, d]; logits via RexModel.logits(hidden)
    aux_loss: torch.Tensor               # lb_coef * Σ_layers L_LB + z_coef * Σ_layers L_Z (scalar)
    lb_loss: torch.Tensor                # [L] per stored layer (empty for dense FFN)
    z_loss: torch.Tensor                 # [L] per stored layer (empty for dense FFN)
    load: Optional[torch.Tensor]         # [L, E] f_mb pooled over applications (detached); None if dense
    routing: List[dict] = field(default_factory=list)
    # one entry per MoE application: {"layer", "pass", "counts" [E], "max_weight_mean", "top_i"?}


class InferenceCache:
    """Per-application KV caches for one sequence: each of the L + (T-1)·C applications owns one."""

    def __init__(self) -> None:
        self.entries: Dict[Tuple[int, Optional[int]], KVEntry] = {}

    def get(self, layer: int, pass_idx: Optional[int]) -> KVEntry:
        return self.entries.setdefault((layer, pass_idx), KVEntry())


def packed_positions(cu: torch.Tensor) -> torch.Tensor:
    """pos[N] (position of each token inside its sequence) from cu[n_seq+1]."""
    lens = (cu[1:] - cu[:-1]).to(torch.int64)
    starts = cu[:-1].to(torch.int64).repeat_interleave(lens)
    return torch.arange(int(cu[-1]), device=cu.device, dtype=torch.int64) - starts


def next_token_targets(ids: torch.Tensor, cu: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """targets[i] = ids[i+1] if i+1 is in the same sequence, else ignore_index (§5.8)."""
    tgt = torch.full_like(ids, ignore_index)
    tgt[:-1] = ids[1:]
    ends = cu[1:].to(torch.int64) - 1
    tgt[ends[ends >= 0]] = ignore_index
    return tgt


class RexModel(nn.Module):
    """rex-v2 decoder. Construct with a validated :class:`RexConfig` (uses ``cfg.model``,
    ``cfg.loss`` for the auxiliary losses and ``cfg.train.grad_ckpt / tok_chunk / max_len``)."""

    def __init__(self, cfg: RexConfig):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.mcfg = m
        self.d, self.L = m.d_model, m.n_layers
        self.eps = m.rms_eps
        self.n_passes = m.n_passes
        self.bs = m.block_size
        self.residual = m.residual
        self.is_moe = m.ffn.type == "moe"
        self.lb_coef, self.z_coef, self.rho = cfg.loss.lb_coef, cfg.loss.z_coef, cfg.loss.lb_load_ema
        self.grad_ckpt = cfg.train.grad_ckpt
        self.tok_chunk = cfg.train.tok_chunk

        self.emb = nn.Embedding(m.vocab_size, m.d_model)
        self.emb_norm = RMSNorm(m.d_model, m.rms_eps)
        self.layers = nn.ModuleList([Layer(cfg, l) for l in range(m.n_layers)])
        self.mix_out = DepthMix(m.d_model) if m.residual == "attnres" else None
        self.final_norm = RMSNorm(m.d_model, m.rms_eps)
        self.lm_head = None if m.tie_embeddings else nn.Linear(m.d_model, m.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ utilities
    def output_weight(self) -> torch.Tensor:
        return self.emb.weight if self.lm_head is None else self.lm_head.weight

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """§5.8: hfinal @ Emb^T over columns 0..unpadded_vocab_size-1, then c·tanh(x/c), float32."""
        w = self.output_weight()[: self.mcfg.unpadded_vocab_size]
        z = F.linear(hidden, w).float()
        c = self.mcfg.final_logit_softcap
        return c * torch.tanh(z / c)

    def gate_mixes(self) -> List[GateMix]:
        return [mod for mod in self.modules() if isinstance(mod, GateMix)]

    def depth_mixes(self) -> List[DepthMix]:
        return [mod for mod in self.modules() if isinstance(mod, DepthMix)]

    def set_gate_override(self, mode: Optional[str]) -> None:
        """None | "no_message" (γ = 0) | "identity" (γ = 0, κ = 1) on every GateMix."""
        if mode not in GATE_OVERRIDES:
            raise ValueError(mode)
        for g in self.gate_mixes():
            g.override = mode

    @torch.no_grad()
    def update_balance_biases(self, f_step: torch.Tensor) -> None:
        """``balance: bias`` ablation; f_step [L, E] is the load of the optimiser step."""
        for layer, f in zip(self.layers, f_step):
            layer.ffn.update_balance_bias(f, self.cfg.loss.bias_update_speed)

    def new_cache(self) -> InferenceCache:
        return InferenceCache()

    # ------------------------------------------------------------------ forward
    def forward(self, ids: torch.Tensor, cu: torch.Tensor, pos: torch.Tensor, T: Optional[int] = None, *,
                recorder: Optional[Callable[[dict], None]] = None, return_routing: bool = False,
                cache: Optional[InferenceCache] = None) -> RexOutput:
        """ids [N] int64, cu [n_seq+1] int32, pos [N] int64 (packed layout, §5.1).

        T: number of core passes (1..n_passes; default n_passes).
        recorder: optional callable receiving dict events ("depthmix", "gatemix", "block")
            for evaluation; intended for no-grad forwards (checkpoint recomputation would
            repeat events).
        cache: InferenceCache for incremental decoding of a single sequence (cu = [0, N]).
        """
        T = self.n_passes if T is None else T
        if not 1 <= T <= self.n_passes:
            raise ValueError(f"T must be in [1, {self.n_passes}], got {T}")
        if cache is not None and cu.numel() != 2:
            raise ValueError("incremental decoding supports a single sequence (cu = [0, N])")
        use_ckpt = self.grad_ckpt == "layer" and torch.is_grad_enabled() and cache is None
        stats: List[List[dict]] = [[] for _ in range(self.L)]
        routing: List[dict] = []

        def ctx(l: int, pass_idx: Optional[int], names: List[str]) -> AppContext:
            return AppContext(pass_idx=pass_idx, src_names=names, recorder=recorder,
                              cache=None if cache is None else cache.get(l, pass_idx),
                              return_routing=return_routing,
                              tok_chunk=self.tok_chunk if use_ckpt else None)

        def run(l: int, done_hat, R, prev, c: AppContext):
            layer = self.layers[l]
            if use_ckpt:
                R, st = checkpoint(layer, done_hat, R, prev, cu, pos, c, use_reentrant=False)
            else:
                R, st = layer(done_hat, R, prev, cu, pos, c)
            self._collect(st, l, c.pass_idx, stats, routing)
            return R

        def record_block(name: str, pass_idx, value):
            if recorder is not None:
                recorder({"kind": "block", "name": name, "pass": pass_idx, "value": value.detach()})

        v0 = self.emb_norm(self.emb(ids))

        if self.residual == "stream":
            x = v0
            for l in range(self.L):
                c = ctx(l, None, [])
                layer = self.layers[l]
                if use_ckpt:
                    x, st = checkpoint(layer.forward_stream, x, cu, pos, c, use_reentrant=False)
                else:
                    x, st = layer.forward_stream(x, cu, pos, c)
                self._collect(st, l, None, stats, routing)
            hidden = self.final_norm(x)
            return self._finish(hidden, stats, routing)

        bs = self.bs
        n = lambda t: nrm(t, self.eps)  # noqa: E731
        v0h = n(v0)

        # B1 (prelude)
        R = None
        for l in range(0, bs):
            R = run(l, [v0h], R, None, ctx(l, None, ["v0"]))
        V1 = R
        record_block("V1", None, V1)
        V1h = n(V1)

        # Core: B2, B3 with shared weights, T passes; every pass restarts from (v0, V1).
        prev = None
        V2 = V3 = None
        for t in range(1, T + 1):
            R = None
            names = ["v0", "V1"]
            for l in range(bs, 2 * bs):
                R = run(l, [v0h, V1h], R, prev, ctx(l, t, names))
            V2 = R
            record_block("V2", t, V2)
            V2h = n(V2)
            R = None
            names = ["v0", "V1", f"V2[{t}]"]
            for l in range(2 * bs, 3 * bs):
                R = run(l, [v0h, V1h, V2h], R, prev, ctx(l, t, names))
            V3 = R
            record_block("V3", t, V3)
            V3h = n(V3)
            prev = (V2h, V3h)  # read by pass t+1

        # B4 (coda), sources from the last pass only
        done_hat = [v0h, V1h, V2h, V3h]
        names = ["v0", "V1", f"V2[{T}]", f"V3[{T}]"]
        R = None
        for l in range(3 * bs, 4 * bs):
            R = run(l, done_hat, R, None, ctx(l, None, names))
        V4 = R
        record_block("V4", None, V4)

        out_sources = done_hat + [n(V4)]
        if recorder is None:
            x = self.mix_out(out_sources)
        else:
            x, alpha = self.mix_out(out_sources, return_weights=True)
            recorder({"kind": "depthmix", "reader": "out", "layer": None, "pass": None,
                      "sources": names + ["V4"], "alpha": alpha.detach()})
        hidden = self.final_norm(x)
        return self._finish(hidden, stats, routing)

    # ------------------------------------------------------------------ aux plumbing
    def _collect(self, st, l, pass_idx, stats, routing) -> None:
        if st is None:
            return
        stats[l].append(st)
        entry = {"layer": l, "pass": pass_idx, "counts": st["counts"].detach(),
                 "max_weight_mean": (st["maxw_sum"] / st["n_tokens"]).detach()}
        if "top_i" in st:
            entry["top_i"] = st["top_i"]
        routing.append(entry)

    def _finish(self, hidden: torch.Tensor, stats: List[List[dict]], routing: List[dict]) -> RexOutput:
        dev = hidden.device
        if not self.is_moe:
            z = torch.zeros((), device=dev)
            return RexOutput(hidden, z, torch.zeros(0, device=dev), torch.zeros(0, device=dev), None, routing)
        lbs, zs, loads = [], [], []
        update = self.training
        for l, apps in enumerate(stats):
            # pool all applications of stored layer l in this micro-batch (§5.7)
            counts = torch.stack([a["counts"] for a in apps]).sum(0)
            prob_sum = torch.stack([a["prob_sum"] for a in apps]).sum(0)
            z_sum = torch.stack([a["z_sum"] for a in apps]).sum(0)
            n_tok = torch.stack([a["n_tokens"] for a in apps]).sum(0)
            lb, lz, f_mb = self.layers[l].ffn.aux_losses(counts, prob_sum, z_sum, n_tok, self.rho, update)
            lbs.append(lb)
            zs.append(lz)
            loads.append(f_mb)
        lb_loss, z_loss = torch.stack(lbs), torch.stack(zs)
        aux = self.lb_coef * lb_loss.sum() + self.z_coef * z_loss.sum()
        return RexOutput(hidden, aux, lb_loss, z_loss, torch.stack(loads), routing)


def lm_loss(model: RexModel, ids: torch.Tensor, cu: torch.Tensor, pos: torch.Tensor,
            T: Optional[int] = None) -> Tuple[torch.Tensor, RexOutput]:
    """Unchunked mean next-token cross-entropy (§5.8) for small inputs (tests, quick evals).
    Training must use the chunked head (train/losses.py) at real sizes."""
    out = model(ids, cu, pos, T)
    logits = model.logits(out.hidden)
    ce = F.cross_entropy(logits, next_token_targets(ids, cu), ignore_index=-100)
    return ce, out
