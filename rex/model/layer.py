"""Layer_l (§5.4): attention sublayer and FFN sublayer with depth readers and cross-pass gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..config import RexConfig
from .attention import Attention, KVEntry
from .attnres import DepthMix
from .ffn import DenseFFN
from .loop import GateMix
from .moe import MoE
from .norm import RMSNorm, nrm

Prev = Optional[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class AppContext:
    """Per-application bookkeeping that is not part of the differentiable state."""
    pass_idx: Optional[int] = None            # 1..T for core layers, None for prelude/coda
    src_names: List[str] = field(default_factory=list)  # names of the completed sources
    recorder: Optional[Callable[[dict], None]] = None
    cache: Optional[KVEntry] = None
    return_routing: bool = False
    tok_chunk: Optional[int] = None           # nested token-chunk checkpoint (only inside a segment)


def _sum_stats(parts: List[Optional[Dict[str, torch.Tensor]]]):
    if parts[0] is None:
        return None
    out = {}
    for k in parts[0]:
        vals = [p[k] for p in parts]
        out[k] = torch.cat(vals, 0) if k == "top_i" else torch.stack(vals).sum(0)
    return out


class Layer(nn.Module):
    def __init__(self, cfg: RexConfig, l: int):
        super().__init__()
        m = cfg.model
        d = m.d_model
        self.l = l
        self.eps = m.rms_eps
        self.residual = m.residual
        self.is_global = l in m.attn.global_layers
        self.is_core = m.core_start <= l < m.core_end
        self.attn_norm = RMSNorm(d, m.rms_eps)  # one gain shared by the Q, K, V streams [J]
        self.ffn_norm = RMSNorm(d, m.rms_eps)
        self.attn = Attention(m, l, cfg.train.max_len)
        if m.ffn.type == "moe":
            self.ffn = MoE(d, m.ffn, cfg.loss)
        else:
            self.ffn = DenseFFN(d, m.ffn.dense_hidden)

        # Depth readers (attnres only). Layer 0's attention has a single source: no reader.
        self.mix_attn = self.mix_q = self.mix_k = self.mix_v = self.mix_moe = None
        if self.residual == "attnres":
            if self.is_global:
                self.mix_q, self.mix_k, self.mix_v = DepthMix(d), DepthMix(d), DepthMix(d)
            elif l != 0:
                self.mix_attn = DepthMix(d)
            self.mix_moe = DepthMix(d)

        # Cross-pass readers, per core layer, separate for Q, K, V, M (only if the loop runs).
        self.cross = None
        if self.residual == "attnres" and self.is_core and m.n_passes > 1:
            self.cross = nn.ModuleDict({s: GateMix(d, m.gate_rank, m.rms_eps) for s in "QKVM"})

    # ------------------------------------------------------------------ helpers
    def _name(self, s: str) -> str:
        return f"L{self.l}.{s}"

    def _mix(self, mod: DepthMix, name: str, s_hat: List[torch.Tensor], names: List[str], ctx: AppContext):
        if ctx.recorder is None:
            return mod(s_hat)
        out, alpha = mod(s_hat, return_weights=True)
        ctx.recorder({"kind": "depthmix", "reader": name, "layer": self.l, "pass": ctx.pass_idx,
                      "sources": list(names), "alpha": alpha.detach()})
        return out

    def _gate(self, s: str, x: torch.Tensor, prev: Prev, ctx: AppContext):
        if prev is None:
            return x
        g = self.cross[s]
        if ctx.recorder is None:
            return g(x, prev)
        out, st = g(x, prev, return_stats=True)
        ctx.recorder({"kind": "gatemix", "reader": self._name(f"cross.{s}"), "layer": self.l,
                      "pass": ctx.pass_idx, **(st or {})})
        return out

    # ------------------------------------------------------------------ token-local parts
    def pre_attention(self, done_hat: List[torch.Tensor], R: Optional[torch.Tensor], prev: Prev, ctx: AppContext):
        """DepthMix (+ GateMix) + RMSNorm for the three attention input streams."""
        s_hat = list(done_hat) + ([nrm(R, self.eps)] if R is not None else [])
        names = list(ctx.src_names) + (["R"] if R is not None else [])
        if self.is_global:
            xQ = self._mix(self.mix_q, self._name("attn.Q"), s_hat, names, ctx)
            xK = self._mix(self.mix_k, self._name("attn.K"), s_hat, names, ctx)
            xV = self._mix(self.mix_v, self._name("attn.V"), s_hat, names, ctx)
        elif self.mix_attn is not None:
            xQ = xK = xV = self._mix(self.mix_attn, self._name("attn"), s_hat, names, ctx)
        else:
            # SPEC: §5.4 "Layer 0 has no q^A (single source, the mix is the identity)":
            # a one-source DepthMix returns the normalised source.
            if len(s_hat) != 1:
                raise RuntimeError("attention sublayer without a reader must have exactly one source")
            xQ = xK = xV = s_hat[0]
        if prev is not None:
            xQ, xK, xV = self._gate("Q", xQ, prev, ctx), self._gate("K", xK, prev, ctx), self._gate("V", xV, prev, ctx)
        return self.attn_norm(xQ), self.attn_norm(xK), self.attn_norm(xV)

    def post_attention(self, done_hat: List[torch.Tensor], R: torch.Tensor, prev: Prev, ctx: AppContext):
        """FFN sublayer: x = DepthMix(q^M, done + [R]); GateMix^M; FFN(RMSNorm(x)); return R + o_ffn."""
        s_hat = list(done_hat) + [nrm(R, self.eps)]
        x = self._mix(self.mix_moe, self._name("moe"), s_hat, list(ctx.src_names) + ["R"], ctx)
        x = self._gate("M", x, prev, ctx)
        o, stats = self.ffn(self.ffn_norm(x), return_routing=ctx.return_routing)
        return R + o, stats

    # ------------------------------------------------------------------ forward
    def forward(self, done_hat: List[torch.Tensor], R: Optional[torch.Tensor], prev: Prev,
                cu: torch.Tensor, pos: torch.Tensor, ctx: AppContext):
        """One application of Layer_l in attnres mode. Returns (R_new, ffn_stats)."""
        if ctx.tok_chunk:
            return self._forward_tok_chunked(done_hat, R, prev, cu, pos, ctx)
        nQ, nK, nV = self.pre_attention(done_hat, R, prev, ctx)
        o_attn = self.attn(nQ, nK, nV, cu, pos, cache=ctx.cache)
        R = o_attn if R is None else R + o_attn
        return self.post_attention(done_hat, R, prev, ctx)

    def _forward_tok_chunked(self, done_hat, R, prev, cu, pos, ctx: AppContext):
        """§6.4 tok_chunk: token-local parts per chunk under a nested checkpoint;
        attention runs at full length."""
        N, c = done_hat[0].shape[0], ctx.tok_chunk
        bounds = [(a, min(a + c, N)) for a in range(0, N, c)]

        def sl(t, a, b):
            return None if t is None else t[a:b]

        def slp(p, a, b):
            return None if p is None else (p[0][a:b], p[1][a:b])

        def pre_fn(dh, r, pr):
            return self.pre_attention(dh, r, pr, ctx)

        parts = [checkpoint(pre_fn, [s[a:b] for s in done_hat], sl(R, a, b), slp(prev, a, b), use_reentrant=False)
                 for a, b in bounds]
        nQ, nK, nV = (torch.cat([p[i] for p in parts], 0) for i in range(3))
        o_attn = self.attn(nQ, nK, nV, cu, pos, cache=ctx.cache)
        R = o_attn if R is None else R + o_attn

        def post_fn(dh, r, pr):
            return self.post_attention(dh, r, pr, ctx)

        outs = [checkpoint(post_fn, [s[a:b] for s in done_hat], R[a:b], slp(prev, a, b), use_reentrant=False)
                for a, b in bounds]
        return torch.cat([o[0] for o in outs], 0), _sum_stats([o[1] for o in outs])

    def forward_stream(self, x: torch.Tensor, cu: torch.Tensor, pos: torch.Tensor, ctx: AppContext):
        """§11 ``residual: stream``: x += Attention(RMSNorm(x)); x += FFN(RMSNorm(x))."""
        n = self.attn_norm(x)
        x = x + self.attn(n, n, n, cu, pos, cache=ctx.cache)
        o, stats = self.ffn(self.ffn_norm(x), return_routing=ctx.return_routing)
        return x + o, stats
