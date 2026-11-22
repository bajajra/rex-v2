"""§10 evaluation probes: held-out loss (nats, bits per byte), forced-T losses, message ablation,
gate / β / pass-change / token-correlation statistics, α maps, block-sum and running-sum RMS,
pass overlap of expert sets, output-gate means, attention distance, loss by position bucket,
context truncation and the repeat probe.

All probes run without gradients in eval mode, so the model's load buffers are not touched.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.loader import MicroBatch, collate_sequences
from ..model.model import next_token_targets
from ..model.norm import nrm
from ..train.losses import HeadStats, bits_per_byte, chunked_lm_eval

POS_BUCKETS = (0, 1024, 8192, 65536, 262144)


@contextmanager
def eval_mode(model):
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was)


@contextmanager
def gate_override(model, mode):
    model.set_gate_override(mode)
    try:
        yield
    finally:
        model.set_gate_override(None)


def _autocast(device, precision):
    if precision == "bf16":
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
    return torch.autocast(device_type=torch.device(device).type, enabled=False)


def eval_loss(model, batches: Sequence[MicroBatch], *, chunk_tokens: int, T: Optional[int] = None,
              per_token: bool = False, device="cpu", precision="fp32") -> dict:
    st = HeadStats()
    nbytes = 0.0
    nll_all, pos_all = [], []
    with eval_mode(model):
        for mb in batches:
            mb = mb.to(device)
            with _autocast(device, precision):
                out = model(mb.ids, mb.cu, mb.pos, T)
                s, nll = chunked_lm_eval(model, out.hidden, next_token_targets(mb.ids, mb.cu),
                                         chunk_tokens=chunk_tokens, per_token=per_token)
            st.add(s)
            nbytes += mb.bytes_est
            if per_token:
                tgt = next_token_targets(mb.ids, mb.cu)
                keep = tgt != -100
                nll_all.append(nll[keep].cpu())
                pos_all.append(mb.pos[keep].cpu())
    nats = st.ce_sum / max(st.n_targets, 1)
    res = {"nats": nats, "bpb": bits_per_byte(st.ce_sum, nbytes), "n_targets": st.n_targets,
           "softcap_saturation": st.n_saturated / max(st.n_logits, 1)}
    if per_token:
        res["nll"] = torch.cat(nll_all) if nll_all else torch.zeros(0)
        res["pos"] = torch.cat(pos_all) if pos_all else torch.zeros(0, dtype=torch.long)
    return res


def position_buckets(nll: torch.Tensor, pos: torch.Tensor, buckets=POS_BUCKETS) -> Dict[str, float]:
    """Mean loss by position of the predicting token: 0-1k, 1k-8k, 8k-64k, 64k-262k (§10)."""
    out = {}
    for a, b in zip(buckets[:-1], buckets[1:]):
        m = (pos >= a) & (pos < b)
        out[f"{a}-{b}"] = float(nll[m].mean()) if bool(m.any()) else float("nan")
    return out


def forced_T_losses(model, batches, *, chunk_tokens, device="cpu", precision="fp32") -> Dict[str, float]:
    return {f"loss_T{t}": eval_loss(model, batches, chunk_tokens=chunk_tokens, T=t, device=device,
                                    precision=precision)["nats"]
            for t in range(1, model.n_passes + 1)}


def message_ablation(model, batches, *, chunk_tokens, device="cpu", precision="fp32") -> float:
    """Loss with γ forced to 0 in every GateMix (§10)."""
    with gate_override(model, "no_message"):
        return eval_loss(model, batches, chunk_tokens=chunk_tokens, device=device, precision=precision)["nats"]


def _jaccard(a: torch.Tensor, b: torch.Tensor, E: int) -> float:
    oa = torch.zeros(a.shape[0], E, dtype=torch.bool, device=a.device).scatter_(1, a, True)
    ob = torch.zeros(b.shape[0], E, dtype=torch.bool, device=b.device).scatter_(1, b, True)
    inter = (oa & ob).sum(-1).double()
    union = (oa | ob).sum(-1).double()
    return float((inter / union).mean())


def internal_stats(model, batches: Sequence[MicroBatch], device="cpu", precision="fp32") -> dict:
    """One recorded forward per batch: gate statistics (γ RMS, κ mean, β on V3) per reader type,
    core layer and pass; pass change and token correlation of V3; α maps; block-sum and
    running-sum RMS; output-gate means; pass overlap of expert sets."""
    acc = defaultdict(lambda: [0.0, 0.0])  # key -> [sum, count]

    def add(key, s, n):
        a = acc[key]
        a[0] += float(s)
        a[1] += float(n)

    alpha_sum: Dict[tuple, torch.Tensor] = {}
    alpha_n: Dict[tuple, float] = defaultdict(float)
    alpha_src: Dict[tuple, list] = {}
    state = {"prev_V3": None}

    def recorder(ev):
        k = ev["kind"]
        if k == "depthmix":
            key = (ev["reader"], ev["pass"])
            a = ev["alpha"].double().sum(0).cpu()
            alpha_sum[key] = alpha_sum.get(key, 0) + a
            alpha_n[key] += ev["alpha"].shape[0]
            alpha_src[key] = ev["sources"]
        elif k == "gatemix":
            l, p = ev["layer"], ev["pass"]
            s = ev["reader"].rsplit(".", 1)[-1]
            g, kap, beta = ev["gamma"].float(), ev["kappa"].float(), ev["beta"].float()
            add(f"gate/gamma_rms/{s}/L{l}/p{p}", g.pow(2).sum(), g.numel())
            add(f"gate/kappa_mean/{s}/L{l}/p{p}", kap.sum(), kap.numel())
            add(f"gate/beta_V3/{s}/L{l}/p{p}", beta[:, 1].sum(), beta.shape[0])
        elif k == "block":
            v = ev["value"].float()
            name = ev["name"] + ("" if ev["pass"] is None else f"[{ev['pass']}]")
            add(f"block_rms/{name}", v.pow(2).sum(), v.numel())
            if ev["name"] == "V3":
                u = torch.nn.functional.normalize(nrm(v, 1e-6), dim=-1)
                N = u.shape[0]
                if N > 1:
                    s = u.sum(0)
                    add(f"loop/token_corr/p{ev['pass']}", (s @ s - N) / (N * (N - 1)), 1)
                pv = state["prev_V3"]
                if pv is not None and ev["pass"] is not None and ev["pass"] > 1:
                    d = v - pv
                    cos = torch.nn.functional.cosine_similarity(v, pv, dim=-1)
                    add(f"loop/pass_change_cos/p{ev['pass']}", cos.sum(), cos.numel())
                    add(f"loop/pass_change_relnorm/p{ev['pass']}", d.norm() / pv.norm().clamp_min(1e-30), 1)
                state["prev_V3"] = v

    hooks = []
    calls = defaultdict(int)
    mcfg = model.mcfg

    def layer_hook(mod, args, output):
        l = mod.l
        calls[l] += 1
        p = calls[l] if mcfg.core_start <= l < mcfg.core_end else None
        R = output[0].float()
        add(f"running_rms/L{l}" + ("" if p is None else f"/p{p}"), R.pow(2).sum(), R.numel())

    def gate_hook_for(l):
        def h(mod, args, output):
            g = 2.0 * torch.sigmoid(output.float())
            add(f"attn/out_gate_mean/L{l}", g.sum(), g.numel())
        return h

    for layer in model.layers:
        if model.residual == "attnres":
            hooks.append(layer.register_forward_hook(layer_hook))
        if layer.attn.gate is not None:
            hooks.append(layer.attn.gate.register_forward_hook(gate_hook_for(layer.l)))

    overlap = defaultdict(lambda: [0.0, 0])
    try:
        with eval_mode(model):
            for mb in batches:
                mb = mb.to(device)
                calls.clear()
                state["prev_V3"] = None
                with _autocast(device, precision):
                    out = model(mb.ids, mb.cu, mb.pos, recorder=recorder, return_routing=model.is_moe)
                if model.is_moe:
                    E = mcfg.ffn.n_experts
                    by = defaultdict(dict)
                    for r in out.routing:
                        if r["pass"] is not None:
                            by[r["layer"]][r["pass"]] = r["top_i"]
                    for l, d in by.items():
                        ps = sorted(d)
                        for i in range(len(ps)):
                            for j in range(i + 1, len(ps)):
                                o = overlap[f"moe/pass_overlap/L{l}/p{ps[i]}-p{ps[j]}"]
                                o[0] += _jaccard(d[ps[i]], d[ps[j]], E)
                                o[1] += 1
    finally:
        for h in hooks:
            h.remove()

    res: Dict[str, object] = {}
    for k, (s, n) in acc.items():
        v = s / max(n, 1e-30)
        res[k] = math.sqrt(v) if ("rms" in k) else v
    for k, (s, n) in overlap.items():
        res[k] = s / max(n, 1)
    res["alpha"] = {f"{r}" + ("" if p is None else f"/p{p}"): {"sources": alpha_src[(r, p)],
                                                               "mean": (alpha_sum[(r, p)] / alpha_n[(r, p)]).tolist()}
                    for (r, p) in alpha_sum}
    return res


def attention_distance(model, batches: Sequence[MicroBatch], threshold: Optional[int] = None,
                       max_seq: int = 8192, device="cpu") -> Dict[str, float]:
    """Share of full-attention-layer attention mass at distance >= ``threshold`` (default: the
    window, 1,024), averaged over queries that can see that far; reference math, short sequences."""
    thr = threshold or model.mcfg.attn.window
    acc = defaultdict(lambda: [0.0, 0])
    calls = defaultdict(int)
    hooks = []

    def hook_for(l, attn):
        def h(mod, args, output):
            calls[l] += 1
            nQ, nK, nV, cu, pos = args[:5]
            q, k, _ = attn.project(nQ, nK, nV, pos)
            group = q.shape[1] // k.shape[1]
            bounds = cu.tolist()
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a <= thr or b - a > max_seq:
                    continue
                qf, kf = q[a:b].float(), k[a:b].float().repeat_interleave(group, dim=1)
                lg = torch.einsum("qhd,khd->hqk", qf, kf) / math.sqrt(q.shape[-1])
                rel = pos[a:b, None] - pos[None, a:b]
                lg = lg.masked_fill(~(rel >= 0)[None], float("-inf"))
                p = torch.softmax(lg, -1)
                far = (p * (rel >= thr)[None]).sum(-1)          # [H, n]
                sel = pos[a:b] >= thr
                o = acc[f"attn/far_mass/L{l}/app{calls[l]}"]
                o[0] += float(far[:, sel].mean(0).sum())
                o[1] += int(sel.sum())
        return h

    for layer in model.layers:
        if layer.attn.is_global:
            hooks.append(layer.attn.register_forward_hook(hook_for(layer.l, layer.attn)))
    try:
        with eval_mode(model):
            for mb in batches:
                calls.clear()
                mb = mb.to(device)
                model(mb.ids, mb.cu, mb.pos)
    finally:
        for h in hooks:
            h.remove()
    return {k: s / n for k, (s, n) in acc.items() if n > 0}


def _seq_losses(model, seq: np.ndarray, chunk_tokens: int, device, precision) -> torch.Tensor:
    mb = collate_sequences([seq]).to(device)
    with eval_mode(model), _autocast(device, precision):
        out = model(mb.ids, mb.cu, mb.pos)
        _, nll = chunked_lm_eval(model, out.hidden, next_token_targets(mb.ids, mb.cu),
                                 chunk_tokens=chunk_tokens, per_token=True)
    return nll.cpu()  # nll[i] = loss of predicting seq[i+1]


def context_truncation(model, docs: Sequence[np.ndarray], *, bos_id: int, contexts=(1024, 8192, 65536, None),
                       target_len: int = 512, max_total: int = 262144, chunk_tokens: int = 2048,
                       device="cpu", precision="fp32") -> Dict[str, float]:
    """Loss on the same ``target_len`` target tokens given the last c tokens of context (None =
    all available context, capped at ``max_total``). Each input starts with ``bos_id``."""
    # SPEC: §10 does not say whether truncated inputs start with bos; every training sequence
    # does (§6.5), so they do here.
    res = defaultdict(list)
    finite = [c for c in contexts if c is not None]
    need = (max(finite) if finite else 0) + target_len
    for doc in docs:
        e = min(len(doc), max_total - 1)
        if e < need:
            continue
        for c in contexts:
            a = 0 if c is None else e - target_len - c
            seq = np.concatenate([[bos_id], doc[a:e]]).astype(np.int64)
            nll = _seq_losses(model, seq, chunk_tokens, device, precision)
            res["all" if c is None else str(c)].append(float(nll[-target_len - 1:-1].mean()))
    return {f"trunc/ctx_{k}": float(np.mean(v)) for k, v in res.items()}


def repeat_probe(model, passages: Sequence[np.ndarray], filler: np.ndarray, *, bos_id: int,
                 gaps=(2048, 16384, 131072), chunk_tokens: int = 2048, device="cpu", precision="fp32") -> Dict[str, float]:
    """A passage repeated after ``gap`` unrelated tokens: loss on the repeat vs first occurrence."""
    res = defaultdict(list)
    for P in passages:
        n = len(P)
        for g in gaps:
            if len(filler) < g:
                continue
            seq = np.concatenate([[bos_id], P, filler[:g], P]).astype(np.int64)
            nll = _seq_losses(model, seq, chunk_tokens, device, precision)
            first = nll[1:n].mean()                        # predicting P[1:] in the first copy
            s2 = 1 + n + g                                 # index of the second copy's P[0]
            rep = nll[s2: s2 + n - 1].mean()               # predicting P[1:] in the repeat
            res[f"repeat/first_gap{g}"].append(float(first))
            res[f"repeat/repeat_gap{g}"].append(float(rep))
    return {k: float(np.mean(v)) for k, v in res.items()}


def evaluate(model, batches: Sequence[MicroBatch], *, chunk_tokens: int, device="cpu", precision="fp32",
             attn_batches: Optional[Sequence[MicroBatch]] = None) -> dict:
    """The periodic "eval" items of §10 on fixed held-out batches."""
    base = eval_loss(model, batches, chunk_tokens=chunk_tokens, per_token=True, device=device, precision=precision)
    res = {"val/nats": base["nats"], "val/bpb": base["bpb"], "val/n_targets": base["n_targets"],
           "val/softcap_saturation": base["softcap_saturation"]}
    res.update({f"val/pos_bucket/{k}": v for k, v in position_buckets(base["nll"], base["pos"]).items()})
    if model.residual == "attnres" and model.n_passes > 1:
        res.update({f"val/{k}": v for k, v in forced_T_losses(model, batches, chunk_tokens=chunk_tokens,
                                                              device=device, precision=precision).items()})
        res["val/loss_no_message"] = message_ablation(model, batches, chunk_tokens=chunk_tokens, device=device,
                                                      precision=precision)
    res.update(internal_stats(model, batches, device=device, precision=precision))
    res.update(attention_distance(model, attn_batches if attn_batches is not None else batches, device=device))
    return res
