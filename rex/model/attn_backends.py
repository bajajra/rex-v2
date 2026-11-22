"""Attention kernels over the packed layout (§5.6 "Backends").

* ``reference``: per sequence, explicit softmax(QK^T / sqrt(dh) + mask) with a boolean
  [n_q, n_k] mask, in float32 (float64 if the inputs are float64, for the float64 tests).
* ``flash``: ``flash_attn.flash_attn_varlen_func``. Optional dependency, CUDA only. When it is
  unavailable (e.g. on CPU) the reference kernel is used and a warning is emitted once.
  The flash path is untested in this repository's CPU environment (see T11).

Visibility: token i sees token j iff both are in the same sequence and
pos_i - window < pos_j <= pos_i (window = None means unlimited).
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch

try:  # pragma: no cover - not installable on CPU
    from flash_attn import flash_attn_varlen_func as _flash_varlen
except Exception:  # noqa: BLE001
    _flash_varlen = None

_warned = False


def flash_available() -> bool:
    return _flash_varlen is not None and torch.cuda.is_available()


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           pos_q: torch.Tensor, pos_k: torch.Tensor, window: Optional[int]) -> torch.Tensor:
    """One sequence. q [nq, Hq, dh]; k, v [nk, Hkv, dh]; query head h uses KV head h // (Hq/Hkv)."""
    hq, hkv, dh = q.shape[1], k.shape[1], q.shape[2]
    group = hq // hkv
    ct = torch.promote_types(q.dtype, torch.float32)
    with torch.autocast(q.device.type, enabled=False):
        qf, kf, vf = q.to(ct), k.to(ct), v.to(ct)
        if group > 1:
            kf = kf.repeat_interleave(group, dim=1)
            vf = vf.repeat_interleave(group, dim=1)
        logits = torch.einsum("qhd,khd->hqk", qf, kf) / math.sqrt(dh)
        rel = pos_q[:, None] - pos_k[None, :]
        mask = rel >= 0
        if window is not None:
            mask = mask & (rel < window)
        logits = logits.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(logits, dim=-1)
        out = torch.einsum("hqk,khd->qhd", p, vf)
    return out.to(q.dtype)


def reference_varlen(q, k, v, cu: torch.Tensor, pos: torch.Tensor, window: Optional[int]) -> torch.Tensor:
    bounds = cu.tolist()
    outs = [attend(q[a:b], k[a:b], v[a:b], pos[a:b], pos[a:b], window)
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return torch.cat(outs, dim=0)


def flash_varlen(q, k, v, cu: torch.Tensor, pos: torch.Tensor, window: Optional[int]) -> torch.Tensor:
    global _warned
    if not (flash_available() and q.is_cuda):
        if not _warned:
            warnings.warn("flash-attn unavailable (or tensors not on CUDA); using the reference attention backend")
            _warned = True
        return reference_varlen(q, k, v, cu, pos, window)
    # Untested here (no GPU). The packed layout assumes pos runs 0..n-1 inside each sequence.
    dt = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    cu32 = cu.to(torch.int32)
    max_len = int((cu32[1:] - cu32[:-1]).max())
    ws = (window - 1, 0) if window is not None else (-1, -1)
    out = _flash_varlen(q.to(dt), k.to(dt), v.to(dt), cu32, cu32, max_len, max_len,
                        softmax_scale=1.0 / math.sqrt(q.shape[-1]), causal=True, window_size=ws)
    return out.to(q.dtype)


def varlen_attention(backend: str, q, k, v, cu, pos, window):
    if backend == "reference":
        return reference_varlen(q, k, v, cu, pos, window)
    if backend == "flash":
        return flash_varlen(q, k, v, cu, pos, window)
    raise ValueError(backend)
