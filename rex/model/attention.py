"""Attention sublayer (§5.6): projections, QK treatment, RoPE, length scaling, output gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attn_backends import attend, varlen_attention
from .norm import nrm
from .rope import apply_rope, rope_tables


def trunc_normal_(w: torch.Tensor, std: float = 0.02) -> None:
    """§5.9: truncated normal, std 0.02, cut at ±0.06 (3 std)."""
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)


@dataclass
class KVEntry:
    """Inference cache of one layer application (§5.10): post-norm/RoPE K, V and positions."""
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, max_len: int):
        super().__init__()
        a = cfg.attn
        d = cfg.d_model
        self.d, self.dh, self.hq, self.hkv = d, a.head_dim, a.n_q_heads, a.n_kv_heads
        self.eps = cfg.rms_eps
        self.is_global = layer_idx in a.global_layers
        self.window = None if self.is_global else a.window
        self.use_rope = (not self.is_global) or a.global_pos == "rope"
        self.rope_base = a.global_rope_base if self.is_global else a.rope_base
        self.max_len = max_len
        self.qk_mode = a.qk_mode
        self.qk_scale_init = a.qk_scale_init
        self.alpha, self.n_floor = a.log_scale_alpha, a.log_scale_n_floor
        self.backend = a.backend

        self.wq = nn.Linear(d, self.hq * self.dh, bias=False)
        self.wk = nn.Linear(d, self.hkv * self.dh, bias=False)
        self.wv = nn.Linear(d, self.hkv * self.dh, bias=False)
        self.wo = nn.Linear(self.hq * self.dh, d, bias=False)
        # qk_mode l2clip: "s is then fixed to 1" -> kept as a frozen parameter of ones so the
        # parameter count does not depend on the ablation switch.
        self.qk_scale = nn.Parameter(torch.empty(self.hq), requires_grad=(a.qk_mode == "norm"))
        self.gate = nn.Linear(d, self.hq, bias=True) if a.out_gate else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.wq, self.wk, self.wv, self.wo):
            trunc_normal_(lin.weight)
        with torch.no_grad():
            self.qk_scale.fill_(self.qk_scale_init if self.qk_mode == "norm" else 1.0)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def _qk(self, x: torch.Tensor) -> torch.Tensor:
        if self.qk_mode == "norm":
            return nrm(x, self.eps).to(x.dtype)
        c = 2.0 * math.sqrt(self.dh)
        n = x.float().norm(dim=-1, keepdim=True)
        return x * torch.clamp(c / n.clamp_min(1e-30), max=1.0).to(x.dtype)

    def project(self, nQ, nK, nV, pos):
        N = nQ.shape[0]
        q = self.wq(nQ).view(N, self.hq, self.dh)
        k = self.wk(nK).view(N, self.hkv, self.dh)
        v = self.wv(nV).view(N, self.hkv, self.dh)
        q, k = self._qk(q), self._qk(k)
        if self.use_rope:
            cos, sin = rope_tables(self.rope_base, self.dh, self.max_len, q.device,
                                   torch.promote_types(q.dtype, torch.float32))
            q, k = apply_rope(q, pos, cos, sin), apply_rope(k, pos, cos, sin)
        q = q * self.qk_scale.to(q.dtype)[None, :, None]
        if self.is_global and self.alpha != 0.0:
            p = (pos + 1).to(torch.float64).clamp_min(self.n_floor)
            lam = 1.0 + self.alpha * torch.log(p / self.n_floor)
            q = q * lam.to(q.dtype)[:, None, None]
        return q, k, v

    def forward(self, nQ, nK, nV, cu: torch.Tensor, pos: torch.Tensor,
                cache: Optional[KVEntry] = None) -> torch.Tensor:
        N = nQ.shape[0]
        q, k, v = self.project(nQ, nK, nV, pos)
        if cache is None:
            a = varlen_attention(self.backend, q, k, v, cu, pos, self.window)
        else:
            # Incremental decoding of a single sequence (reference math).
            if cache.k is not None:
                k_all = torch.cat([cache.k, k], 0)
                v_all = torch.cat([cache.v, v], 0)
                p_all = torch.cat([cache.pos, pos], 0)
            else:
                k_all, v_all, p_all = k, v, pos
            a = attend(q, k_all, v_all, pos, p_all, self.window)
            keep = k_all.shape[0] if self.window is None else min(self.window, k_all.shape[0])
            cache.k, cache.v, cache.pos = k_all[-keep:], v_all[-keep:], p_all[-keep:]
        if self.gate is not None:
            g = 2.0 * torch.sigmoid(self.gate(nQ))
            a = a * g.to(a.dtype)[:, :, None]
        return self.wo(a.reshape(N, self.hq * self.dh))
