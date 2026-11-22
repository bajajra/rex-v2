"""Routed SwiGLU mixture of experts (§5.7): router, sorted dispatch, dense_ref, aux statistics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FFNConfig, LossConfig
from .attention import trunc_normal_
from .norm import compute_dtype


class MoE(nn.Module):
    """Parameters: W_r [E, d], b_r [E], W_in [E, 2h, d] (rows 0..h-1 gate, h..2h-1 up),
    W_out [E, d, h]. Buffers: ``load_ema`` (f̃ of the smoothed balance loss, init 1/E) and
    ``balance_bias`` (c of the ``balance: bias`` ablation, init 0; selection only).

    ``forward(n)`` returns ``(o_moe, stats)``. ``stats`` holds per-application sums so that a
    stored layer can pool all its applications (§5.7):
      counts   [E]  assignments per expert (no grad)
      prob_sum [E]  Σ_tokens softmax(logits)          (grad)
      z_sum    []   Σ_tokens logsumexp(logits)^2      (grad)
      maxw_sum []   Σ_tokens max_j w[j]               (no grad; router confidence, §10)
      n_tokens []   number of tokens
      top_i    [N, k] (only if ``return_routing``)
    """

    def __init__(self, d: int, cfg: FFNConfig, loss_cfg: LossConfig):
        super().__init__()
        self.d, self.E, self.k, self.h = d, cfg.n_experts, cfg.top_k, cfg.expert_hidden
        self.route_scale = cfg.route_scale
        self.dispatch = cfg.dispatch
        self.use_balance_bias = loss_cfg.balance == "bias"
        self.router_weight = nn.Parameter(torch.empty(self.E, d))
        self.router_bias = nn.Parameter(torch.zeros(self.E)) if cfg.router_bias else None
        self.w_in = nn.Parameter(torch.empty(self.E, 2 * self.h, d))
        self.w_out = nn.Parameter(torch.empty(self.E, d, self.h))
        self.register_buffer("load_ema", torch.full((self.E,), 1.0 / self.E))
        self.register_buffer("balance_bias", torch.zeros(self.E))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router_weight, mean=0.0, std=0.02)
        if self.router_bias is not None:
            nn.init.zeros_(self.router_bias)
        trunc_normal_(self.w_in)
        trunc_normal_(self.w_out)
        with torch.no_grad():
            self.load_ema.fill_(1.0 / self.E)
            self.balance_bias.zero_()

    # ------------------------------------------------------------------ routing
    def route(self, n: torch.Tensor):
        """Returns (logits [N,E] fp32, top_i [N,k], w [N,k]) with Σ_j w = route_scale."""
        ct = compute_dtype(n)
        with torch.autocast(n.device.type, enabled=False):
            logits = n.to(ct) @ self.router_weight.to(ct).t()
            if self.router_bias is not None:
                logits = logits + self.router_bias.to(ct)
            sel = logits + self.balance_bias.to(ct) if self.use_balance_bias else logits
            top_i = torch.topk(sel.detach(), self.k, dim=-1).indices      # no noise
            top_v = logits.gather(1, top_i)                                 # c used for selection only
            w = self.route_scale * torch.softmax(top_v, dim=-1)
        return logits, top_i, w

    # ------------------------------------------------------------------ experts
    def _expert(self, e: int, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        gu = F.linear(x, self.w_in[e])
        g, u = gu.split(self.h, dim=-1)
        hid = F.silu(g) * u * scale[:, None].to(gu.dtype)
        return F.linear(hid, self.w_out[e])

    def _sorted(self, n, top_i, w):
        N = n.shape[0]
        flat_e = top_i.reshape(-1)
        tok = torch.arange(N, device=n.device).repeat_interleave(self.k)
        order = torch.argsort(flat_e, stable=True)
        counts = torch.bincount(flat_e, minlength=self.E)
        tok_o = tok[order]
        xs = n[tok_o]
        ws = w.reshape(-1)[order]
        ys = []
        start = 0
        for e, c in enumerate(counts.tolist()):  # one host sync per application (§9)
            if c > 0:
                ys.append(self._expert(e, xs[start:start + c], ws[start:start + c]))
            start += c
        ys = torch.cat(ys, 0)
        out = torch.zeros(N, self.d, dtype=ys.dtype, device=n.device)
        return out.index_add(0, tok_o, ys)

    def _dense_ref(self, n, top_i, w):
        """Every expert on every token, masked by the (scattered) gate weights. Tests only."""
        N = n.shape[0]
        wfull = torch.zeros(N, self.E, dtype=w.dtype, device=n.device).scatter(1, top_i, w)
        out = 0
        for e in range(self.E):
            out = out + self._expert(e, n, wfull[:, e])
        return out

    def forward(self, n: torch.Tensor, return_routing: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits, top_i, w = self.route(n)
        out = self._sorted(n, top_i, w) if self.dispatch == "sorted" else self._dense_ref(n, top_i, w)
        with torch.autocast(n.device.type, enabled=False):
            probs = torch.softmax(logits, dim=-1)
            stats = {
                "counts": torch.bincount(top_i.reshape(-1), minlength=self.E).to(logits.dtype),
                "prob_sum": probs.sum(0),
                "z_sum": torch.logsumexp(logits, dim=-1).pow(2).sum(),
                "maxw_sum": w.detach().max(dim=-1).values.sum(),
                "n_tokens": torch.tensor(float(n.shape[0]), dtype=logits.dtype, device=n.device),
            }
        if return_routing:
            stats["top_i"] = top_i
        return out, stats

    # ------------------------------------------------------------------ balancing
    def aux_losses(self, counts, prob_sum, z_sum, n_tokens, rho: float, update: bool):
        """L_LB and L_Z for this stored layer from statistics pooled over all its applications.

        f_mb = counts / (N_apps * k); f̃ <- ρ f̃ + (1-ρ) f_mb; L_LB = E Σ f̃ P̄; L_Z = mean lse².
        The buffer is written only when ``update`` (training forward); otherwise the same
        updated value is used without being stored.
        """
        f_mb = (counts / (n_tokens * self.k)).detach()
        f_s = rho * self.load_ema.to(f_mb.dtype) + (1.0 - rho) * f_mb
        if update:
            with torch.no_grad():
                self.load_ema.copy_(f_s.to(self.load_ema.dtype))
        p_bar = prob_sum / n_tokens
        l_lb = self.E * (f_s * p_bar).sum()
        l_z = z_sum / n_tokens
        return l_lb, l_z, f_mb

    @torch.no_grad()
    def update_balance_bias(self, f_step: torch.Tensor, speed: float) -> None:
        """``balance: bias`` ablation: after each optimiser step, c_i += speed * sign(1/E - f_step[i])."""
        self.balance_bias.add_(speed * torch.sign(1.0 / self.E - f_step.to(self.balance_bias)))
