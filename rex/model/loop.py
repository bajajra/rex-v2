"""GateMix: cross-pass message reader with rank-r keep/gain gates (§5.5)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .norm import nrm

# Override modes (used by T4 and the §10 message-ablation probe):
#   None          normal operation
#   "no_message"  γ forced to 0 (κ as learned)
#   "identity"    γ forced to 0 and κ forced to 1, i.e. GateMix(x) = x
GATE_OVERRIDES = (None, "no_message", "identity")


class GateMix(nn.Module):
    def __init__(self, d: int, rank: int, eps: float):
        super().__init__()
        self.d, self.rank, self.eps = d, rank, eps
        self.u = nn.Parameter(torch.zeros(d))
        self.P = nn.Linear(2 * d, rank, bias=False)  # weight [r, 2d]
        self.F = nn.Linear(rank, d, bias=False)      # weight [d, r]
        self.G = nn.Linear(rank, d, bias=False)      # weight [d, r]
        self.override: Optional[str] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # §5.5 / §5.9: u = 0, F = 0, G = 0, P ~ N(0, 0.02^2) -> κ = 1, γ = 0 exactly.
        nn.init.zeros_(self.u)
        nn.init.normal_(self.P.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.F.weight)
        nn.init.zeros_(self.G.weight)

    def forward(self, x: torch.Tensor, prev: Tuple[torch.Tensor, torch.Tensor], return_stats: bool = False):
        if self.override == "identity":
            return (x, None) if return_stats else x
        v2, v3 = prev  # already nrm(V2), nrm(V3)
        with torch.autocast(x.device.type, enabled=False):
            vs = torch.stack([v2, v3], dim=0).to(torch.promote_types(v2.dtype, x.dtype))
            b = torch.einsum("jnd,d->nj", vs, self.u.to(vs.dtype)) / math.sqrt(self.d)
            beta = torch.softmax(b, dim=-1)                       # [N, 2]
            m = torch.einsum("nj,jnd->nd", beta, vs)
            zin = torch.cat([nrm(x, self.eps), nrm(m, self.eps)], dim=-1)
        z = self.P(zin.to(self.P.weight.dtype))
        kappa = 1.0 - torch.tanh(self.F(z))
        gamma = torch.tanh(self.G(z))
        if self.override == "no_message":
            gamma = torch.zeros_like(gamma)
        out = kappa * x + gamma * m
        if return_stats:
            return out, {"beta": beta.detach(), "kappa": kappa.detach(), "gamma": gamma.detach()}
        return out
