"""DepthMix: the depth-wise attention residual reader (§5.3)."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn

from .norm import nrm


def stack_normalised(sources_hat: Sequence[torch.Tensor]) -> torch.Tensor:
    """[n, N, d] from already-normalised sources, in a common float dtype."""
    dt = sources_hat[0].dtype
    for s in sources_hat[1:]:
        dt = torch.promote_types(dt, s.dtype)
    return torch.stack([s.to(dt) for s in sources_hat], dim=0)


class DepthMix(nn.Module):
    """Softmax over depth sources with a learned query q (init 0 -> uniform average).

    ``forward`` takes *already normalised* sources ŝ_j = nrm(s_j); callers normalise each
    completed block sum once and reuse it. Values are the normalised sources
    (attn_res_normalize_values). Computed with autocast disabled (float32 softmax, §9).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.q = nn.Parameter(torch.zeros(d))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.q)

    def weights(self, s_hat: torch.Tensor) -> torch.Tensor:
        """s_hat: [n, N, d] -> α: [N, n]."""
        with torch.autocast(s_hat.device.type, enabled=False):
            a = torch.einsum("jnd,d->nj", s_hat, self.q.to(s_hat.dtype)) / math.sqrt(self.d)
            return torch.softmax(a, dim=-1)

    def forward(self, sources_hat: List[torch.Tensor], return_weights: bool = False):
        s_hat = stack_normalised(sources_hat)
        alpha = self.weights(s_hat)
        with torch.autocast(s_hat.device.type, enabled=False):
            out = torch.einsum("nj,jnd->nd", alpha, s_hat)
        return (out, alpha) if return_weights else out


def depth_mix(q: torch.Tensor, sources: List[torch.Tensor], eps: float) -> torch.Tensor:
    """Functional form taking raw sources (normalises them); used by tests."""
    s_hat = stack_normalised([nrm(s, eps) for s in sources])
    a = torch.einsum("jnd,d->nj", s_hat, q.to(s_hat.dtype)) / math.sqrt(q.shape[-1])
    alpha = torch.softmax(a, dim=-1)
    return torch.einsum("nj,jnd->nd", alpha, s_hat)
