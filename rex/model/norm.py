"""nrm and RMSNorm (§5.1)."""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_dtype(x: torch.Tensor) -> torch.dtype:
    """float32 for half/bf16/float32 inputs; float64 stays float64 (used by the float64 tests)."""
    return torch.promote_types(x.dtype, torch.float32)


def nrm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Parameter-free RMS normalisation over the last dim, computed in (at least) float32.

    Returns the computation dtype (float32, or float64 for float64 inputs).
    """
    xf = x.to(compute_dtype(x))
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    """g ⊙ nrm(x), g initialised to 1."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nrm(x, self.eps)
        return (y * self.weight.to(y.dtype)).to(torch.promote_types(x.dtype, self.weight.dtype))
