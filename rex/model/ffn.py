"""Dense SwiGLU feed-forward for the baselines (§5.7, last paragraph)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import trunc_normal_


class DenseFFN(nn.Module):
    """W_in [2H, d] (rows 0..H-1 gate, H..2H-1 up), W_out [d, H]; o = W_out(silu(g) ⊙ u)."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.w_in = nn.Linear(d, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # SPEC: §5.9 lists "attention and expert matrices"; dense FFN matrices are not named.
        # Using the same truncated normal (std 0.02, ±0.06) as the expert matrices they replace.
        trunc_normal_(self.w_in.weight)
        trunc_normal_(self.w_out.weight)

    def forward(self, n: torch.Tensor, return_routing: bool = False):
        g, u = self.w_in(n).split(self.hidden, dim=-1)
        return self.w_out(F.silu(g) * u), None
