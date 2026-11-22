"""Rotary position embedding (§5.6).

theta_i = base^(-i / (head_dim/2)), i = 0..head_dim/2-1 (the spec writes base^(-i/64) for
head_dim 128). Angle tables are computed in float64 for every position up to max_len-1 and
only then cast, because float32 angles lose the high frequencies at large positions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

_TABLES: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


def rope_tables_f64(base: float, head_dim: int, max_len: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [max_len, head_dim/2] in float64."""
    half = head_dim // 2
    i = torch.arange(half, dtype=torch.float64, device=device)
    theta = base ** (-i / half)
    pos = torch.arange(max_len, dtype=torch.float64, device=device)
    ang = pos[:, None] * theta[None, :]
    return torch.cos(ang), torch.sin(ang)


def rope_tables(base: float, head_dim: int, max_len: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached cos/sin tables cast to ``dtype`` (process-wide cache, not module buffers)."""
    key = (float(base), head_dim, max_len, str(device), dtype)
    t = _TABLES.get(key)
    if t is None:
        c, s = rope_tables_f64(base, head_dim, max_len, device=device)
        t = (c.to(dtype), s.to(dtype))
        _TABLES[key] = t
    return t


def apply_rope(x: torch.Tensor, pos: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [N, H, dh]; pos: [N] int64. Halves (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)."""
    if pos.numel() and int(pos.max()) >= cos.shape[0]:
        raise ValueError(f"position {int(pos.max())} exceeds RoPE table length {cos.shape[0]}")
    ct = torch.promote_types(x.dtype, cos.dtype)
    c = cos[pos].unsqueeze(1).to(ct)
    s = sin[pos].unsqueeze(1).to(ct)
    x1, x2 = x.to(ct).chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)
