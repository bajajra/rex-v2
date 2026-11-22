"""LR schedule and curriculum bookkeeping (§6.5, §9).

LR: linear warm-up over ``warmup_steps``, then cosine from the peak to ``lr_min_ratio`` x peak
at the last step. Steps: ``floor(total_tokens / tokens_per_step)`` (§9: 22,888 steps for 6B;
§13: 190 steps for the 50M smoke tier).
"""

from __future__ import annotations

import math


def total_steps(train_cfg) -> int:
    return max(1, int(train_cfg.total_tokens // train_cfg.tokens_per_step))


def lr_at(step: int, train_cfg, n_steps: int | None = None) -> float:
    """LR for optimiser step ``step`` (0-based)."""
    t = train_cfg
    n = total_steps(t) if n_steps is None else n_steps
    peak, lo = t.lr, t.lr * t.lr_min_ratio
    w = t.warmup_steps
    if w > 0 and step < w:
        # SPEC: §9 "linear warm-up 1,000 steps" does not give the starting value; reading taken:
        # lr = peak·(step+1)/warmup, so step 0 is not a wasted zero-LR step.
        return peak * (step + 1) / w
    last = n - 1
    if last <= w:
        return peak if step < last else lo
    prog = min(1.0, (step - w) / (last - w))
    return lo + 0.5 * (peak - lo) * (1.0 + math.cos(math.pi * prog))


def stage_index(curriculum, total_tokens: float, tokens_consumed: float) -> int:
    frac = tokens_consumed / total_tokens
    for i, s in enumerate(curriculum):
        if frac < s.until:
            return i
    return len(curriculum) - 1
