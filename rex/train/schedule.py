"""LR schedule and curriculum bookkeeping (§6.5, §9). Tokens are the clock.

lr = min(1, (step+1)/warmup_steps) · [lo + ½(peak − lo)(1 + cos(π·p))],
lo = lr_min_ratio·peak, p = min(1, tokens consumed before the step / total_tokens), step 0-based.

The run ends at the first step boundary where ``total_tokens`` have been consumed.
``floor(total_tokens / tokens_per_step)`` is the nominal step count, for planning only (§9:
22,888 for 6B, 190 for the smoke tier); the actual count is slightly lower because steps overshoot.
"""

from __future__ import annotations

import math


def total_steps(train_cfg) -> int:
    """Nominal step count (planning only)."""
    return max(1, int(train_cfg.total_tokens // train_cfg.tokens_per_step))


def progress(tokens_before: float, train_cfg) -> float:
    return min(1.0, float(tokens_before) / float(train_cfg.total_tokens))


def lr_at(step: int, tokens_before: float, train_cfg) -> float:
    """LR for optimiser step ``step`` (0-based) given the tokens consumed before it."""
    t = train_cfg
    peak, lo = t.lr, t.lr * t.lr_min_ratio
    warm = min(1.0, (step + 1) / t.warmup_steps) if t.warmup_steps > 0 else 1.0
    p = progress(tokens_before, t)
    return warm * (lo + 0.5 * (peak - lo) * (1.0 + math.cos(math.pi * p)))


def stage_index(curriculum, total_tokens: float, tokens_consumed: float) -> int:
    frac = tokens_consumed / total_tokens
    for i, s in enumerate(curriculum):
        if frac < s.until:
            return i
    return len(curriculum) - 1
