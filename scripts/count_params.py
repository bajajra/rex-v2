#!/usr/bin/env python
"""Reproduce the §3 parameter table from a config, and check it against the instantiated model.

    python scripts/count_params.py configs/rex-v2-base.yaml [more.yaml ...] [--no-check]

The model is instantiated on the meta device (shapes only, no memory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rex.config import load_config  # noqa: E402
from rex.model import RexModel  # noqa: E402
from rex.model.params import active_counts, analytic_counts, body_total, embedding_total, model_counts  # noqa: E402


def report(path: str, check: bool) -> bool:
    cfg = load_config(path)
    a = analytic_counts(cfg)
    ok = True
    m = None
    if check:
        with torch.device("meta"):
            model = RexModel(cfg)
        m = model_counts(model)
    print(f"== {cfg.name} ({path})")
    print(f"{'component':<22}{'formula':>16}" + (f"{'model':>16}" if m else ""))
    for k, v in a.items():
        if v == 0 and (m is None or m[k] == 0):
            continue
        line = f"{k:<22}{v:>16,}"
        if m is not None:
            line += f"{m[k]:>16,}" + ("" if m[k] == v else "   MISMATCH")
            ok &= m[k] == v
        print(line)
    print(f"{'body':<22}{body_total(a):>16,}" + (f"{body_total(m):>16,}" if m else ""))
    print(f"{'embedding':<22}{embedding_total(a):>16,}" + (f"{embedding_total(m):>16,}" if m else ""))
    print(f"{'total':<22}{body_total(a) + embedding_total(a):>16,}")
    for k, v in active_counts(cfg).items():
        print(f"{k:<22}{v:>16,}")
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--no-check", action="store_true", help="skip instantiating the model")
    args = ap.parse_args()
    ok = all([report(p, not args.no_check) for p in args.configs])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
