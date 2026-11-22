#!/usr/bin/env python
"""Train rex-v2 (or a baseline) from a config.

    python scripts/train.py --config configs/rex-v2-base.yaml --out runs/base \
        --set data.main=/data/fineweb --set data.long_docs=/data/pg19 [--resume]

``data.main`` / ``data.long_docs`` must point at directories prepared by scripts/prep_data.py
(each with ``train/`` and optionally ``val/``). Any config key can be overridden with
``--set section.key=value`` (e.g. ``train.mb_tokens=32768``, ``train.tok_chunk=32768``,
``model.attn.backend=reference``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rex.config import load_config  # noqa: E402
from rex.model import RexModel  # noqa: E402
from rex.train.run import apply_overrides, build_eval_batches, build_loader  # noqa: E402
from rex.train.trainer import Trainer, latest_checkpoint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="run directory (default runs/<config name>)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (schedule still uses the full length)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=8, help="fixed held-out micro-batches (0: no eval)")
    ap.add_argument("--eval-max-len", type=int, default=None, help="default: first curriculum stage max_len")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--keep-ckpts", type=int, default=2)
    ap.add_argument("--allow-missing-long", action="store_true", help="train without a long-document corpus")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = apply_overrides(load_config(args.config), args.set)
    out = Path(args.out or f"runs/{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    torch.manual_seed(args.seed)
    model = RexModel(cfg)
    loader = build_loader(cfg, seed=args.seed, allow_missing_long=args.allow_missing_long)
    eval_batches = build_eval_batches(cfg, args.eval_batches, args.eval_max_len)
    if eval_batches is None and args.eval_batches:
        print("[train] no prepared 'val' split found: periodic evaluation disabled", flush=True)
    tr = Trainer(cfg, model, loader, out_dir=out, device=args.device, eval_batches=eval_batches,
                 log_every=args.log_every, eval_every=args.eval_every, ckpt_every=args.ckpt_every,
                 keep_ckpts=args.keep_ckpts, seed=args.seed, max_steps=args.max_steps)
    if args.resume:
        ck = latest_checkpoint(out)
        if ck is None:
            print(f"[train] --resume: no checkpoint in {out}, starting from scratch", flush=True)
        else:
            tr.load(ck)
            print(f"[train] resumed from {ck} at step {tr.step}", flush=True)
    n_body = sum(p.numel() for n, p in model.named_parameters() if n not in ("emb.weight", "lm_head.weight"))
    print(f"[train] {cfg.name}: body params {n_body:,}; {cfg.train.total_tokens:,.0f} tokens (nominally {tr.n_steps} steps "
          f"of >= {cfg.train.tokens_per_step:,}); device {args.device}; precision {cfg.train.precision}", flush=True)
    tr.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
