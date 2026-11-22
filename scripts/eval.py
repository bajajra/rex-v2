#!/usr/bin/env python
"""Evaluate a checkpoint with the §10 metrics and probes.

    python scripts/eval.py --ckpt runs/base/checkpoints/step_0022888.pt [--set data.long_docs=/data/pg19] \
        [--eval-max-len 8192] [--long-max-len 262144] [--contexts 1024 8192 65536 all] \
        [--gaps 2048 16384 131072] [--out eval.json]

Short-context items (held-out loss in nats and bits per byte, forced T = 1..n_passes, message
ablation, gate / β / pass-change / token-correlation statistics, α maps, block and running-sum
RMS, output-gate means, pass overlap, attention distance) run on fixed batches of the main
corpus's val split. Long-context items (loss by position bucket, context truncation, repeat
probe) use the long-document val split (PG-19 validation by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rex.config import RexConfig  # noqa: E402
from rex.data.loader import fixed_batches  # noqa: E402
from rex.eval.probes import (context_truncation, eval_loss, evaluate, position_buckets,  # noqa: E402
                             repeat_probe)
from rex.model import RexModel  # noqa: E402
from rex.train.run import apply_overrides, build_eval_batches, val_corpora  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-max-len", type=int, default=None)
    ap.add_argument("--long-docs", type=int, default=8, help="number of long val documents for long probes")
    ap.add_argument("--long-max-len", type=int, default=None, help="default: last curriculum stage max_len")
    ap.add_argument("--contexts", nargs="+", default=["1024", "8192", "65536", "all"])
    ap.add_argument("--target-len", type=int, default=512)
    ap.add_argument("--gaps", nargs="+", type=int, default=[2048, 16384, 131072])
    ap.add_argument("--passage-len", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = apply_overrides(RexConfig.from_dict(ck["config"]), args.set)
    model = RexModel(cfg)
    model.load_state_dict(ck["model"])
    model.to(args.device)
    prec = cfg.train.precision
    chunk = cfg.loss.loss_chunk_tokens
    res = {"ckpt": args.ckpt, "step": ck["step"], "tokens": ck["tokens"]}

    batches = build_eval_batches(cfg, args.eval_batches, args.eval_max_len)
    if batches:
        res.update(evaluate(model, batches, chunk_tokens=chunk, device=args.device, precision=prec))
    else:
        print("[eval] no main val split: short-context items skipped")

    main_val, long_val = val_corpora(cfg)
    if long_val is not None:
        Lmax = args.long_max_len or cfg.train.curriculum[-1].max_len
        lb = fixed_batches(long_val, None, max_len=Lmax, n_batches=args.long_docs, mb_tokens=Lmax,
                           long_doc_frac=0.0, bos_id=cfg.model.bos_id, eot_id=cfg.model.eot_id)
        r = eval_loss(model, lb, chunk_tokens=chunk, per_token=True, device=args.device, precision=prec)
        res["long/nats"], res["long/bpb"] = r["nats"], r["bpb"]
        res.update({f"long/pos_bucket/{k}": v for k, v in position_buckets(r["nll"], r["pos"]).items()})
        docs = [long_val.tokens(i, 0, long_val.doc_len(i)) for i in range(long_val.n_docs)]
        docs = sorted(docs, key=len, reverse=True)[: args.long_docs]
        ctx = [None if c == "all" else int(c) for c in args.contexts]
        res.update(context_truncation(model, docs, bos_id=cfg.model.bos_id, contexts=ctx, target_len=args.target_len,
                                      max_total=Lmax, chunk_tokens=chunk, device=args.device, precision=prec))
        filler_src = main_val if main_val is not None else long_val
        filler = np.concatenate([filler_src.tokens(i, 0, filler_src.doc_len(i))
                                 for i in range(filler_src.n_docs)])
        rng = np.random.default_rng(0)
        passages = []
        for d in docs:
            if len(d) > args.passage_len:
                s = int(rng.integers(0, len(d) - args.passage_len))
                passages.append(d[s:s + args.passage_len])
        gaps = [g for g in args.gaps if g + 2 * args.passage_len + 1 <= Lmax]
        res.update(repeat_probe(model, passages[:4], filler, bos_id=cfg.model.bos_id, gaps=gaps,
                                chunk_tokens=chunk, device=args.device, precision=prec))
    else:
        print("[eval] no long-document val split: long-context items skipped")

    text = json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if args.out:
        Path(args.out).write_text(text)
    for k, v in res.items():
        if k != "alpha" and not isinstance(v, dict):
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
