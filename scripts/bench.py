#!/usr/bin/env python
"""Throughput and peak memory of one training micro-batch (forward + chunked head + backward).

    python scripts/bench.py --config configs/rex-v2-base.yaml --seq-len 8192 [--n-seq 1] [--steps 3]

Use it on the target GPU to replace the §6.3 throughput assumptions and the §6.4 memory
estimates (T16 / T23). Random token ids; reports tokens/s and peak memory per token.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rex.config import load_config  # noqa: E402
from rex.model import RexModel, next_token_targets, packed_positions  # noqa: E402
from rex.train.losses import chunked_lm_backward  # noqa: E402
from rex.train.run import apply_overrides  # noqa: E402
from rex.train.trainer import autocast_ctx, peak_memory_bytes  # noqa: E402


def run(cfg, seq_len: int, n_seq: int, steps: int, device: str) -> dict:
    dev = torch.device(device)
    model = RexModel(cfg).to(dev).train()
    cu = torch.arange(n_seq + 1, dtype=torch.int32, device=dev) * seq_len
    pos = packed_positions(cu)
    ids = torch.randint(0, cfg.model.unpadded_vocab_size, (seq_len * n_seq,), device=dev)
    ids[cu[:-1].long()] = cfg.model.bos_id
    tgt = next_token_targets(ids, cu)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    times = []
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.time()
        with autocast_ctx(dev, cfg.train.precision):
            out = model(ids, cu, pos)
            chunked_lm_backward(model, out.hidden, tgt, chunk_tokens=cfg.loss.loss_chunk_tokens,
                                ce_scale=1.0 / max(int((tgt != -100).sum()), 1), aux_loss=out.aux_loss)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times.append(time.time() - t0)
    N = seq_len * n_seq
    best = min(times[1:] if len(times) > 1 else times)
    peak = peak_memory_bytes(dev)
    return {"tokens": N, "sec": best, "tok_s": N / best, "peak_bytes": peak, "peak_bytes_per_token": peak / N}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--n-seq", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    r = run(cfg, args.seq_len, args.n_seq, args.steps, args.device)
    print(f"{cfg.name} seq_len={args.seq_len} n_seq={args.n_seq} device={args.device} "
          f"grad_ckpt={cfg.train.grad_ckpt} tok_chunk={cfg.train.tok_chunk}: {r['tok_s']:.0f} tok/s, "
          f"peak {r['peak_bytes'] / 2**30:.2f} GiB ({r['peak_bytes_per_token'] / 1024:.1f} KiB/token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
