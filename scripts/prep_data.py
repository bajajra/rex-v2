#!/usr/bin/env python
"""Tokenise a corpus into uint32 shards + document index (§8).

Hub presets (needs access to huggingface.co):
    python scripts/prep_data.py --preset fineweb-edu-10BT --out /data/fineweb
    python scripts/prep_data.py --preset pg19 --out /data/pg19

Local files (.parquet with a text column, .jsonl, or plain text files = one document each):
    python scripts/prep_data.py --files '/data/raw/*.parquet' --out /data/mycorpus --holdout-last-file

Output: <out>/train/ and <out>/val/ (val = PG-19 validation split, or the last input file with
--holdout-last-file, per §8: "Held-out: PG-19 validation and test books, plus the last
FineWeb-Edu shard"). The tokenizer's SHA-256 is written to each manifest.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rex.config import load_config  # noqa: E402
from rex.data.prep import HUB_PRESETS, hub_files, iter_texts, prep_corpus  # noqa: E402
from rex.data.tokenizer import INKLING_HUB_ID, load_tokenizer_for  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=sorted(HUB_PRESETS))
    src.add_argument("--files", nargs="+", help="glob(s) of local input files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/rex-v2-base.yaml", help="for vocab size and special ids")
    ap.add_argument("--tokenizer", default=INKLING_HUB_ID, help="tokenizer.json, its directory, or hub id")
    ap.add_argument("--tokenizer-sha256", default=None, help="pin: refuse a tokenizer with another hash")
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--holdout-last-file", action="store_true", help="write the last input file to val/")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--shard-tokens", type=int, default=1 << 28)
    ap.add_argument("--cache-dir", default=None, help="HF download cache (keep it outside the repo)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tok = load_tokenizer_for(cfg, args.tokenizer, expected_sha256=args.tokenizer_sha256)
    out = Path(args.out)
    splits = {}
    if args.preset:
        p = HUB_PRESETS[args.preset]
        field = args.text_field or p["text_field"]
        for split in p["patterns"]:
            splits[split] = hub_files(args.preset, split, args.cache_dir)
        if "validation" in splits:
            # PG-19: validation and test books are both held out (§8); val/ gets validation, test/ test
            splits["val"] = splits.pop("validation")
        elif args.holdout_last_file or args.preset == "fineweb-edu-10BT":
            splits["val"] = splits["train"][-1:]
            splits["train"] = splits["train"][:-1]
    else:
        field = args.text_field or "text"
        files = sorted(f for g in args.files for f in glob.glob(g))
        if not files:
            raise SystemExit("no input files matched")
        splits["train"] = files[:-1] if args.holdout_last_file and len(files) > 1 else files
        if args.holdout_last_file and len(files) > 1:
            splits["val"] = files[-1:]
    for split, files in splits.items():
        print(f"[prep] {split}: {len(files)} files -> {out / split}", flush=True)
        m = prep_corpus(iter_texts(files, field), tok, out / split, sources=files, shard_tokens=args.shard_tokens,
                        max_docs=args.max_docs, vocab_limit=cfg.model.unpadded_vocab_size)
        print(f"[prep] {split}: {m['n_docs']} docs, {m['n_tokens']} tokens, {m['n_bytes']} bytes, "
              f"{m['n_bytes'] / max(m['n_tokens'], 1):.3f} bytes/token; tokenizer sha256 {m['tokenizer_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
