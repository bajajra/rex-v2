#!/usr/bin/env python
"""Build the SYNTHETIC STAND-IN tokenizer and corpora used by the CPU smoke run.

    python scripts/make_standin.py --out runs/standin

Not the Inkling tokenizer and not the spec's datasets: a 500-id byte-level BPE trained on the
local Python standard-library sources, and those sources split into short ("main") and long
documents. See rex/data/standin.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rex.data.standin import build_standin  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/standin")
    ap.add_argument("--root", default=None, help="directory of .py files (default: Python stdlib)")
    args = ap.parse_args(argv)
    r = build_standin(args.out, root=args.root)
    for k, m in r["manifests"].items():
        print(f"{k}: {m['n_docs']} docs, {m['n_tokens']} tokens")
    print(f"tokenizer: {r['tokenizer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
