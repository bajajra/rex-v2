"""Glue used by scripts/train.py and scripts/eval.py: config overrides, corpus resolution,
loader and evaluation-batch construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

from ..config import RexConfig
from ..data.loader import Corpus, CurriculumLoader, fixed_batches


def parse_overrides(items: Sequence[str]) -> dict:
    """``["train.lr=2e-4", "model.attn.backend=reference"]`` -> nested dict for RexConfig.replace."""
    out: dict = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"override {it!r} must look like section.key=value")
        key, val = it.split("=", 1)
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def apply_overrides(cfg: RexConfig, items: Sequence[str]) -> RexConfig:
    return cfg.replace(**parse_overrides(items)) if items else cfg


def resolve_split(spec: Optional[str], split: str) -> Optional[Path]:
    """``data.main`` / ``data.long_docs`` name a prepared corpus directory (output of
    scripts/prep_data.py, holding ``train/`` and ``val/`` splits). The spec's defaults are hub
    dataset names, which must be prepared first."""
    if spec is None or str(spec).lower() in ("", "none", "null"):
        return None
    p = Path(os.path.expanduser(str(spec)))
    if (p / split / "manifest.json").is_file():
        return p / split
    if (p / "manifest.json").is_file() and split == "train":
        return p
    raise FileNotFoundError(
        f"data source {spec!r} has no prepared '{split}' split. Run scripts/prep_data.py and set "
        f"data.main / data.long_docs to the output directory (e.g. --set data.main=/path/to/fineweb)")


def build_loader(cfg: RexConfig, seed: int = 0, allow_missing_long: bool = False) -> CurriculumLoader:
    main = Corpus(resolve_split(cfg.data.main, "train"))
    lp = resolve_split(cfg.data.long_docs, "train") if cfg.data.long_docs else None
    long = Corpus(lp) if lp is not None else None
    t, m = cfg.train, cfg.model
    return CurriculumLoader(main, long, curriculum=t.curriculum, total_tokens=t.total_tokens,
                            tokens_per_step=t.tokens_per_step, mb_tokens=t.mb_tokens, bos_id=m.bos_id,
                            eot_id=m.eot_id, seed=seed, allow_missing_long=allow_missing_long)


def val_corpora(cfg: RexConfig) -> Tuple[Optional[Corpus], Optional[Corpus]]:
    def get(spec):
        try:
            p = resolve_split(spec, "val")
        except FileNotFoundError:
            return None
        return Corpus(p) if p is not None else None
    return get(cfg.data.main), get(cfg.data.long_docs)


def build_eval_batches(cfg: RexConfig, n_batches: int, max_len: Optional[int] = None,
                       mb_tokens: Optional[int] = None, seed: int = 1234):
    """Fixed held-out micro-batches at the first stage's max_len (§10); None without a val split."""
    main, long = val_corpora(cfg)
    if main is None or n_batches <= 0:
        return None
    st = cfg.train.curriculum[0]
    L = max_len or st.max_len
    return fixed_batches(main, long, max_len=L, n_batches=n_batches, mb_tokens=max(mb_tokens or cfg.train.mb_tokens, L),
                         long_doc_frac=st.long_doc_frac, bos_id=cfg.model.bos_id, eot_id=cfg.model.eot_id, seed=seed)
