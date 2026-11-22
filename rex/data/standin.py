"""SYNTHETIC STAND-IN tokenizer and corpora for CPU smoke runs and tests.

NOT the Inkling tokenizer and NOT the spec's datasets. The real tokenizer (§8) could not be
downloaded in the development container, so the pipeline is exercised end to end with:

* a tiny byte-level BPE tokenizer trained here, with exactly ``vocab_size`` ids and the two
  special tokens at the tiny config's ``bos_id`` / ``eot_id`` (498 / 499 of 500);
* text documents taken from local files (by default the Python standard library sources that
  ship with the interpreter), split into a short-document "main" corpus and a long-document
  corpus, each with train/val splits.
"""

from __future__ import annotations

import hashlib
import sysconfig
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .tokenizer import BOS_TOKEN, EOT_TOKEN


def local_documents(root: Optional[str] = None, max_files: int = 4000) -> List[Tuple[str, str]]:
    root = Path(root or sysconfig.get_paths()["stdlib"])
    out = []
    for p in sorted(root.rglob("*.py")):
        if "site-packages" in p.parts:
            continue
        try:
            out.append((str(p.relative_to(root)), p.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
        if len(out) >= max_files:
            break
    return out


def train_standin_tokenizer(texts: Sequence[str], out_file: str | Path, vocab_size: int = 500,
                            bos_id: int = 498, eot_id: int = 499) -> Path:
    """Byte-level BPE with ``vocab_size - 2`` regular ids, then the specials at the end."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if sorted((bos_id, eot_id)) != [vocab_size - 2, vocab_size - 1]:
        raise ValueError("stand-in tokenizer places the specials at the last two ids")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size - 2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != vocab_size - 2:
        raise RuntimeError(f"stand-in BPE produced {tok.get_vocab_size()} ids, wanted {vocab_size - 2}")
    first, second = (BOS_TOKEN, EOT_TOKEN) if bos_id < eot_id else (EOT_TOKEN, BOS_TOKEN)
    tok.add_special_tokens([first, second])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_file))
    return out_file


def split_documents(docs: Sequence[Tuple[str, str]], long_chars: int = 30000, val_frac: float = 0.05):
    """-> {"main": {"train": [...], "val": [...]}, "long": {...}} by length and name hash."""
    out = {"main": {"train": [], "val": []}, "long": {"train": [], "val": []}}
    for name, text in docs:
        kind = "long" if len(text) >= long_chars else "main"
        h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 1000
        out[kind]["val" if h < val_frac * 1000 else "train"].append(text)
    return out


def build_standin(out_dir: str | Path, vocab_size: int = 500, bos_id: int = 498, eot_id: int = 499,
                  root: Optional[str] = None, max_files: int = 4000) -> dict:
    """Write ``tokenizer.json`` and prepared ``main/`` and ``long/`` corpora under ``out_dir``."""
    from .prep import prep_corpus
    from .tokenizer import load_tokenizer

    out_dir = Path(out_dir)
    docs = local_documents(root, max_files)
    tf = train_standin_tokenizer([t for _, t in docs[:800]], out_dir / "tokenizer.json", vocab_size, bos_id, eot_id)
    tok = load_tokenizer(tf, unpadded_vocab_size=vocab_size, bos_id=bos_id, eot_id=eot_id)
    parts = split_documents(docs)
    manifests = {}
    for kind in ("main", "long"):
        for split in ("train", "val"):
            manifests[f"{kind}/{split}"] = prep_corpus(parts[kind][split], tok, out_dir / kind / split,
                                                       sources=["STAND-IN: local Python stdlib sources"],
                                                       vocab_limit=vocab_size, log_every=0)
    return {"tokenizer": str(tf), "manifests": manifests}
