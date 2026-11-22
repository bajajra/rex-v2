"""Pinned Inkling tokenizer (§8).

``tokenizer.json`` from https://huggingface.co/thinkingmachines/Inkling-2-Mini, loaded with the
``tokenizers`` library. Ids 0..200,057 are valid; only ``<|begin_of_text|>`` (200,028) and
``<|endoftext|>`` (199,999) are used in pre-training. The file's SHA-256 is recorded in every
data-prep manifest and can be pinned with ``expected_sha256``.

``spec`` may be a local ``tokenizer.json``, a local directory that contains one, or a hub repo id
(downloaded with ``huggingface_hub`` into the user's HF cache, outside the repository).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence

INKLING_HUB_ID = "thinkingmachines/Inkling-2-Mini"
BOS_TOKEN = "<|begin_of_text|>"
EOT_TOKEN = "<|endoftext|>"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_tokenizer_file(spec: str | Path, revision: Optional[str] = None) -> Path:
    p = Path(os.path.expanduser(str(spec)))
    if p.is_file():
        return p
    if p.is_dir():
        f = p / "tokenizer.json"
        if not f.is_file():
            raise FileNotFoundError(f"{p} has no tokenizer.json")
        return f
    if str(spec).count("/") != 1 or str(spec).startswith((".", "/")):
        raise FileNotFoundError(f"tokenizer {spec!r} is neither a local path nor a hub repo id")
    from huggingface_hub import hf_hub_download  # imported lazily: optional for local use

    return Path(hf_hub_download(repo_id=str(spec), filename="tokenizer.json", revision=revision))


class RexTokenizer:
    """Thin wrapper over ``tokenizers.Tokenizer``; ``encode`` never adds special tokens
    (the loader adds ``bos_id``/``eot_id`` itself, §6.5)."""

    def __init__(self, path: Path, bos_id: int, eot_id: int):
        from tokenizers import Tokenizer

        self.path = Path(path)
        self.sha256 = file_sha256(self.path)
        self.tok = Tokenizer.from_file(str(self.path))
        self.bos_id, self.eot_id = bos_id, eot_id

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)


def load_tokenizer(spec: str | Path = INKLING_HUB_ID, *, unpadded_vocab_size: int = 200058,
                   bos_id: int = 200028, eot_id: int = 199999, bos_token: str = BOS_TOKEN,
                   eot_token: str = EOT_TOKEN, expected_sha256: Optional[str] = None,
                   revision: Optional[str] = None) -> RexTokenizer:
    """Load and check the tokenizer against the model config: vocabulary size equals
    ``unpadded_vocab_size`` and the two special tokens map to ``bos_id``/``eot_id``."""
    path = resolve_tokenizer_file(spec, revision)
    t = RexTokenizer(path, bos_id, eot_id)
    if expected_sha256 is not None and t.sha256 != expected_sha256:
        raise ValueError(f"tokenizer hash mismatch: {t.sha256} != pinned {expected_sha256}")
    if t.vocab_size != unpadded_vocab_size:
        raise ValueError(f"tokenizer has {t.vocab_size} ids, config expects {unpadded_vocab_size}")
    for name, tok_str, want in (("bos", bos_token, bos_id), ("eot", eot_token, eot_id)):
        got = t.tok.token_to_id(tok_str)
        if got != want:
            raise ValueError(f"{name} token {tok_str!r} has id {got}, config expects {want}")
    return t


def load_tokenizer_for(cfg, spec: str | Path = INKLING_HUB_ID, **kw) -> RexTokenizer:
    m = cfg.model
    return load_tokenizer(spec, unpadded_vocab_size=m.unpadded_vocab_size, bos_id=m.bos_id,
                          eot_id=m.eot_id, **kw)
