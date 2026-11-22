"""Data prep (§8): uint32 token shards plus a document-offset index.

Layout of a prepared corpus split (``<out>/<split>/``)::

    shard_00000.bin ...   uint32 token ids, documents concatenated, no special tokens
    index.npy             int64 [n_docs, 4]: shard, start (in tokens), length, utf-8 bytes
    manifest.json         tokenizer path + SHA-256, counts, sources

Documents are stored without ``bos``/``eot``; the loader adds them per sequence (§6.5).
Documents never straddle shards. Byte counts are kept for bits-per-byte reporting (§8).
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

# Hub presets for the default corpora of §8. Untested in the development container (hub access
# was rate-limited); file patterns follow the dataset repos' published layouts.
HUB_PRESETS = {
    # FineWeb-Edu sample-10BT parquet files
    "fineweb-edu-10BT": dict(repo_id="HuggingFaceFW/fineweb-edu", revision=None,
                             patterns={"train": ["sample/10BT/*.parquet"]}, text_field="text"),
    # PG-19 via its parquet conversion (§8: "read its parquet conversion instead")
    "pg19": dict(repo_id="deepmind/pg19", revision="refs/convert/parquet",
                 patterns={"train": ["default/train/*.parquet"],
                           "validation": ["default/validation/*.parquet"],
                           "test": ["default/test/*.parquet"]}, text_field="text"),
}


def hub_files(preset: str, split: str, cache_dir: Optional[str] = None) -> List[str]:
    from huggingface_hub import snapshot_download

    p = HUB_PRESETS[preset]
    pats = p["patterns"][split]
    root = snapshot_download(repo_id=p["repo_id"], repo_type="dataset", revision=p["revision"],
                             allow_patterns=pats, cache_dir=cache_dir)
    files: List[str] = []
    for pat in pats:
        files += sorted(glob.glob(os.path.join(root, pat)))
    return files


def iter_texts(files: Sequence[str], text_field: str = "text") -> Iterator[str]:
    """Texts from .parquet (column ``text_field``), .jsonl (key ``text_field``) or plain files
    (one document per file)."""
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(f)
            for rb in pf.iter_batches(columns=[text_field], batch_size=1024):
                for t in rb.column(0).to_pylist():
                    yield t or ""
        elif f.endswith(".jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)[text_field]
        else:
            with open(f, encoding="utf-8", errors="replace") as fh:
                yield fh.read()


class ShardWriter:
    def __init__(self, out_dir: str | Path, shard_tokens: int = 1 << 28):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if any(self.dir.glob("shard_*.bin")) or (self.dir / "index.npy").exists():
            raise FileExistsError(f"{self.dir} already holds a prepared split")
        self.shard_tokens = shard_tokens
        self.shard = -1
        self.fh = None
        self.pos = 0
        self.index: List[tuple] = []
        self.n_tokens = 0
        self.n_bytes = 0
        self._open()

    def _open(self) -> None:
        if self.fh is not None:
            self.fh.close()
        self.shard += 1
        self.fh = open(self.dir / f"shard_{self.shard:05d}.bin", "wb")
        self.pos = 0

    def add(self, ids: Sequence[int], n_bytes: int) -> None:
        a = np.asarray(ids, dtype=np.uint32)
        if self.pos > 0 and self.pos + a.size > self.shard_tokens:
            self._open()
        self.fh.write(a.tobytes())
        self.index.append((self.shard, self.pos, a.size, n_bytes))
        self.pos += a.size
        self.n_tokens += a.size
        self.n_bytes += n_bytes

    def close(self, manifest: dict) -> dict:
        self.fh.close()
        idx = np.asarray(self.index, dtype=np.int64).reshape(-1, 4)
        np.save(self.dir / "index.npy", idx)
        manifest = dict(manifest, n_docs=int(idx.shape[0]), n_tokens=int(self.n_tokens),
                        n_bytes=int(self.n_bytes), n_shards=self.shard + 1, dtype="uint32",
                        created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


def prep_corpus(texts: Iterable[str], tokenizer, out_dir: str | Path, *, sources: Sequence[str] = (),
                shard_tokens: int = 1 << 28, batch_docs: int = 256, max_docs: Optional[int] = None,
                vocab_limit: Optional[int] = None, log_every: int = 100000) -> dict:
    """Tokenise ``texts`` (one document each) into ``out_dir``. ``vocab_limit`` (normally
    ``unpadded_vocab_size``) is checked for every id."""
    w = ShardWriter(out_dir, shard_tokens)
    buf: List[str] = []
    n = 0
    t0 = time.time()

    def flush():
        nonlocal buf
        for t, ids in zip(buf, tokenizer.encode_batch(buf)):
            if vocab_limit is not None and ids and max(ids) >= vocab_limit:
                raise ValueError(f"token id {max(ids)} >= vocab limit {vocab_limit}")
            w.add(ids, len(t.encode("utf-8")))
        buf = []

    for t in texts:
        buf.append(t)
        n += 1
        if len(buf) >= batch_docs:
            flush()
        if log_every and n % log_every == 0:
            print(f"[prep] {n} docs, {w.n_tokens} tokens, {time.time() - t0:.0f}s", flush=True)
        if max_docs is not None and n >= max_docs:
            break
    if buf:
        flush()
    return w.close({"tokenizer_path": str(tokenizer.path), "tokenizer_sha256": tokenizer.sha256,
                    "sources": list(sources)})
