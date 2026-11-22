"""Token-budget batches of whole documents and crops, with the length curriculum (§6.5).

* A step draws whole sequences until its token count reaches or exceeds ``tokens_per_step``;
  the last one may overshoot and nothing is carried forward, so a step holds at least
  ``tokens_per_step`` and fewer than ``tokens_per_step + max_len`` tokens (§6.5, T24).
* Every sequence is ``[bos] + document tokens (+ [eot])``; a document longer than the stage
  ``max_len`` contributes one random contiguous crop, and ``eot`` is present only when the
  sequence contains the document's true end.
* Sequences of a step are packed into micro-batches of at most ``mb_tokens`` tokens (first-fit
  decreasing). The packed layout is the model's (§5.1): ``ids[N]``, ``cu[n_seq+1]``, ``pos[N]``.
* The long-document share of tokens per stage (``long_doc_frac``) is met by a deterministic
  deficit rule: the long corpus is drawn whenever its share of the stage's tokens so far is
  below the target.

The loader is fully deterministic given its seed and ``state_dict()``, so resume is exact.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

MAIN, LONG = 0, 1


class Corpus:
    """A prepared split (see prep.py), memory-mapped."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        mf = self.path / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(
                f"{self.path} is not a prepared corpus split (no manifest.json); run scripts/prep_data.py")
        self.manifest = json.loads(mf.read_text())
        self.index = np.load(self.path / "index.npy")
        self.shards = [np.memmap(self.path / f"shard_{i:05d}.bin", dtype=np.uint32, mode="r")
                       if (self.path / f"shard_{i:05d}.bin").stat().st_size > 0 else np.zeros(0, np.uint32)
                       for i in range(self.manifest["n_shards"])]

    @property
    def n_docs(self) -> int:
        return int(self.index.shape[0])

    def doc_len(self, i: int) -> int:
        return int(self.index[i, 2])

    def doc_bytes(self, i: int) -> int:
        return int(self.index[i, 3])

    def tokens(self, i: int, a: int, b: int) -> np.ndarray:
        sh, st = int(self.index[i, 0]), int(self.index[i, 1])
        return np.asarray(self.shards[sh][st + a: st + b], dtype=np.int64)


@dataclass
class SeqSpec:
    corpus: int
    doc: int
    start: int          # first content token of the crop
    n_content: int      # number of document tokens
    eot: bool           # sequence contains the document's end

    @property
    def length(self) -> int:
        return 1 + self.n_content + int(self.eot)

    def to_list(self) -> list:
        return [self.corpus, self.doc, self.start, self.n_content, int(self.eot)]

    @staticmethod
    def from_list(v) -> "SeqSpec":
        return SeqSpec(int(v[0]), int(v[1]), int(v[2]), int(v[3]), bool(v[4]))


@dataclass
class MicroBatch:
    ids: torch.Tensor        # [N] int64
    cu: torch.Tensor         # [n_seq+1] int32
    pos: torch.Tensor        # [N] int64
    n_tokens: int
    n_targets: int           # positions with a next token in the same sequence
    bytes_est: float         # utf-8 bytes covered by the targets (estimated for crops)
    seq_lens: List[int] = field(default_factory=list)
    sources: List[int] = field(default_factory=list)

    def to(self, device) -> "MicroBatch":
        return MicroBatch(self.ids.to(device), self.cu.to(device), self.pos.to(device), self.n_tokens,
                          self.n_targets, self.bytes_est, self.seq_lens, self.sources)


@dataclass
class StepBatch:
    micro: List[MicroBatch]
    stage: int
    max_len: int
    n_tokens: int
    n_targets: int


def crop_spec(corpus_id: int, doc: int, doc_len: int, max_len: int, rng: np.random.Generator) -> SeqSpec:
    """Whole document if ``1 + doc_len + 1 <= max_len``, else one random contiguous crop.

    The crop is a window of ``W = max_len - 1`` positions over ``doc + [eot]``; the start that
    would end exactly at the last document token (true end but no room for ``eot``) is moved one
    position later so that ``eot`` is present iff the crop reaches the document end.
    """
    if max_len < 2:
        raise ValueError("max_len must be >= 2")
    if doc_len + 2 <= max_len:
        return SeqSpec(corpus_id, doc, 0, doc_len, True)
    W = max_len - 1
    s = int(rng.integers(0, doc_len + 1 - W + 1))      # windows over doc_len + 1 positions
    if s + W == doc_len:
        s += 1
    end = s + W
    if end > doc_len:                                  # window includes the eot position
        return SeqSpec(corpus_id, doc, s, doc_len - s, True)
    return SeqSpec(corpus_id, doc, s, W, False)


def pack_micro(lengths: Sequence[int], mb_tokens: int) -> List[List[int]]:
    """First-fit decreasing; returns lists of indices. A sequence longer than ``mb_tokens`` gets
    a micro-batch of its own (sequences are never split, §5.1)."""
    # §6.5: mb_tokens is a packing budget; a longer sequence forms a micro-batch of its own.
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    bins: List[List[int]] = []
    room: List[int] = []
    for i in order:
        L = lengths[i]
        for b in range(len(bins)):
            if room[b] >= L:
                bins[b].append(i)
                room[b] -= L
                break
        else:
            bins.append([i])
            room.append(mb_tokens - L)
    return [sorted(b) for b in bins]


class CurriculumLoader:
    def __init__(self, main: Corpus, long: Optional[Corpus], *, curriculum, total_tokens: float,
                 tokens_per_step: int, mb_tokens: int, bos_id: int, eot_id: int, seed: int = 0,
                 allow_missing_long: bool = False):
        self.corpora = [main, long]
        self.curriculum = list(curriculum)
        self.total_tokens = float(total_tokens)
        self.tokens_per_step = int(tokens_per_step)
        self.mb_tokens = int(mb_tokens)
        self.bos_id, self.eot_id = bos_id, eot_id
        self.seed = int(seed)
        if long is None and any(s.long_doc_frac > 0 for s in self.curriculum):
            if not allow_missing_long:
                raise ValueError("curriculum asks for long documents but no long-document corpus was given")
            warnings.warn("no long-document corpus: every sequence is drawn from the main corpus")
        self.allow_missing_long = allow_missing_long
        if main.n_docs == 0 or (long is not None and long.n_docs == 0):
            raise ValueError("empty corpus")
        # state
        self.epoch = [0, 0]
        self.cursor = [0, 0]
        self.stage_tokens = [0, 0]       # tokens drawn per source within the current stage
        self.stage = -1
        self._perm_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        return {"epoch": list(self.epoch), "cursor": list(self.cursor), "stage_tokens": list(self.stage_tokens),
                "stage": self.stage,
                "seed": self.seed}

    def load_state_dict(self, s: dict) -> None:
        self.epoch, self.cursor = list(s["epoch"]), list(s["cursor"])
        self.stage_tokens, self.stage = list(s["stage_tokens"]), int(s["stage"])
        if int(s["seed"]) != self.seed:
            warnings.warn(f"loader seed changed on resume ({s['seed']} -> {self.seed}); keeping the saved seed")
            self.seed = int(s["seed"])

    # ------------------------------------------------------------------ curriculum
    def stage_index(self, tokens_consumed: float) -> int:
        frac = tokens_consumed / self.total_tokens
        for i, s in enumerate(self.curriculum):
            if frac < s.until:
                return i
        return len(self.curriculum) - 1

    # ------------------------------------------------------------------ drawing
    def _perm(self, c: int, epoch: int) -> np.ndarray:
        key = (c, epoch)
        p = self._perm_cache.get(key)
        if p is None:
            p = np.random.default_rng([self.seed, c, epoch]).permutation(self.corpora[c].n_docs)
            self._perm_cache = {key: p} | {k: v for k, v in self._perm_cache.items() if k[0] != c}
        return p

    def _draw(self, c: int, max_len: int) -> SeqSpec:
        corpus = self.corpora[c]
        doc = int(self._perm(c, self.epoch[c])[self.cursor[c]])
        rng = np.random.default_rng([self.seed, c, self.epoch[c], self.cursor[c], 7])
        self.cursor[c] += 1
        if self.cursor[c] >= corpus.n_docs:
            self.cursor[c] = 0
            self.epoch[c] += 1
        return crop_spec(c, doc, corpus.doc_len(doc), max_len, rng)

    def _choose_source(self, frac: float) -> int:
        if self.corpora[LONG] is None or frac <= 0:
            return MAIN
        tot = self.stage_tokens[MAIN] + self.stage_tokens[LONG]
        return LONG if self.stage_tokens[LONG] < frac * max(tot, 1) else MAIN

    def next_step(self, tokens_consumed: float) -> StepBatch:
        si = self.stage_index(tokens_consumed)
        st = self.curriculum[si]
        if si != self.stage:
            self.stage = si
            self.stage_tokens = [0, 0]
        # §6.5: draw whole sequences until the step reaches or exceeds tokens_per_step; the last
        # one may overshoot and nothing is carried forward.
        specs: List[SeqSpec] = []
        total = 0
        while total < self.tokens_per_step:
            s = self._draw(self._choose_source(st.long_doc_frac), st.max_len)
            self.stage_tokens[s.corpus] += s.length
            specs.append(s)
            total += s.length
        micro = [self.collate([specs[i] for i in idx]) for idx in pack_micro([s.length for s in specs], self.mb_tokens)]
        return StepBatch(micro, si, st.max_len, total, sum(m.n_targets for m in micro))

    def materialize(self, s: SeqSpec) -> np.ndarray:
        body = self.corpora[s.corpus].tokens(s.doc, s.start, s.start + s.n_content)
        parts = [np.array([self.bos_id], dtype=np.int64), body]
        if s.eot:
            parts.append(np.array([self.eot_id], dtype=np.int64))
        return np.concatenate(parts)

    def collate(self, specs: Sequence[SeqSpec]) -> MicroBatch:
        return collate_sequences([self.materialize(s) for s in specs],
                                 [self._bytes_est(s) for s in specs], [s.corpus for s in specs])

    def _bytes_est(self, s: SeqSpec) -> float:
        c = self.corpora[s.corpus]
        L = c.doc_len(s.doc)
        if L == 0:
            return 0.0
        # SPEC: §8 asks for bits per byte; bytes of a crop are estimated pro rata by tokens.
        return c.doc_bytes(s.doc) * min(1.0, s.n_content / L)


def collate_sequences(seqs: Sequence[np.ndarray], bytes_est: Sequence[float] = (), sources: Sequence[int] = ()) -> MicroBatch:
    lens = [int(len(s)) for s in seqs]
    ids = torch.from_numpy(np.concatenate(seqs).astype(np.int64))
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32)
    starts = torch.repeat_interleave(cu[:-1].to(torch.int64), torch.tensor(lens))
    pos = torch.arange(ids.numel(), dtype=torch.int64) - starts
    return MicroBatch(ids, cu, pos, int(ids.numel()), int(sum(max(L - 1, 0) for L in lens)),
                      float(sum(bytes_est)), lens, list(sources))


def fixed_batches(main: Corpus, long: Optional[Corpus], *, max_len: int, n_batches: int, mb_tokens: int,
                  long_doc_frac: float, bos_id: int, eot_id: int, seed: int = 1234) -> List[MicroBatch]:
    """Fixed held-out micro-batches for evaluation (§10: "fixed held-out batches")."""
    ld = CurriculumLoader(main, long, curriculum=[_Stage(1.0, max_len, long_doc_frac if long is not None else 0.0)],
                          total_tokens=1.0, tokens_per_step=mb_tokens, mb_tokens=mb_tokens,
                          bos_id=bos_id, eot_id=eot_id, seed=seed)
    out: List[MicroBatch] = []
    while len(out) < n_batches:
        out.extend(ld.next_step(0).micro)
    return out[:n_batches]


@dataclass
class _Stage:
    until: float
    max_len: int
    long_doc_frac: float
