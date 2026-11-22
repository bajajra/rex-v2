"""T24: curriculum loader. Every sequence starts with bos_id, never exceeds the stage max_len,
eot_id appears only at true document ends, and the token budget per step is within one sequence.

Uses SYNTHETIC corpora (token ids are document-tagged counters, not text), written in the
prepared-shard format of rex/data/prep.py.
"""

import numpy as np
import pytest

from rex.config import CurriculumStage
from rex.data.loader import Corpus, CurriculumLoader, crop_spec, pack_micro
from rex.data.prep import ShardWriter

BOS, EOT = 498, 499


def _write(path, lengths, seed, shard_tokens=5000):
    """Synthetic documents: token j of a document is (j % 400) + 1, so content never equals
    BOS/EOT and the position inside the document is recoverable."""
    w = ShardWriter(path, shard_tokens=shard_tokens)
    for L in lengths:
        w.add([(j % 400) + 1 for j in range(L)], n_bytes=2 * L)
    w.close({"tokenizer_path": "SYNTHETIC", "tokenizer_sha256": "SYNTHETIC", "sources": ["SYNTHETIC"]})
    return Corpus(path)


@pytest.fixture
def corpora(tmp_path):
    rng = np.random.default_rng(0)
    main = _write(tmp_path / "main", list(rng.integers(0, 300, 400)), 0)
    long = _write(tmp_path / "long", list(rng.integers(500, 6000, 40)), 1)
    return main, long


CURR = [CurriculumStage(0.5, 256, 0.15), CurriculumStage(0.8, 1024, 0.5), CurriculumStage(1.0, 4096, 0.7)]


def _loader(main, long, budget=8192, mb=2048, seed=3, total=8192 * 40):
    return CurriculumLoader(main, long, curriculum=CURR, total_tokens=total, tokens_per_step=budget,
                            mb_tokens=mb, bos_id=BOS, eot_id=EOT, seed=seed)


def _sequences(mb):
    cu = mb.cu.tolist()
    return [mb.ids[a:b].numpy() for a, b in zip(cu[:-1], cu[1:])]


def test_t24_curriculum_loader(corpora):
    main, long = corpora
    budget = 8192
    ld = _loader(main, long, budget)
    tokens = 0
    stages_seen = set()
    src_tokens = {}
    while tokens < ld.total_tokens:
        sb = ld.next_step(tokens)
        stage = CURR[sb.stage]
        stages_seen.add(sb.stage)
        assert sb.stage == ld.stage_index(tokens)
        assert sb.max_len == stage.max_len
        # token budget per step within one sequence
        assert abs(sb.n_tokens - budget) <= stage.max_len
        if ld.carry is not None:
            assert budget - sb.n_tokens < ld.carry.length
        assert sb.n_tokens == sum(m.n_tokens for m in sb.micro)
        for mb in sb.micro:
            seqs = _sequences(mb)
            assert mb.n_tokens <= ld.mb_tokens or len(seqs) == 1
            assert mb.n_targets == sum(len(s) - 1 for s in seqs)
            assert mb.pos.tolist() == [i for s in seqs for i in range(len(s))]
            for s, src in zip(seqs, mb.sources):
                assert s[0] == BOS                                     # starts with bos
                assert len(s) <= stage.max_len                         # never exceeds max_len
                assert (s[1:] == BOS).sum() == 0
                body_has_eot = (s == EOT).sum()
                assert body_has_eot <= 1 and (body_has_eot == 0 or s[-1] == EOT)
                src_tokens.setdefault(sb.stage, [0, 0])[src] += len(s)
        tokens += sb.n_tokens
    assert stages_seen == {0, 1, 2}
    # long-document share of tokens per stage (deficit rule: within one long sequence of target)
    for si, (m, l) in src_tokens.items():
        assert abs(l / (m + l) - CURR[si].long_doc_frac) < 0.12, (si, m, l)


def test_t24_eot_only_at_true_ends(corpora):
    """Reconstruct each sequence's document and check eot placement against the true length."""
    main, long = corpora
    rng = np.random.default_rng(0)
    for c, corpus in ((0, main), (1, long)):
        for doc in range(corpus.n_docs):
            L = corpus.doc_len(doc)
            for max_len in (2, 3, 64, 256, 1024):
                s = crop_spec(c, doc, L, max_len, rng)
                assert s.length <= max_len
                assert 0 <= s.start and s.start + s.n_content <= L
                assert s.eot == (s.start + s.n_content == L)           # eot iff the crop reaches the end
                if L + 2 <= max_len:
                    assert s.start == 0 and s.n_content == L and s.eot  # whole document


def test_t24_crop_covers_whole_range():
    rng = np.random.default_rng(1)
    starts = {crop_spec(0, 0, 50, 10, rng).start for _ in range(3000)}
    assert min(starts) == 0 and max(starts) == 50 - 8   # last start keeps eot: [bos] + 8 tokens + eot
    assert 50 - 9 not in starts                          # would end at the true end without eot


def test_t24_loader_resume_is_exact(corpora):
    main, long = corpora
    a = _loader(main, long)
    b = _loader(main, long)
    t = 0
    for _ in range(7):
        t += a.next_step(t).n_tokens
    state = a.state_dict()
    b.load_state_dict(state)
    t2 = t
    for _ in range(10):
        sa, sb = a.next_step(t), b.next_step(t2)
        t, t2 = t + sa.n_tokens, t2 + sb.n_tokens
        for x, y in zip(sa.micro, sb.micro, strict=True):
            assert x.ids.equal(y.ids) and x.cu.equal(y.cu)


def test_pack_micro():
    lens = [100, 900, 300, 2048, 5000, 50]
    bins = pack_micro(lens, 2048)
    assert sorted(i for b in bins for i in b) == list(range(len(lens)))
    for b in bins:
        assert sum(lens[i] for i in b) <= 2048 or len(b) == 1


def test_missing_long_corpus_is_an_error(corpora):
    main, _ = corpora
    with pytest.raises(ValueError):
        _loader(main, None)
    with pytest.warns(UserWarning):
        CurriculumLoader(main, None, curriculum=CURR, total_tokens=1e5, tokens_per_step=4096, mb_tokens=4096,
                         bos_id=BOS, eot_id=EOT, allow_missing_long=True)


def test_t24_base_config_budget(tmp_path):
    """The base config's step budget (262,144 tokens) and curriculum, a few steps per stage."""
    from rex.config import RexConfig

    cfg = RexConfig()
    t, m = cfg.train, cfg.model
    rng = np.random.default_rng(5)
    main = _write(tmp_path / "m", list(rng.integers(0, 4000, 300)), 0, shard_tokens=1 << 20)
    long = _write(tmp_path / "l", list(rng.integers(50_000, 300_000, 12)), 1, shard_tokens=1 << 20)
    ld = CurriculumLoader(main, long, curriculum=t.curriculum, total_tokens=t.total_tokens,
                          tokens_per_step=t.tokens_per_step, mb_tokens=t.mb_tokens, bos_id=m.bos_id,
                          eot_id=m.eot_id, seed=0)
    for frac in (0.0, 0.5, 0.85, 0.95, 0.99):
        tokens = frac * t.total_tokens
        for _ in range(2):
            sb = ld.next_step(tokens)
            st = t.curriculum[sb.stage]
            assert abs(sb.n_tokens - t.tokens_per_step) <= st.max_len
            for mb in sb.micro:
                for s in _sequences(mb):
                    assert s[0] == m.bos_id and len(s) <= st.max_len
                    assert (s == m.eot_id).sum() <= 1 and ((s == m.eot_id).sum() == 0 or s[-1] == m.eot_id)
