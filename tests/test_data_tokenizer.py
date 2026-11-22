"""Tokenizer loading and data prep.

The Inkling tokenizer could not be downloaded in the development container, so these tests use
a SYNTHETIC STAND-IN tokenizer (a 500-id byte-level BPE trained on a few local files). The check
against the real tokenizer runs only when REX_TOKENIZER points at Inkling's tokenizer.json (or
its directory).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rex.config import RexConfig
from rex.data.loader import Corpus
from rex.data.prep import iter_texts, prep_corpus
from rex.data.standin import local_documents, train_standin_tokenizer
from rex.data.tokenizer import file_sha256, load_tokenizer, load_tokenizer_for


@pytest.fixture(scope="module")
def standin(tmp_path_factory):
    d = tmp_path_factory.mktemp("standin")
    docs = local_documents(max_files=60)
    tf = train_standin_tokenizer([t for _, t in docs], d / "tokenizer.json")
    return tf, docs


def test_standin_tokenizer_checks(standin):
    tf, _ = standin
    tok = load_tokenizer(tf, unpadded_vocab_size=500, bos_id=498, eot_id=499)
    assert tok.vocab_size == 500
    assert tok.decode(tok.encode("def f(x):\n    return x\n")) == "def f(x):\n    return x\n"
    assert tok.sha256 == file_sha256(tf)
    load_tokenizer(tf.parent, unpadded_vocab_size=500, bos_id=498, eot_id=499, expected_sha256=tok.sha256)
    with pytest.raises(ValueError, match="hash"):
        load_tokenizer(tf, unpadded_vocab_size=500, bos_id=498, eot_id=499, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="ids"):
        load_tokenizer(tf, unpadded_vocab_size=200058, bos_id=498, eot_id=499)
    with pytest.raises(ValueError, match="bos"):
        load_tokenizer(tf, unpadded_vocab_size=500, bos_id=497, eot_id=499)


def test_prep_roundtrip(standin, tmp_path):
    tf, docs = standin
    tok = load_tokenizer(tf, unpadded_vocab_size=500, bos_id=498, eot_id=499)
    texts = [t for _, t in docs[:30]] + [""]
    jl = tmp_path / "in.jsonl"
    jl.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    m = prep_corpus(iter_texts([str(jl)]), tok, tmp_path / "out", shard_tokens=20000, vocab_limit=500, log_every=0)
    assert m["n_docs"] == len(texts) and m["tokenizer_sha256"] == tok.sha256 and m["n_shards"] > 1
    c = Corpus(tmp_path / "out")
    for i, t in enumerate(texts):
        ids = c.tokens(i, 0, c.doc_len(i))
        assert ids.tolist() == tok.encode(t)
        assert c.doc_bytes(i) == len(t.encode("utf-8"))
        assert tok.decode(ids.tolist()) == t
    with pytest.raises(FileExistsError):
        prep_corpus(iter_texts([str(jl)]), tok, tmp_path / "out")


def test_prep_parquet(standin, tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    tf, docs = standin
    tok = load_tokenizer(tf, unpadded_vocab_size=500, bos_id=498, eot_id=499)
    texts = [t for _, t in docs[:5]]
    pq.write_table(pa.table({"text": texts}), tmp_path / "x.parquet")
    assert list(iter_texts([str(tmp_path / "x.parquet")])) == texts


@pytest.mark.skipif(not os.environ.get("REX_TOKENIZER"), reason="set REX_TOKENIZER to Inkling's tokenizer.json")
def test_real_inkling_tokenizer():
    tok = load_tokenizer_for(RexConfig(), os.environ["REX_TOKENIZER"])
    assert tok.vocab_size == 200058
    assert tok.tok.token_to_id("<|begin_of_text|>") == 200028
    assert tok.tok.token_to_id("<|endoftext|>") == 199999
    ids = tok.encode("Hello world")
    assert ids and max(ids) < 199998
