"""Tests that need the full-size base model on CPU (marked slow):
T2 initial loss, and T7's source lists / reader count for the base config."""

import gc
import math

import pytest
import torch

from conftest import packed
from rex.config import RexConfig
from rex.model import RexModel, lm_loss
from test_t05_attnres import check_sources


@pytest.fixture(scope="module")
def base_model():
    torch.manual_seed(0)
    cfg = RexConfig().replace(model={"attn": {"backend": "reference"}})
    m = RexModel(cfg).eval()
    yield m
    del m
    gc.collect()


@pytest.mark.slow
def test_t2_initial_loss(base_model):
    ids, cu, pos = packed([256, 192], vocab=base_model.mcfg.unpadded_vocab_size, seed=1)
    with torch.no_grad():
        ce, out = lm_loss(base_model, ids, cu, pos)
    # §5.9 expects about ln(200,058) + 0.12 = 12.32
    assert 12.1 <= ce.item() <= 12.6, ce.item()
    assert math.isfinite(out.aux_loss.item())


@pytest.mark.slow
def test_t7_base_sources_and_reader_count(base_model):
    assert len(base_model.depth_mixes()) == 56
    events = []
    ids, cu, pos = packed([12, 9], vocab=base_model.mcfg.unpadded_vocab_size, seed=2)
    with torch.no_grad():
        base_model(ids, cu, pos, recorder=events.append)
    readers = check_sources(base_model.cfg, events, T=3)
    assert len(readers) == 56

