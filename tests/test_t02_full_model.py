"""Tests that need the full-size base model on CPU (marked slow):
T2 initial loss, and T7's source lists / reader count for the base config."""

import gc
import math

import pytest
import torch

from conftest import packed
from rex.config import RexConfig
from rex.model import RexModel, lm_loss
from test_t05_attnres import check_sources, scale_block


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


@pytest.mark.slow
def test_t6_base_block_scale_invariance(base_model):
    """T6 on the base config at default init (float32, tolerance 1e-4)."""
    ids, cu, pos = packed([24, 16], vocab=base_model.mcfg.unpadded_vocab_size, seed=3)
    with torch.no_grad():
        ref = base_model.logits(base_model(ids, cu, pos).hidden)
        bs = base_model.bs
        for b in range(4):
            saved = [(base_model.layers[l].attn.wo.weight.clone(), base_model.layers[l].ffn.w_out.clone())
                     for l in range(b * bs, (b + 1) * bs)]
            scale_block(base_model, b, 3.0)
            out = base_model.logits(base_model(ids, cu, pos).hidden)
            for l, (wo, wout) in zip(range(b * bs, (b + 1) * bs), saved):
                base_model.layers[l].attn.wo.weight.copy_(wo)
                base_model.layers[l].ffn.w_out.copy_(wout)
            torch.testing.assert_close(out, ref, rtol=0, atol=1e-4)
