"""T22 baselines (B0 stream runs T2, T8, T20; validation rejects stream with n_passes > 1),
plus config loading and validation."""

import gc

import pytest
import torch

from conftest import CONFIGS, make_tiny, packed
from rex.config import RexConfig, load_config, tiny_config
from rex.model import RexModel, lm_loss, next_token_targets

STREAM = {"model": {"residual": "stream", "n_passes": 1, "ffn": {"type": "dense"}}}


def test_validation_rejects_stream_with_loop():
    with pytest.raises(ValueError, match="stream requires n_passes: 1"):
        RexConfig().replace(model={"residual": "stream"})
    with pytest.raises(ValueError):
        load_config(CONFIGS / "b0.yaml").replace(model={"n_passes": 3})


@pytest.mark.parametrize("bad", [
    {"model": {"n_layers": 20}},
    {"model": {"core_start": 4}},
    {"model": {"core_end": 12}},
    {"model": {"attn": {"n_kv_heads": 4}}},
    {"model": {"ffn": {"top_k": 29}}},
])
def test_validation_rejects_bad_configs(bad):
    with pytest.raises(ValueError):
        RexConfig().replace(**bad)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text((CONFIGS / "rex-v2-base.yaml").read_text().replace("n_passes: 3", "n_passes: 3\n  n_pases: 3"))
    with pytest.raises(KeyError):
        load_config(p)


@pytest.mark.parametrize("name", ["rex-v2-base", "a1", "b0", "b1", "b2", "b3", "smoke"])
def test_all_configs_load(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg.to_dict()["model"]["d_model"] in (576, 896)


def test_b0_stream_has_no_readers():
    m = make_tiny(**STREAM)
    assert m.mix_out is None and not m.depth_mixes() and not m.gate_mixes()


def test_t22_b0_causality():
    model = make_tiny(**STREAM).eval()
    ids, cu, pos = packed([30])
    with torch.no_grad():
        a = model.logits(model(ids, cu, pos).hidden)
        for j in (3, 20):
            ids2 = ids.clone()
            ids2[j] = (ids[j] + 1) % 500
            b = model.logits(model(ids2, cu, pos).hidden)
            assert torch.equal(a[:j], b[:j]) and (a[j:] - b[j:]).abs().max() > 0


def test_t22_b0_overfit():
    from test_t19_training import _overfit
    torch.manual_seed(0)
    model = make_tiny(dtype=torch.float32, **STREAM).train()
    batch = packed([64, 64, 64, 64], vocab=model.mcfg.unpadded_vocab_size, seed=5)
    assert _overfit(model, batch) < 0.05


def test_dense_attnres_baseline_runs():
    model = make_tiny(model={"ffn": {"type": "dense"}})  # tiny analogue of B1/B2
    ids, cu, pos = packed([12, 9])
    ce, out = lm_loss(model, ids, cu, pos)
    assert out.load is None and out.aux_loss.item() == 0.0
    ce.backward()


@pytest.mark.slow
def test_t22_b0_initial_loss():
    """B0 at full size (d=896, ~366M parameters, ~1.5 GB in float32)."""
    torch.manual_seed(0)
    cfg = load_config(CONFIGS / "b0.yaml").replace(model={"attn": {"backend": "reference"}})
    model = RexModel(cfg).eval()
    ids, cu, pos = packed([256, 192], vocab=cfg.model.unpadded_vocab_size, seed=1)
    with torch.no_grad():
        out = model(ids, cu, pos)
        logits = model.logits(out.hidden)
    del model
    gc.collect()
    tgt = next_token_targets(ids, cu)
    ce = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=-100)
    # §14 T22 / §11: as is within [12.1, 13.0]; with each position's logit for its own input
    # token id excluded from the softmax within [12.1, 12.6].
    assert 12.1 <= ce.item() <= 13.0, ce.item()
    masked = logits.scatter(1, ids[:, None], float("-inf"))
    keep = tgt != ids  # a target equal to the input token would have its own logit removed
    ce_ex = torch.nn.functional.cross_entropy(masked[keep], tgt[keep], ignore_index=-100)
    assert 12.1 <= ce_ex.item() <= 12.6, ce_ex.item()
