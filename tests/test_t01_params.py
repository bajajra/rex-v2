"""T1: parameter counts equal §3 row by row and §11 for the baselines; count_params agrees with the model."""

import subprocess
import sys

import pytest
import torch

from conftest import CONFIGS, ROOT
from rex.config import RexConfig, load_config
from rex.model import RexModel
from rex.model.params import active_counts, analytic_counts, body_total, embedding_total, model_counts

# §3 table, verbatim.
SPEC_ROWS = {
    "token_embedding": 115_789_824,
    "embedding_norm": 576,
    "attn_wq": 10_616_832,
    "attn_wk": 3_538_944,
    "attn_wv": 3_538_944,
    "attn_wo": 10_616_832,
    "qk_scale": 144,
    "out_gate": 83_088,
    "sublayer_norms": 27_648,
    "depth_readers_attn": 17_856,
    "depth_readers_moe": 13_824,
    "depth_reader_out": 576,
    "final_norm": 576,
    "router_weight": 387_072,
    "router_bias": 672,
    "expert_w_in": 111_476_736,
    "expert_w_out": 55_738_368,
    "cross_u": 27_648,
    "cross_P": 884_736,
    "cross_F": 442_368,
    "cross_G": 442_368,
}
SPEC_BODY = 197_855_808
SPEC_EMB = 115_789_824

# §11 table: (body, embedding)
SPEC_BASELINES = {
    "rex-v2-base.yaml": (197_855_808, 115_789_824),
    "a1.yaml": (196_058_688, 115_789_824),
    "b1.yaml": (78_028_704, 115_789_824),
    "b2.yaml": (190_255_280, 180_117_504),
    "b3.yaml": (186_083_504, 180_117_504),
    "b0.yaml": (186_033_328, 180_117_504),
}


def meta_model(cfg):
    with torch.device("meta"):
        return RexModel(cfg)


def test_base_yaml_equals_defaults():
    assert load_config(CONFIGS / "rex-v2-base.yaml") == RexConfig()


def test_base_rows_analytic():
    c = analytic_counts(load_config(CONFIGS / "rex-v2-base.yaml"))
    for k, v in SPEC_ROWS.items():
        assert c[k] == v, k
    assert all(c[k] == 0 for k in c if k not in SPEC_ROWS)
    assert body_total(c) == SPEC_BODY
    assert embedding_total(c) == SPEC_EMB
    assert body_total(c) < 200_000_000


def test_base_rows_instantiated():
    cfg = load_config(CONFIGS / "rex-v2-base.yaml")
    m = model_counts(meta_model(cfg))
    for k, v in SPEC_ROWS.items():
        assert m[k] == v, k
    assert body_total(m) == SPEC_BODY and embedding_total(m) == SPEC_EMB
    assert m == analytic_counts(cfg)


def test_derived_counts():
    cfg = RexConfig()
    a = active_counts(cfg)
    assert a["stored_active"] == 78_416_448
    assert a["effective_active"] == 156_653_568
    assert a["layer_applications"] == 48
    untied = cfg.replace(model={"tie_embeddings": False})
    c = analytic_counts(untied)
    assert body_total(c) + embedding_total(c) == 429_435_456
    assert body_total(analytic_counts(cfg.replace(model={"ffn": {"n_experts": 29}}))) == 203_841_624
    assert body_total(analytic_counts(cfg.replace(model={"ffn": {"n_experts": 27}}))) == 191_869_992


@pytest.mark.parametrize("fname", list(SPEC_BASELINES))
def test_baselines(fname):
    cfg = load_config(CONFIGS / fname)
    body, emb = SPEC_BASELINES[fname]
    a = analytic_counts(cfg)
    m = model_counts(meta_model(cfg))
    assert m == a
    assert body_total(m) == body
    assert embedding_total(m) == emb


def test_count_params_script():
    paths = [str(CONFIGS / f) for f in SPEC_BASELINES]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "count_params.py"), *paths],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MISMATCH" not in r.stdout
    assert "197,855,808" in r.stdout


@pytest.mark.slow
def test_base_real_instantiation():
    """Allocates the full base model on CPU (~1.25 GB in float32)."""
    torch.manual_seed(0)
    model = RexModel(RexConfig())
    m = model_counts(model)
    assert body_total(m) == SPEC_BODY and embedding_total(m) == SPEC_EMB
    assert sum(p.numel() for p in model.parameters()) == SPEC_BODY + SPEC_EMB
