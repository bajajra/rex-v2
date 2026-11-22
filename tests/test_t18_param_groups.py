"""T18: every parameter is in exactly one group; the no-decay list matches §9."""

import re

import pytest
import torch

from conftest import CONFIGS, make_tiny
from rex.config import load_config
from rex.model import RexModel
from rex.train.optim import build_optimizer, param_groups, zeropower_via_newtonschulz5

# §9 weight decay 0: RMSNorm gains, biases, QK scales, depth queries, cross-pass u, P, F, G.
NO_DECAY = re.compile(
    r"(norm\.weight$|\.bias$|\.router_bias$|\.qk_scale$|(^|\.)mix_\w+\.q$|^mix_out\.q$|\.cross\.[QKVM]\.(u|P\.weight|F\.weight|G\.weight)$)")
# §9 weight decay 0.1: Emb, attention, experts, router weight, output-gate weight (+ dense FFN).
DECAY = re.compile(r"(^emb\.weight$|^lm_head\.weight$|\.attn\.w[qkvo]\.weight$|\.ffn\.w_(in|out)(\.weight)?$|"
                   r"\.ffn\.router_weight$|\.attn\.gate\.weight$)")
MUON = re.compile(r"(\.attn\.w[qkvo]\.weight$|\.ffn\.w_(in|out)$)")


def _check(model, use_muon=False):
    groups = param_groups(model, 0.1, use_muon=use_muon)
    seen = {}
    for g in groups:
        for n, p in zip(g["names"], g["params"]):
            assert n not in seen, f"{n} in two groups"
            seen[n] = g["name"]
    names = dict(model.named_parameters())
    assert set(seen) == set(names), "every parameter must be in exactly one group"
    assert sum(len(g["params"]) for g in groups) == len(names)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    for n, g in seen.items():
        if NO_DECAY.search(n):
            assert g == "no_decay", n
        else:
            assert DECAY.search(n), f"{n} matches neither list"
            assert g == ("muon" if use_muon and MUON.search(n) else "decay"), (n, g)
    for g in groups:
        assert g["weight_decay"] == (0.0 if g["name"] == "no_decay" else 0.1)
    return seen


@pytest.mark.parametrize("cfg", ["rex-v2-base", "a1", "b0", "b1", "b2", "b3"])
@pytest.mark.parametrize("use_muon", [False, True])
def test_t18_groups_full_configs(cfg, use_muon):
    c = load_config(CONFIGS / f"{cfg}.yaml")
    with torch.device("meta"):
        m = RexModel(c)
    seen = _check(m, use_muon)
    if cfg == "rex-v2-base":
        nd = [n for n, g in seen.items() if g == "no_decay"]
        # 24 layers x (2 norm gains + qk_scale + gate bias + router bias) + emb/final norm
        # + 56 depth queries + 12 core layers x 4 readers x (u, P, F, G)
        assert len(nd) == 24 * 5 + 2 + 56 + 12 * 4 * 4


def test_t18_tiny_untied_and_dense():
    _check(make_tiny(model={"tie_embeddings": False}))
    _check(make_tiny(model={"ffn": {"type": "dense"}}))
    _check(make_tiny(model={"residual": "stream", "n_passes": 1}))


def test_t18_optimizer_builds_and_steps():
    for opt in ("adamw", "muon"):
        m = make_tiny(dtype=torch.float32, train={"optimizer": opt})
        o = build_optimizer(m, m.cfg.train)
        names = [g["name"] for g in o.param_groups]
        assert ("muon" in names) == (opt == "muon")
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 1e-2
        o.step()
        moved = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
        assert len(moved) == len(before)
        o.load_state_dict(o.state_dict())


def test_muon_newton_schulz_orthogonalises():
    torch.manual_seed(0)
    G = torch.randn(3, 32, 16)               # batched like expert slices
    X = zeropower_via_newtonschulz5(G)
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.3   # quintic NS maps singular values into ~[0.7, 1.2]
