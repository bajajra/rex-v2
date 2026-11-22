"""§10 metrics and probes on the tiny model (synthetic random ids)."""

import math

import numpy as np
import pytest
import torch

from conftest import make_tiny
from rex.data.loader import collate_sequences
from rex.eval.metrics import DeadExpertTracker, RoutingAccumulator, load_entropy, max_min_ratio
from rex.eval.probes import (attention_distance, context_truncation, eval_loss, evaluate, forced_T_losses,
                             internal_stats, message_ablation, position_buckets, repeat_probe)

BOS = 498


def _batches(n=2, lens=(40, 25, 60), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        seqs = [np.concatenate([[BOS], rng.integers(0, 498, L - 1)]) for L in lens]
        out.append(collate_sequences(seqs, [2.0 * L for L in lens]))
    return out


def test_entropy_and_ratio():
    assert load_entropy(torch.ones(6)) == pytest.approx(1.0)
    assert load_entropy(torch.tensor([1.0, 0, 0, 0])) == pytest.approx(0.0)
    assert max_min_ratio(torch.tensor([1.0, 4.0])) == 4.0 and max_min_ratio(torch.tensor([0.0, 1.0])) == math.inf


def test_routing_accumulator_and_dead_experts():
    m = make_tiny(dtype=torch.float32).train()
    b = _batches(1)[0]
    acc = RoutingAccumulator(m.L, m.mcfg.ffn.n_experts, m.mcfg.ffn.route_scale)
    out = m(b.ids, b.cu, b.pos)
    acc.add(out)
    s = acc.summary()
    assert len(s["moe/load_entropy"]) == m.L
    assert all(0 <= e <= 1 for e in s["moe/load_entropy"])
    # load over the window equals the model's pooled f_mb for one micro-batch
    torch.testing.assert_close(acc.load().float(), out.load.float(), rtol=1e-6, atol=1e-7)
    # core layers report each of the 3 passes
    assert {"L2.p1", "L2.p2", "L2.p3"} <= set(s["moe/load_entropy_per_pass"])
    conf = s["moe/router_confidence"]
    assert all(1 / m.mcfg.ffn.top_k - 1e-6 <= c <= 1 for c in conf)
    tr = DeadExpertTracker(2, 3, warmup_steps=10, patience=20)
    dead = torch.tensor([[True, False, False], [False, False, False]])
    assert tr.update(5, dead) == []
    assert tr.update(29, dead) == []
    assert tr.update(30, dead) == [(0, 0)]


def test_probes_at_init_loop_identity():
    m = make_tiny(dtype=torch.float32)
    bs = _batches()
    L = forced_T_losses(m, bs, chunk_tokens=16)
    # §5.5: at initialisation the T-pass model equals the 1-pass model
    assert L["loss_T1"] == pytest.approx(L["loss_T3"], abs=1e-5)
    assert message_ablation(m, bs, chunk_tokens=16) == pytest.approx(L["loss_T3"], abs=1e-5)
    st = internal_stats(m, bs)
    assert st["gate/gamma_rms/Q/L2/p2"] == 0.0 and st["gate/kappa_mean/M/L5/p3"] == pytest.approx(1.0)
    assert st["moe/pass_overlap/L2/p1-p2"] == pytest.approx(1.0)       # identical routing at init
    assert st["loop/pass_change_relnorm/p2"] == pytest.approx(0.0, abs=1e-6)
    a = st["alpha"]["L1.attn.Q"]
    assert a["sources"] == ["v0", "R"] and a["mean"] == pytest.approx([0.5, 0.5])


def test_probes_after_randomisation():
    m = make_tiny(dtype=torch.float32, randomize=True)
    bs = _batches()
    L = forced_T_losses(m, bs, chunk_tokens=16)
    assert L["loss_T1"] != pytest.approx(L["loss_T3"], abs=1e-4)
    r = eval_loss(m, bs, chunk_tokens=16, per_token=True)
    assert r["nll"].numel() == r["n_targets"] == sum(sum(L - 1 for L in b.seq_lens) for b in bs)
    assert r["bpb"] == pytest.approx(r["nats"] * r["n_targets"] / (math.log(2) * sum(b.bytes_est for b in bs)))
    pb = position_buckets(r["nll"], r["pos"], buckets=(0, 8, 32, 1000))
    assert all(math.isfinite(v) for v in pb.values())
    ad = attention_distance(m, bs)
    assert ad and all(0 <= v <= 1 for v in ad.values())
    assert set(k.split("/")[2] for k in ad) == {"L1", "L3", "L5", "L7"}    # full-attention layers only
    ev = evaluate(m, bs, chunk_tokens=16)
    assert "val/loss_no_message" in ev and "attn/far_mass/L3/app2" in ev


def test_long_context_probes():
    m = make_tiny(dtype=torch.float32, randomize=True)
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 498, 300) for _ in range(2)]
    tr = context_truncation(m, docs, bos_id=BOS, contexts=(8, 64, None), target_len=16, chunk_tokens=32)
    assert set(tr) == {"trunc/ctx_8", "trunc/ctx_64", "trunc/ctx_all"}
    rp = repeat_probe(m, [docs[0][:20]], docs[1], bos_id=BOS, gaps=(10, 50), chunk_tokens=32)
    assert set(rp) == {"repeat/first_gap10", "repeat/repeat_gap10", "repeat/first_gap50", "repeat/repeat_gap50"}
