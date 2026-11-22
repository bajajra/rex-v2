"""T5 DepthMix; T6 block-scale invariance; T7 source lists."""

import torch

from conftest import make_tiny, packed
from rex.model.attnres import DepthMix, depth_mix
from rex.model.norm import nrm


# ---------------------------------------------------------------- T5
def test_t5_uniform_at_zero_query():
    torch.manual_seed(0)
    dm = DepthMix(16).double()
    for n in (1, 2, 3, 5):
        srcs = [nrm(torch.randn(7, 16, dtype=torch.float64), 1e-6) for _ in range(n)]
        out, alpha = dm(srcs, return_weights=True)
        assert torch.equal(alpha, torch.full_like(alpha, 1.0 / n))
        torch.testing.assert_close(out, torch.stack(srcs).mean(0), rtol=1e-12, atol=1e-12)


def test_t5_weights_sum_to_one_and_scale_invariance():
    torch.manual_seed(1)
    d = 16
    q = torch.randn(d, dtype=torch.float64) * 3
    srcs = [torch.randn(9, d, dtype=torch.float64) * (i + 1) for i in range(4)]
    dm = DepthMix(d).double()
    with torch.no_grad():
        dm.q.copy_(q)
    _, alpha = dm([nrm(s, 1e-6) for s in srcs], return_weights=True)
    torch.testing.assert_close(alpha.sum(-1), torch.ones(9, dtype=torch.float64), rtol=1e-12, atol=0)
    ref = depth_mix(q, srcs, 1e-6)
    for j in range(4):
        scaled = list(srcs)
        scaled[j] = srcs[j] * 7.3
        # eps = 1e-6 makes the invariance approximate at the ~1e-6 level for unit-RMS sources
        torch.testing.assert_close(depth_mix(q, scaled, 1e-6), ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- T6
# §14 T6: tiny model at default init, float64, rms_eps 1e-12 for this test only (the
# invariance is exact only as eps -> 0, §5.3), tolerance 1e-8, no weight rescaling.
def scale_block(model, b, factor):
    bs = model.bs
    with torch.no_grad():
        for l in range(b * bs, (b + 1) * bs):
            model.layers[l].attn.wo.weight.mul_(factor)
            w_out = model.layers[l].ffn.w_out
            (w_out if isinstance(w_out, torch.Tensor) else w_out.weight).mul_(factor)


def _logits64(model, hidden):
    # model.logits() returns float32 by §5.8, whose ulp (~6e-8) exceeds the 1e-8 tolerance;
    # the same softcapped head is evaluated here in float64.
    w = model.output_weight()[: model.mcfg.unpadded_vocab_size]
    c = model.mcfg.final_logit_softcap
    return c * torch.tanh(torch.nn.functional.linear(hidden, w) / c)


def test_t6_block_scale_invariance(tiny_batch):
    ids, cu, pos = tiny_batch
    eps = {"model": {"rms_eps": 1e-12}}
    base = make_tiny(dtype=torch.float64, **eps).eval()
    with torch.no_grad():
        ref = _logits64(base, base(ids, cu, pos).hidden)
    for b in range(4):  # prelude, both core blocks, coda
        m = make_tiny(dtype=torch.float64, **eps).eval()
        scale_block(m, b, 3.0)
        with torch.no_grad():
            out = _logits64(m, m(ids, cu, pos).hidden)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-8)


# ---------------------------------------------------------------- T7
def expected_sources(cfg, T):
    """{(reader, pass): [source names]} from the §5.3 table."""
    m = cfg.model
    bs, glob = m.block_size, set(m.attn.global_layers)
    exp = {}
    for l in range(m.n_layers):
        b = l // bs
        passes = list(range(1, T + 1)) if b in (1, 2) else [None]
        for t in passes:
            done = [["v0"], ["v0", "V1"], ["v0", "V1", f"V2[{t}]"], ["v0", "V1", f"V2[{T}]", f"V3[{T}]"]][b]
            first = l % bs == 0
            attn_src = done + ([] if first else ["R"])
            if l in glob:
                for s in "QKV":
                    exp[(f"L{l}.attn.{s}", t)] = attn_src
            elif l != 0:
                exp[(f"L{l}.attn", t)] = attn_src
            exp[(f"L{l}.moe", t)] = done + ["R"]
    exp[("out", None)] = ["v0", "V1", f"V2[{T}]", f"V3[{T}]", "V4"]
    return exp


def check_sources(cfg, events, T):
    got = {}
    for e in events:
        if e["kind"] == "depthmix":
            key = (e["reader"], e["pass"])
            assert key not in got, f"reader {key} fired twice"
            got[key] = e["sources"]
            assert e["alpha"].shape[-1] == len(e["sources"])
    assert got == expected_sources(cfg, T)
    return {r for r, _ in got}


def test_t7_source_lists_tiny():
    model = make_tiny().eval()
    ids, cu, pos = packed([10, 7])
    for T in (1, 2, 3):
        events = []
        with torch.no_grad():
            model(ids, cu, pos, T, recorder=events.append)
        readers = check_sources(model.cfg, events, T)
        assert len(readers) == len(model.depth_mixes())
        # gated readers fire exactly in passes 2..T of core layers
        gates = {(e["reader"], e["pass"]) for e in events if e["kind"] == "gatemix"}
        assert gates == {(f"L{l}.cross.{s}", t) for l in range(2, 6) for s in "QKVM" for t in range(2, T + 1)}
    # tiny analogue of "31 + 24 + 1": 7 + 2*4 attention readers, 8 MoE readers, 1 output reader
    assert len(model.depth_mixes()) == 15 + 8 + 1


def test_t7_base_reader_count_meta():
    from rex.config import RexConfig
    from rex.model import RexModel
    with torch.device("meta"):
        m = RexModel(RexConfig())
    assert len(m.depth_mixes()) == 56
