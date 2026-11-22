"""T8 causality, T9 window, T10 sequence isolation, T11 kernel equivalence (GPU), T12 RoPE;
plus length scaling and the CPU fallback of the flash backend."""

import math
import warnings

import pytest
import torch

from conftest import make_tiny, packed
from rex.config import tiny_config
from rex.model import attn_backends
from rex.model.attention import Attention
from rex.model.rope import rope_tables, rope_tables_f64


def _forward_all(model, ids, cu, pos, T):
    events = []
    with torch.no_grad():
        out = model(ids, cu, pos, T, recorder=events.append)
    blocks = {(e["name"], e["pass"]): e["value"] for e in events if e["kind"] == "block"}
    return model.logits(out.hidden), blocks


# ---------------------------------------------------------------- T8
# Exact (torch.equal) comparison is run under ffn.dispatch: dense_ref, whose per-token
# arithmetic does not depend on other tokens. Under sorted dispatch, changing token j changes
# the per-expert slice sizes and hence BLAS blocking, which can move other rows by ~1e-12
# without any information flow; sorted == dense_ref is checked with a tolerance in T14
# (test_t13_moe.py) and by test_t8_causality_sorted_dispatch below.
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("j", [0, 5, 17, 29])
def test_t8_causality(j, train):
    model = make_tiny(randomize=True, model={"ffn": {"dispatch": "dense_ref"}}).train(train)
    ids, cu, pos = packed([30])
    ids2 = ids.clone()
    ids2[j] = (ids[j] + 1) % 500
    for T in (1, 2, 3):
        a, ba = _forward_all(model, ids, cu, pos, T)
        b, bb = _forward_all(model, ids2, cu, pos, T)
        assert torch.equal(a[:j], b[:j])
        for key in ba:  # block sums of every pass
            assert torch.equal(ba[key][:j], bb[key][:j]), key
        assert (a[j:] - b[j:]).abs().max() > 0  # the change is visible from j on


@pytest.mark.parametrize("j", [0, 11, 29])
def test_t8_causality_sorted_dispatch(j):
    model = make_tiny(randomize=True).eval()
    ids, cu, pos = packed([30])
    ids2 = ids.clone()
    ids2[j] = (ids[j] + 1) % 500
    for T in (1, 2, 3):
        a, ba = _forward_all(model, ids, cu, pos, T)
        b, bb = _forward_all(model, ids2, cu, pos, T)
        torch.testing.assert_close(a[:j], b[:j], rtol=0, atol=1e-9)
        for key in ba:
            torch.testing.assert_close(ba[key][:j], bb[key][:j], rtol=0, atol=1e-9)


# ---------------------------------------------------------------- T9
def _windowed_attention(dtype=torch.float64):
    cfg = tiny_config().model
    torch.manual_seed(0)
    att = Attention(cfg, layer_idx=0, max_len=262144).to(dtype)
    assert not att.is_global and att.window == 8
    with torch.no_grad():
        att.gate.weight.normal_(0, 0.3)
    return att


def test_t9_window():
    att = _windowed_attention()
    N, d, w = 40, 64, att.window
    cu = torch.tensor([0, N], dtype=torch.int32)
    pos = torch.arange(N)
    x = torch.randn(3, N, d, dtype=torch.float64)
    with torch.no_grad():
        ref = att(x[0], x[1], x[2], cu, pos)
        for j in (0, 10, 25):
            x2 = x.clone()
            x2[:, j] += torch.randn(3, d, dtype=torch.float64)
            out = att(x2[0], x2[1], x2[2], cu, pos)
            # positions i with j <= i - window are unaffected
            assert torch.equal(out[j + w:], ref[j + w:])
            assert (out[j:j + w] - ref[j:j + w]).abs().amax(-1).min() > 0


# ---------------------------------------------------------------- T10
def test_t10_sequence_isolation():
    model = make_tiny(randomize=True).eval()
    lens = [17, 11, 23]
    ids, cu, pos = packed(lens)
    with torch.no_grad():
        full = model.logits(model(ids, cu, pos).hidden)
        # packed result equals each sequence alone
        for s in range(3):
            a, b = int(cu[s]), int(cu[s + 1])
            cu1 = torch.tensor([0, b - a], dtype=torch.int32)
            alone = model.logits(model(ids[a:b], cu1, pos[a:b]).hidden)
            torch.testing.assert_close(alone, full[a:b], rtol=1e-5, atol=1e-12)
        # changing any token of sequence A (index 1) changes nothing in sequences 0 and 2
        ids2 = ids.clone()
        ids2[int(cu[1]):int(cu[2])] = (ids2[int(cu[1]):int(cu[2])] + 7) % 500
        other = model.logits(model(ids2, cu, pos).hidden)
    for s in (0, 2):
        a, b = int(cu[s]), int(cu[s + 1])
        assert torch.equal(other[a:b], full[a:b])


# ---------------------------------------------------------------- T11
@pytest.mark.gpu
@pytest.mark.skipif(not attn_backends.flash_available(), reason="needs CUDA and flash-attn")
def test_t11_kernel_equivalence():  # pragma: no cover - not runnable on CPU
    torch.manual_seed(0)
    dev = "cuda"
    hq, hkv, dh = 6, 2, 128
    lens = [700, 1500, 64]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    from rex.model import packed_positions
    pos = packed_positions(cu)
    N = int(cu[-1])
    q = torch.randn(N, hq, dh, device=dev, dtype=torch.bfloat16)
    k = torch.randn(N, hkv, dh, device=dev, dtype=torch.bfloat16)
    v = torch.randn(N, hkv, dh, device=dev, dtype=torch.bfloat16)
    for window in (1024, None):
        ref = attn_backends.reference_varlen(q, k, v, cu, pos, window).float()
        fl = attn_backends.flash_varlen(q, k, v, cu, pos, window).float()
        assert (ref - fl).abs().max() <= 2e-2


def test_flash_backend_falls_back_on_cpu():
    torch.manual_seed(0)
    q = torch.randn(20, 2, 32)
    k = torch.randn(20, 1, 32)
    v = torch.randn(20, 1, 32)
    cu = torch.tensor([0, 12, 20], dtype=torch.int32)
    from rex.model import packed_positions
    pos = packed_positions(cu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = attn_backends.varlen_attention("flash", q, k, v, cu, pos, 8)
    torch.testing.assert_close(out, attn_backends.reference_varlen(q, k, v, cu, pos, 8))


# ---------------------------------------------------------------- T12
def test_t12_rope_relative():
    att = _windowed_attention(torch.float32)
    N, d = 30, 64
    cu = torch.tensor([0, N], dtype=torch.int32)
    pos = torch.arange(N)
    x = torch.randn(N, d)
    with torch.no_grad():
        a = att(x, x, x, cu, pos)
        b = att(x, x, x, cu, pos + 100_000)
    torch.testing.assert_close(b, a, rtol=0, atol=1e-3)


def test_t12_tables_built_in_float64():
    c64, s64 = rope_tables_f64(10000.0, 128, 262144)
    assert c64.dtype == torch.float64 and c64.shape == (262144, 64)
    c32, s32 = rope_tables(10000.0, 128, 262144, "cpu", torch.float32)
    assert torch.equal(c32, c64.float()) and torch.equal(s32, s64.float())
    # float32 angles would be measurably wrong at large positions
    pos = torch.arange(262144, dtype=torch.float32)[-1]
    ang32 = pos * (10000.0 ** (-torch.arange(64, dtype=torch.float32) / 64))
    assert (torch.cos(ang32) - c64[-1].float()).abs().max() > 1e-3


def test_length_scaling_factor():
    cfg = tiny_config().model
    att = Attention(cfg, layer_idx=1, max_len=262144).double()  # full-attention layer
    assert att.is_global and att.window is None and not att.use_rope
    with torch.no_grad():
        att.qk_scale.fill_(1.0)
    N = 4
    pos = torch.tensor([0, 8191, 65535, 262143])
    x = torch.randn(N, 64, dtype=torch.float64)
    q, _, _ = att.project(x, x, x, pos)
    q_unit, _, _ = att.project(x, x, x, torch.zeros(N, dtype=torch.long))
    lam = (q.norm(dim=-1) / q_unit.norm(dim=-1))[:, 0]
    exp = torch.tensor([1.0, 1.0, 1 + 0.1 * math.log(8), 1 + 0.1 * math.log(32)], dtype=torch.float64)
    torch.testing.assert_close(lam, exp)
