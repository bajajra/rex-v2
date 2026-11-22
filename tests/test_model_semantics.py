"""Behavioural checks of individual spec formulas that the T-numbered tests do not pin down.
Each test names the passage of docs/DESIGN.md it checks against."""

import math

import torch
import torch.nn.functional as F

from conftest import make_tiny, packed
from rex.config import tiny_config
from rex.model import lm_loss, next_token_targets
from rex.model.attention import Attention
from rex.model.attn_backends import reference_varlen
from rex.model.loop import GateMix
from rex.model.norm import nrm


def _ref_nrm(x, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# ---------------------------------------------------------------- §5.8 targets
def test_next_token_targets_and_lm_loss():
    """§5.8: L_CE is the mean cross-entropy over positions that have a next token in the same sequence."""
    ids = torch.tensor([5, 6, 7, 8, 9, 10])
    cu = torch.tensor([0, 3, 4, 6], dtype=torch.int32)
    assert next_token_targets(ids, cu).tolist() == [6, 7, -100, -100, 10, -100]

    model = make_tiny().eval()
    ids, cu, pos = packed([9, 1, 6])
    with torch.no_grad():
        ce, out = lm_loss(model, ids, cu, pos)
        logits = model.logits(out.hidden)
    terms = [F.cross_entropy(logits[i:i + 1], ids[i + 1:i + 2])
             for s in range(3) for i in range(int(cu[s]), int(cu[s + 1]) - 1)]
    torch.testing.assert_close(ce, torch.stack(terms).mean())


# ---------------------------------------------------------------- §5.8 softcap
def test_logits_softcap():
    """§5.8: logits = hfinal @ Emb^T over columns 0..unpadded-1, then 30·tanh(x/30), float32."""
    model = make_tiny()
    h = torch.randn(7, 64, dtype=torch.float64) * 200.0  # raw logits far beyond ±30
    with torch.no_grad():
        got = model.logits(h)
        raw = h @ model.emb.weight[:500].t()
    assert got.dtype == torch.float32 and got.shape == (7, 500)
    assert raw.abs().max() > 60
    torch.testing.assert_close(got, (30.0 * torch.tanh(raw / 30.0)).float())
    assert got.abs().max() <= 30.0


# ---------------------------------------------------------------- §5.5 GateMix
def test_gatemix_matches_spec_formula():
    """§5.5: b_j = v̂_j·u/sqrt(d); β = softmax; m = Σβ v̂; z = P·[nrm(x), nrm(m)];
    κ = 1 - tanh(Fz); γ = tanh(Gz); out = κ x + γ m. x deliberately not unit-RMS."""
    torch.manual_seed(0)
    d, r, N = 24, 4, 11
    g = GateMix(d, r, 1e-6).double()
    with torch.no_grad():
        for p in (g.u, g.F.weight, g.G.weight):
            p.normal_(0, 0.5)
        g.P.weight.normal_(0, 0.5)
    x = torch.randn(N, d, dtype=torch.float64) * 4.0 + 1.0
    v2 = _ref_nrm(torch.randn(N, d, dtype=torch.float64))
    v3 = _ref_nrm(torch.randn(N, d, dtype=torch.float64))
    b = torch.stack([v2 @ g.u, v3 @ g.u], -1) / math.sqrt(d)
    beta = torch.softmax(b, -1)
    m = beta[:, :1] * v2 + beta[:, 1:] * v3
    z = torch.cat([_ref_nrm(x), _ref_nrm(m)], -1) @ g.P.weight.t()
    exp = (1 - torch.tanh(z @ g.F.weight.t())) * x + torch.tanh(z @ g.G.weight.t()) * m
    with torch.no_grad():
        torch.testing.assert_close(g(x, (v2, v3)), exp, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------- §5.3 / §5.4 sources by value
def test_readers_receive_the_named_tensors():
    """§5.3 table and §5.4 schedule, checked on values rather than names: every DepthMix gets
    nrm of the block sums its source list names (V2[t], V3[t] of the right pass), and every
    GateMix in pass t gets prev = (nrm(V2[t-1]), nrm(V3[t-1]))."""
    model = make_tiny(randomize=True).eval()  # live loop: block sums differ between passes
    ids, cu, pos = packed([13, 9])
    events, mix_inputs, gate_inputs = [], [], []
    hooks = [dm.register_forward_hook(lambda mod, args, out: mix_inputs.append(list(args[0])))
             for dm in model.depth_mixes()]
    hooks += [gm.register_forward_hook(lambda mod, args, out: gate_inputs.append(args[1]))
              for gm in model.gate_mixes()]
    with torch.no_grad():
        model(ids, cu, pos, recorder=events.append)
        v0 = model.emb_norm(model.emb(ids))
    for h in hooks:
        h.remove()
    eps = model.eps
    vals = {"v0": nrm(v0, eps)}
    for e in events:
        if e["kind"] == "block":
            vals[e["name"] if e["pass"] is None else f"{e['name']}[{e['pass']}]"] = nrm(e["value"], eps)
    mix_events = [e for e in events if e["kind"] == "depthmix"]
    assert len(mix_events) == len(mix_inputs)
    # sanity: the passes really differ, so a wrong-pass or wrong-block source would be detected
    assert (vals["V2[1]"] - vals["V2[3]"]).abs().max() > 1e-3
    assert (vals["V1"] - vals["V2[1]"]).abs().max() > 1e-3
    for e, srcs in zip(mix_events, mix_inputs):
        assert len(srcs) == len(e["sources"])
        for name, t in zip(e["sources"], srcs):
            if name != "R":
                torch.testing.assert_close(t, vals[name], rtol=0, atol=1e-12, msg=f"{e['reader']} {name}")
    gate_events = [e for e in events if e["kind"] == "gatemix"]
    assert len(gate_events) == len(gate_inputs) > 0
    for e, (p2, p3) in zip(gate_events, gate_inputs):
        t = e["pass"]
        torch.testing.assert_close(p2, vals[f"V2[{t - 1}]"], rtol=0, atol=1e-12)
        torch.testing.assert_close(p3, vals[f"V3[{t - 1}]"], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- §5.6 attention
def _attn(layer_idx):
    torch.manual_seed(0)
    att = Attention(tiny_config().model, layer_idx=layer_idx, max_len=262144).double()
    with torch.no_grad():
        att.qk_scale.copy_(torch.tensor([1.44, 0.7], dtype=torch.float64))
        att.gate.weight.normal_(0, 0.5)
        att.gate.bias.normal_(0, 0.5)
    return att


def test_qk_norm():
    """§5.6: q = nrm(q), k = nrm(k) per head over head_dim, then q *= s[h] (RoPE preserves norms)."""
    for layer_idx in (0, 1):  # windowed (RoPE) and full-attention (λ = 1 at pos < 8192)
        att = _attn(layer_idx)
        x = torch.randn(10, 64, dtype=torch.float64) * 7.0
        q, k, _ = att.project(x, x, x, torch.arange(10))
        rms_q = q.pow(2).mean(-1).sqrt()
        rms_k = k.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms_q, att.qk_scale.detach().expand(10, 2), rtol=1e-5, atol=0)
        torch.testing.assert_close(rms_k, torch.ones_like(rms_k), rtol=1e-5, atol=0)


def test_attention_output_gate():
    """§5.6: a = a · (2·sigmoid(W_g nQ + b_g)) per head; o = W_O a."""
    att = _attn(0)
    N = 12
    cu = torch.tensor([0, 7, N], dtype=torch.int32)
    pos = torch.tensor(list(range(7)) + list(range(5)))
    x = torch.randn(3, N, 64, dtype=torch.float64)
    with torch.no_grad():
        q, k, v = att.project(x[0], x[1], x[2], pos)
        a = reference_varlen(q, k, v, cu, pos, att.window)
        gate = 2.0 * torch.sigmoid(x[0] @ att.gate.weight.t() + att.gate.bias)
        exp = (a * gate[:, :, None]).reshape(N, -1) @ att.wo.weight.t()
        torch.testing.assert_close(att(x[0], x[1], x[2], cu, pos), exp, rtol=1e-10, atol=1e-12)
        # identity at init (§5.6: W_g = 0, b_g = 0 gives gate 1)
        att.gate.weight.zero_()
        att.gate.bias.zero_()
        torch.testing.assert_close(att(x[0], x[1], x[2], cu, pos),
                                   a.reshape(N, -1) @ att.wo.weight.t(), rtol=1e-10, atol=1e-12)
