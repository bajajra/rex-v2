"""T3 loop identity at init; T4 loop is live."""

import torch

from conftest import make_tiny


def _run(model, batch, T):
    ids, cu, pos = batch
    events = []
    with torch.no_grad():
        out = model(ids, cu, pos, T, recorder=events.append)
    blocks = {(e["name"], e["pass"]): e["value"] for e in events if e["kind"] == "block"}
    return model.logits(out.hidden), blocks


def test_t3_loop_identity_at_init(tiny_batch):
    model = make_tiny().eval()
    l1, _ = _run(model, tiny_batch, 1)
    l2, _ = _run(model, tiny_batch, 2)
    l3, b3 = _run(model, tiny_batch, 3)
    # κ = 1 - tanh(0) = 1 and γ = tanh(0) = 0 exactly, so the passes are bitwise identical.
    assert torch.equal(l1, l2) and torch.equal(l1, l3)
    for t in (2, 3):
        assert torch.equal(b3[("V2", t)], b3[("V2", 1)])
        assert torch.equal(b3[("V3", t)], b3[("V3", 1)])


def test_t4_loop_is_live(tiny_batch):
    model = make_tiny().eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for gm in model.gate_mixes():
            gm.F.weight.copy_(torch.randn(gm.F.weight.shape, generator=g, dtype=torch.float64))
            gm.G.weight.copy_(torch.randn(gm.G.weight.shape, generator=g, dtype=torch.float64))
    l1, _ = _run(model, tiny_batch, 1)
    l3, b = _run(model, tiny_batch, 3)
    assert (l1 - l3).abs().max() > 1e-3
    assert (b[("V3", 1)] - b[("V3", 3)]).abs().max() > 1e-3
    model.set_gate_override("identity")  # γ forced to 0, κ forced to 1
    l1i, _ = _run(model, tiny_batch, 1)
    l3i, _ = _run(model, tiny_batch, 3)
    torch.testing.assert_close(l1i, l3i, rtol=1e-5, atol=0)
    torch.testing.assert_close(l1i, l1, rtol=1e-5, atol=0)  # T = 1 never gates
    model.set_gate_override("no_message")  # message ablation: γ = 0 only
    l3m, _ = _run(model, tiny_batch, 3)
    assert (l3m - l3).abs().max() > 1e-6
