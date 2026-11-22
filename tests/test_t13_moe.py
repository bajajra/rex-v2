"""T13 router, T14 dispatch, T15 auxiliary losses (+ load buffer and bias-balancing plumbing)."""

import pytest
import torch

from conftest import make_tiny, packed
from rex.config import FFNConfig, LossConfig
from rex.model.moe import MoE


def base_moe(dispatch="sorted", balance="aux"):
    torch.manual_seed(0)
    return MoE(576, FFNConfig(dispatch=dispatch), LossConfig(balance=balance))


# ---------------------------------------------------------------- T13
def test_t13_router_invariants():
    moe = base_moe()
    n = torch.randn(64, 576)
    logits, top_i, w = moe.route(n)
    assert logits.dtype == torch.float32
    assert top_i.shape == (64, 8)
    assert all(len(set(row.tolist())) == 8 for row in top_i)
    torch.testing.assert_close(w.sum(-1), torch.full((64,), 8.0))


def test_t13_no_gradient_to_unselected_experts():
    moe = base_moe()
    n = torch.randn(1, 576)
    out, _ = moe(n)
    out.sum().backward()
    _, top_i, _ = moe.route(n)
    sel = set(top_i[0].tolist())
    for e in range(28):
        nz_in = moe.w_in.grad[e].abs().sum() > 0
        nz_out = moe.w_out.grad[e].abs().sum() > 0
        nz_r = moe.router_weight.grad[e].abs().sum() > 0
        assert nz_in == nz_out == nz_r == (e in sel), e


# ---------------------------------------------------------------- T14
def test_t14_sorted_equals_dense_ref():
    torch.manual_seed(0)
    cfg_s, cfg_d = FFNConfig(n_experts=6, top_k=2, expert_hidden=16), FFNConfig(n_experts=6, top_k=2, expert_hidden=16, dispatch="dense_ref")
    a = MoE(32, cfg_s, LossConfig()).double()
    b = MoE(32, cfg_d, LossConfig()).double()
    b.load_state_dict(a.state_dict())
    x = torch.randn(50, 32, dtype=torch.float64)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya, sa = a(xa)
    yb, sb = b(xb)
    torch.testing.assert_close(ya, yb, rtol=1e-10, atol=1e-12)
    g = torch.randn_like(ya)
    (ya * g).sum().backward()
    (yb * g).sum().backward()
    torch.testing.assert_close(xa.grad, xb.grad, rtol=1e-10, atol=1e-12)
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-10, atol=1e-12, msg=na)


def test_t14_full_model_dispatch_equivalence(tiny_batch):
    ids, cu, pos = tiny_batch
    a = make_tiny(randomize=True)
    b = make_tiny(randomize=True, model={"ffn": {"dispatch": "dense_ref"}})
    b.load_state_dict(a.state_dict())
    oa, ob = a(ids, cu, pos), b(ids, cu, pos)
    torch.testing.assert_close(oa.hidden, ob.hidden, rtol=1e-10, atol=1e-12)
    (oa.hidden.sum() + oa.aux_loss).backward()
    (ob.hidden.sum() + ob.aux_loss).backward()
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None:
            assert pb.grad is None
            continue
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-8, atol=1e-12, msg=na)


# ---------------------------------------------------------------- T15
def test_t15_lb_uniform_is_one():
    moe = base_moe()
    N, E, k = 280, 28, 8
    counts = torch.full((E,), N * k / E)
    prob_sum = torch.full((E,), N / E)
    lb, _, f = moe.aux_losses(counts, prob_sum, torch.tensor(0.0), torch.tensor(float(N)), rho=0.0, update=False)
    torch.testing.assert_close(f, torch.full((E,), 1 / E))
    torch.testing.assert_close(lb, torch.tensor(1.0))


def test_t15_lb_collapsed_is_e_over_k():
    moe = base_moe().double()
    with torch.no_grad():
        moe.router_weight.zero_()
        moe.router_bias.zero_()
        moe.router_bias[[1, 4, 6, 9, 13, 17, 22, 27]] = 60.0  # all mass evenly on the same 8 experts
    out, st = moe(torch.randn(40, 576, dtype=torch.float64))
    lb, lz, _ = moe.aux_losses(st["counts"], st["prob_sum"], st["z_sum"], st["n_tokens"], rho=0.0, update=False)
    torch.testing.assert_close(lb, torch.tensor(3.5, dtype=torch.float64), rtol=1e-5, atol=0)
    torch.testing.assert_close(lz, torch.tensor((60.0 + torch.log(torch.tensor(8.0, dtype=torch.float64)).item()) ** 2,
                                                dtype=torch.float64), rtol=1e-6, atol=0)


def test_t15_core_layer_pools_all_passes(tiny_batch):
    ids, cu, pos = tiny_batch
    model = make_tiny(randomize=True, loss={"lb_load_ema": 0.0}).eval()
    N, k = ids.numel(), model.mcfg.ffn.top_k
    out = model(ids, cu, pos, return_routing=True)
    apps = {}
    for r in out.routing:
        apps.setdefault(r["layer"], []).append(r)
    for l in range(model.L):
        core = 2 <= l < 6
        assert sorted(r["pass"] for r in apps[l]) == ([1, 2, 3] if core else [None])
        counts = torch.stack([r["counts"] for r in apps[l]]).sum(0)
        torch.testing.assert_close(out.load[l], counts / (len(apps[l]) * N * k))
        assert out.load[l].sum().item() == pytest.approx(1.0)
    assert out.lb_loss.shape == (8,) and out.z_loss.shape == (8,)
    torch.testing.assert_close(out.aux_loss, 0.01 * out.lb_loss.sum() + 0.001 * out.z_loss.sum())


def test_load_ema_buffer_update():
    ids, cu, pos = packed([25, 15])
    model = make_tiny(randomize=True)  # rho = 0.9
    E = model.mcfg.ffn.n_experts
    model.eval()
    out = model(ids, cu, pos)
    for layer in model.layers:  # eval forwards do not write the buffer
        torch.testing.assert_close(layer.ffn.load_ema, torch.full((E,), 1 / E, dtype=torch.float64))
    model.train()
    out = model(ids, cu, pos)
    for l, layer in enumerate(model.layers):
        exp = 0.9 / E + 0.1 * out.load[l]
        torch.testing.assert_close(layer.ffn.load_ema, exp)
    assert out.lb_loss.requires_grad


def test_balance_bias_selection_only():
    moe = base_moe(balance="bias")
    n = torch.randn(16, 576)
    logits, top_i, w = moe.route(n)
    with torch.no_grad():
        moe.balance_bias[:] = torch.linspace(-5, 5, 28)
    logits2, top_i2, w2 = moe.route(n)
    torch.testing.assert_close(logits, logits2)  # c does not enter the logits / z-loss
    torch.testing.assert_close(w2, 8.0 * torch.softmax(logits.gather(1, top_i2), -1))
    f = torch.zeros(28)
    f[0] = 1.0
    before = moe.balance_bias.clone()
    moe.update_balance_bias(f, 0.001)
    delta = moe.balance_bias - before
    assert delta[0].item() == pytest.approx(-0.001, abs=1e-6) and delta[1].item() == pytest.approx(0.001, abs=1e-6)
