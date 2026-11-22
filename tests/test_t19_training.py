"""T19 gradient reach; T20 overfit one batch.

Both use plain torch AdamW (the recipe's optimiser module is stage 2). §14 does not give the
learning rate for these tiny-model tests; the values below are test choices.
"""

import pytest
import torch

from conftest import make_tiny, packed
from rex.model import lm_loss


def _step_loss(model, batch):
    ids, cu, pos = batch
    ce, out = lm_loss(model, ids, cu, pos)
    return ce, ce + out.aux_loss


def test_t19_gradient_reach():
    model = make_tiny(dtype=torch.float32).train()
    U = model.mcfg.unpadded_vocab_size
    batch = packed([64, 64], vocab=U, seed=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for step in range(3):
        opt.zero_grad()
        _, loss = _step_loss(model, batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            if name == "emb.weight":
                assert torch.all(p.grad[U:] == 0)
                assert torch.all(p.grad[:U].abs().sum(-1) > 0)
                continue
            nonzero = p.grad.abs().sum() > 0
            if step == 0 and (name.endswith(".u") or name.endswith(".P.weight")):
                continue  # §14: at step 0 only P and u may be zero
            assert nonzero, f"step {step}: zero gradient for {name}"
        if step < 2:
            opt.step()


def _overfit(model, batch, steps=500, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    ce = None
    for _ in range(steps):
        opt.zero_grad()
        ce, loss = _step_loss(model, batch)
        loss.backward()
        opt.step()
        if ce.item() < 0.05:
            break
    return ce.item()


def test_t20_overfit_one_batch():
    torch.manual_seed(0)
    model = make_tiny(dtype=torch.float32).train()
    batch = packed([64, 64, 64, 64], vocab=model.mcfg.unpadded_vocab_size, seed=5)
    # SPEC: §14 says "loss < 0.05 with auxiliary losses on". The aux term alone is about
    # 0.01·Σ_8 L_LB ≈ 0.08, so the threshold is applied to the cross-entropy while the
    # optimised objective includes the auxiliary losses.
    assert _overfit(model, batch) < 0.05
