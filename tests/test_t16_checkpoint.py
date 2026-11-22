"""T16: gradients with grad_ckpt: layer (and with tok_chunk) equal those with none.
The peak-memory measurement part of T16 needs a GPU and is not run here."""

import pytest
import torch

from conftest import make_tiny, packed
from rex.model import lm_loss


def _grads(model, batch):
    ids, cu, pos = batch
    model.zero_grad()
    ce, out = lm_loss(model, ids, cu, pos)
    (ce + out.aux_loss).backward()
    return ce.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("train", [{"grad_ckpt": "layer"}, {"grad_ckpt": "layer", "tok_chunk": 7}])
@pytest.mark.parametrize("stream", [False, True])
def test_t16_checkpoint_grads_equal(train, stream):
    batch = packed([22, 19])
    extra = {"model": {"residual": "stream", "n_passes": 1}} if stream else {}
    ref = make_tiny(randomize=not stream, loss={"lb_load_ema": 0.0}, **extra)
    ck = make_tiny(randomize=not stream, loss={"lb_load_ema": 0.0}, train=train, **extra)
    ck.load_state_dict(ref.state_dict())
    assert ck.grad_ckpt == "layer"
    l_ref, g_ref = _grads(ref, batch)
    l_ck, g_ck = _grads(ck, batch)
    torch.testing.assert_close(l_ck, l_ref, rtol=1e-12, atol=0)
    assert g_ref.keys() == g_ck.keys()
    for n in g_ref:
        torch.testing.assert_close(g_ck[n], g_ref[n], rtol=1e-7, atol=1e-12, msg=n)


@pytest.mark.gpu
@pytest.mark.skip(reason="T16 peak-memory measurement at 8k/65k needs a GPU; not runnable on CPU")
def test_t16_peak_memory():  # pragma: no cover
    pass
