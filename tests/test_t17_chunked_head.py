"""T17: the chunked head (§5.8) gives the unchunked loss and gradients; padded rows of Emb get
zero gradient."""

import pytest
import torch

from conftest import make_tiny, packed
from rex.model import lm_loss, next_token_targets
from rex.train.losses import chunked_lm_backward, chunked_lm_eval


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("chunk", [1, 7, 64, 10_000])
@pytest.mark.parametrize("ckpt", ["none", "layer"])
def test_t17_chunked_equals_unchunked(chunk, ckpt):
    batch = packed([20, 13, 31])
    ref = make_tiny(randomize=True, loss={"lb_load_ema": 0.0})
    ch = make_tiny(randomize=True, loss={"lb_load_ema": 0.0}, train={"grad_ckpt": ckpt})
    ch.load_state_dict(ref.state_dict())
    ids, cu, pos = batch
    U = ref.mcfg.unpadded_vocab_size

    ref.zero_grad()
    ce, out = lm_loss(ref, ids, cu, pos)
    (ce + out.aux_loss).backward()
    g_ref = _grads(ref)

    ch.zero_grad()
    out2 = ch(ids, cu, pos)
    tgt = next_token_targets(ids, cu)
    n_t = int((tgt != -100).sum())
    st = chunked_lm_backward(ch, out2.hidden, tgt, chunk_tokens=chunk, ce_scale=1.0 / n_t, aux_loss=out2.aux_loss)
    g_ch = _grads(ch)

    # model.logits returns float32 (§5.8) even for the float64 test model, so the loss and the
    # head gradients carry float32 rounding; the §14 tolerance (1e-5 relative) applies.
    assert abs(st.ce_sum / st.n_targets - ce.item()) <= 1e-6 * abs(ce.item())
    assert st.n_targets == n_t
    assert g_ref.keys() == g_ch.keys()
    for n in g_ref:
        torch.testing.assert_close(g_ch[n], g_ref[n], rtol=1e-5, atol=1e-9, msg=n)
    assert torch.all(g_ch["emb.weight"][U:] == 0)


def test_t17_chunked_eval_matches():
    model = make_tiny(randomize=True).eval()
    ids, cu, pos = packed([9, 30])
    with torch.no_grad():
        ce, out = lm_loss(model, ids, cu, pos)
        st, nll = chunked_lm_eval(model, out.hidden, next_token_targets(ids, cu), chunk_tokens=5, per_token=True)
    assert abs(st.ce_sum / st.n_targets - ce.item()) < 1e-5
    assert abs(nll.sum().item() - st.ce_sum) < 1e-4


def test_t17_untied_head():
    """The chunked head also covers an untied output projection."""
    model = make_tiny(model={"tie_embeddings": False}, loss={"lb_load_ema": 0.0})
    ref = make_tiny(model={"tie_embeddings": False}, loss={"lb_load_ema": 0.0})
    ref.load_state_dict(model.state_dict())
    ids, cu, pos = packed([17, 12])
    ce, out = lm_loss(ref, ids, cu, pos)
    (ce + out.aux_loss).backward()
    out2 = model(ids, cu, pos)
    tgt = next_token_targets(ids, cu)
    chunked_lm_backward(model, out2.hidden, tgt, chunk_tokens=4, ce_scale=1.0 / int((tgt != -100).sum()),
                        aux_loss=out2.aux_loss)
    for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-9, msg=n)
    assert torch.all(model.lm_head.weight.grad[model.mcfg.unpadded_vocab_size:] == 0)
