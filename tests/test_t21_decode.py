"""T21: token-by-token generation with per-application caches reproduces the full-sequence logits."""

import pytest
import torch

from conftest import make_tiny, packed


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefill", [1, 5])
def test_t21_incremental_decoding(stream, prefill):
    extra = {"model": {"residual": "stream", "n_passes": 1}} if stream else {}
    model = make_tiny(randomize=not stream, **extra).eval()
    ids, cu, pos = packed([21])  # longer than the window (8)
    with torch.no_grad():
        full = model.logits(model(ids, cu, pos).hidden)
        cache = model.new_cache()
        outs = []
        i = 0
        while i < ids.numel():
            n = prefill if i == 0 else 1
            sl = slice(i, i + n)
            cu1 = torch.tensor([0, n], dtype=torch.int32)
            outs.append(model.logits(model(ids[sl], cu1, pos[sl], cache=cache).hidden))
            i += n
    inc = torch.cat(outs, 0)
    torch.testing.assert_close(inc, full, rtol=1e-5, atol=1e-10)
    # one cache per layer application: L + (T-1)·C, windowed ones trimmed to the window
    T = model.n_passes
    assert len(cache.entries) == model.L + (T - 1) * (model.mcfg.core_end - model.mcfg.core_start)
    for (l, _), e in cache.entries.items():
        att = model.layers[l].attn
        assert e.k.shape[0] == (ids.numel() if att.window is None else att.window)
