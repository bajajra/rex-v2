"""T23: one forward and backward at the stage-3 max_len completes within memory on the target
card, and the flash backend accepts that length (GPU only).

Set REX_T23_MAX_LEN to cap the length (e.g. 131072 on a 24 GB card, §6.4) and REX_T23_TOK_CHUNK
for the nested token-chunk checkpoint. The CPU test below runs the same code path (grad_ckpt
layer + tok_chunk + chunked head, one sequence as its own micro-batch) on the tiny model at a
short length; it is a mechanics check, not T23.
"""

import os

import pytest
import torch

from conftest import CONFIGS, make_tiny
from rex.config import load_config
from rex.model import RexModel, next_token_targets, packed_positions
from rex.model.attn_backends import flash_available
from rex.train.losses import chunked_lm_backward


def _fwd_bwd(model, cfg, N, device):
    cu = torch.tensor([0, N], dtype=torch.int32, device=device)
    pos = packed_positions(cu)
    ids = torch.randint(0, cfg.model.unpadded_vocab_size, (N,), device=device)
    ids[0] = cfg.model.bos_id
    tgt = next_token_targets(ids, cu)
    dt = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.autocast(device_type=device.type, dtype=dt, enabled=device.type == "cuda"):
        out = model(ids, cu, pos)
        st = chunked_lm_backward(model, out.hidden, tgt, chunk_tokens=cfg.loss.loss_chunk_tokens,
                                 ce_scale=1.0 / (N - 1), aux_loss=out.aux_loss)
    return st


@pytest.mark.gpu
@pytest.mark.skipif(not flash_available(), reason="T23 needs CUDA and flash-attn")
def test_t23_long_sequence_gpu():  # pragma: no cover - not runnable on CPU
    cfg = load_config(CONFIGS / "rex-v2-base.yaml")
    N = int(os.environ.get("REX_T23_MAX_LEN", cfg.train.curriculum[-1].max_len))
    tc = os.environ.get("REX_T23_TOK_CHUNK")
    cfg = cfg.replace(train={"grad_ckpt": "layer", "tok_chunk": int(tc) if tc else None})
    dev = torch.device("cuda")
    model = RexModel(cfg).to(dev).train()
    torch.cuda.reset_peak_memory_stats(dev)
    st = _fwd_bwd(model, cfg, N, dev)
    peak = torch.cuda.max_memory_allocated(dev)
    print(f"T23: N={N} peak={peak / 2**30:.2f} GiB ({peak / N / 1024:.1f} KiB/token)")
    assert st.n_targets == N - 1
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_long_sequence_path_cpu():
    model = make_tiny(dtype=torch.float32, train={"grad_ckpt": "layer", "tok_chunk": 96}).train()
    cfg = model.cfg
    st = _fwd_bwd(model, cfg, 640, torch.device("cpu"))    # 80x the tiny window
    assert st.n_targets == 639
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in model.named_parameters()
               if not (n.endswith(".u") or n.endswith(".P.weight")))
