import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rex.config import tiny_config  # noqa: E402
from rex.model import RexModel, packed_positions  # noqa: E402

CONFIGS = ROOT / "configs"

torch.set_num_threads(min(4, os.cpu_count() or 1))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: instantiates a full-size model on CPU (seconds to minutes, ~1.5 GB RAM)")
    config.addinivalue_line("markers", "gpu: needs CUDA and flash-attn; skipped on CPU")


def packed(lengths, vocab=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    pos = packed_positions(cu)
    ids = torch.randint(0, vocab, (int(cu[-1]),), generator=g)
    return ids, cu, pos


def make_tiny(seed=0, dtype=torch.float64, randomize=False, **sections):
    """Tiny model (§14). ``randomize`` gives depth queries and cross-pass gates random values so
    that the readers and the loop are non-trivial (default init makes them uniform/identity)."""
    torch.manual_seed(seed)
    m = RexModel(tiny_config(**sections)).to(dtype)
    if randomize:
        randomize_readers(m, seed)
    return m


@torch.no_grad()
def randomize_readers(model, seed=0):
    g = torch.Generator().manual_seed(seed + 1000)
    for dm in model.depth_mixes():
        dm.q.copy_(torch.randn(dm.q.shape, generator=g, dtype=dm.q.dtype) * 2.0)
    for gm in model.gate_mixes():
        for p in (gm.u, gm.F.weight, gm.G.weight):
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.5)
        gm.P.weight.copy_(torch.randn(gm.P.weight.shape, generator=g, dtype=gm.P.weight.dtype) * 0.3)
    for layer in model.layers:
        if layer.attn.gate is not None:
            layer.attn.gate.weight.copy_(torch.randn(layer.attn.gate.weight.shape, generator=g,
                                                     dtype=layer.attn.gate.weight.dtype) * 0.3)


@pytest.fixture
def tiny_batch():
    return packed([20, 13, 31])
