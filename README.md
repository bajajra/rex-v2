# rex-v2

A looped mixture-of-experts decoder with attention residuals, under 200M non-embedding
parameters. `docs/DESIGN.md` is the specification; this repository implements it.

Status: model, configs and model tests (stage 1). Tokenizer, data pipeline, training loop
and evaluation scripts are not implemented yet.

## Install

```bash
# CPU build of PyTorch (use the CUDA index URL of your choice on a GPU machine)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
# optional, CUDA only: the production attention backend
# pip install flash-attn --no-build-isolation
```

Without `flash-attn` (or on CPU), `attn.backend: flash` falls back to the pure-PyTorch
reference kernel and prints a warning.

## Tests

```bash
pytest                 # everything (the slow tests need about 1.5 GB RAM and a few seconds each)
pytest -m "not slow"   # skip tests that instantiate the full-size models
```

Tests marked `gpu` (T11 kernel equivalence, T16 peak memory) skip unless CUDA and flash-attn
are available.

## Parameter counts

```bash
python scripts/count_params.py configs/rex-v2-base.yaml configs/a1.yaml configs/b0.yaml \
    configs/b1.yaml configs/b2.yaml configs/b3.yaml
```

This prints the §3 breakdown from the formulas next to the counts read from the instantiated
model (built on the meta device, so no memory is allocated).

## Using the model

```python
import torch
from rex.config import load_config
from rex.model import RexModel, packed_positions

cfg = load_config("configs/rex-v2-base.yaml")
model = RexModel(cfg)
cu = torch.tensor([0, 100, 250], dtype=torch.int32)   # two packed sequences
pos = packed_positions(cu)
ids = torch.randint(0, cfg.model.unpadded_vocab_size, (250,))
out = model(ids, cu, pos)          # RexOutput: hidden, aux_loss, lb_loss, z_loss, load, routing
logits = model.logits(out.hidden)  # [N, unpadded_vocab_size], softcapped, float32
```
