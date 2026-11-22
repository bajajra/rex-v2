# rex-v2

A looped mixture-of-experts decoder with attention residuals, under 200M non-embedding
parameters. `docs/DESIGN.md` is the specification; this repository implements it.

Status: model, configs and model tests (stage 1); tokenizer, data pipeline, chunked head,
training loop, §10 metrics/probes and entry-point scripts (stage 2). The production paths
(flash-attn, bf16 on GPU, T11/T16-memory/T23) have not been run: development was CPU-only.

## Install

```bash
# CPU build of PyTorch (use the CUDA index URL of your choice on a GPU machine)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test,data]"
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

## Layout (stage 2)

```
rex/data/tokenizer.py   pinned Inkling tokenizer (local file/dir or hub id; SHA-256 recorded)
rex/data/prep.py        uint32 token shards + document index (+ hub presets for FineWeb-Edu / PG-19)
rex/data/loader.py      curriculum loader: whole documents / crops, bos/eot, token-budget steps,
                        micro-batch packing into the model's packed layout (ids, cu, pos)
rex/data/standin.py     SYNTHETIC stand-in tokenizer + corpus for CPU smoke runs (not the spec's data)
rex/train/losses.py     chunked output head (§5.8), auxiliary-loss weighting
rex/train/optim.py      parameter groups (§9), AdamW, Muon (ablation)
rex/train/schedule.py   warm-up + cosine LR, curriculum stage lookup
rex/train/trainer.py    accumulation, bf16 autocast, clipping, balance-bias updates, logging, checkpoints
rex/eval/metrics.py     MoE load / dead experts / router confidence, update ratios, length histograms
rex/eval/probes.py      held-out loss (nats, bits/byte), forced T, message ablation, gate/β/pass stats,
                        α maps, block and running-sum RMS, pass overlap, attention distance,
                        position buckets, context truncation, repeat probe
scripts/                train.py, eval.py, prep_data.py, bench.py, make_standin.py, count_params.py
```

## Smoke run on CPU (synthetic stand-in data, a few minutes)

`configs/tiny-cpu.yaml` is the §14 tiny model with a scaled-down 3-stage curriculum. It trains on
a stand-in 500-id BPE tokenizer and the local Python standard-library sources: it exercises the
whole pipeline and says nothing about rex-v2 quality.

```bash
python scripts/make_standin.py --out runs/standin            # tokenizer + main/long corpora
python scripts/train.py --config configs/tiny-cpu.yaml --out runs/tiny-cpu \
    --log-every 25 --eval-every 75 --eval-batches 4 --ckpt-every 75 --max-steps 150
python scripts/train.py --config configs/tiny-cpu.yaml --out runs/tiny-cpu \
    --log-every 25 --eval-every 75 --eval-batches 4 --ckpt-every 75 --resume   # runs until total_tokens are consumed
python scripts/eval.py --ckpt runs/tiny-cpu/checkpoints/$(cat runs/tiny-cpu/checkpoints/latest) --long-max-len 2048 \
    --contexts 64 512 all --target-len 128 --gaps 128 512 --passage-len 128
```

Metrics go to `<out>/metrics.jsonl` (one JSON record per log/eval/checkpoint event).

## Real run (GPU)

1. Prepare data (needs huggingface.co access; keep outputs outside the repo):

```bash
python scripts/prep_data.py --preset fineweb-edu-10BT --out /data/fineweb   # train/ + val/ (last shard)
python scripts/prep_data.py --preset pg19 --out /data/pg19                  # train/ val/ test/
# or any local corpus: --files '/raw/*.parquet' --holdout-last-file --text-field text
# a local tokenizer: --tokenizer /path/to/tokenizer.json [--tokenizer-sha256 <pin>]
```

2. Smoke tier (§13: base model, 50M tokens (nominally 190 steps), warm-up 20, stage 1 only), then the full run:

```bash
pip install flash-attn --no-build-isolation
python scripts/train.py --config configs/smoke.yaml --out /runs/smoke \
    --set data.main=/data/fineweb --set data.long_docs=/data/pg19 --log-every 10 --eval-every 50
python scripts/train.py --config configs/rex-v2-base.yaml --out /runs/base \
    --set data.main=/data/fineweb --set data.long_docs=/data/pg19 [--resume]
```

Ablation tier (§12): `--set train.total_tokens=6e8 --set train.warmup_steps=200` plus the
stage-1-only curriculum (edit a copy of the config). Memory switches: `train.mb_tokens`,
`train.grad_ckpt` (`layer`|`none`), `train.tok_chunk` (e.g. 32768). On a 24 GB card cap the last
curriculum stage at 131,072 (§6.4). Measure throughput / peak memory first:

```bash
python scripts/bench.py --config configs/rex-v2-base.yaml --seq-len 65536 --set train.tok_chunk=32768
```

`--set` takes any config key (`section.key=value`, YAML-parsed). A sequence longer than
`mb_tokens` forms a micro-batch of its own. Tokens are the clock (§6.5, §9): each step holds at
least `tokens_per_step` tokens, the run stops at the first step boundary past `total_tokens`, and
the LR cosine and curriculum stage follow the tokens consumed before each step; the step count
`floor(total_tokens / tokens_per_step)` is nominal.
