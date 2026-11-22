"""Training loop: schedule, micro-batch accumulation, loss decrease, exact checkpoint resume,
balance-bias updates, pass sampling, bf16 autocast on CPU, metric emission.

Runs the tiny model on SYNTHETIC learnable documents (arithmetic progressions of token ids),
written in the prepared-shard format.
"""

import json
import math

import numpy as np
import pytest
import torch

from rex.config import CurriculumStage, tiny_config
from rex.data.loader import Corpus, CurriculumLoader, fixed_batches
from rex.data.prep import ShardWriter
from rex.model import RexModel
from rex.train.run import parse_overrides
from rex.train.schedule import lr_at, stage_index, total_steps
from rex.train.trainer import Trainer, latest_checkpoint

BOS, EOT = 498, 499


def _synthetic(path, n_docs, lo, hi, seed):
    rng = np.random.default_rng(seed)
    w = ShardWriter(path, shard_tokens=1 << 16)
    for _ in range(n_docs):
        L = int(rng.integers(lo, hi))
        a, s = int(rng.integers(0, 400)), int(rng.choice([1, 2, 3, 5]))
        w.add([(a + s * j) % 400 + 10 for j in range(L)], n_bytes=3 * L)
    w.close({"tokenizer_path": "SYNTHETIC", "tokenizer_sha256": "SYNTHETIC", "sources": ["SYNTHETIC"]})
    return Corpus(path)


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    return (_synthetic(d / "main", 300, 5, 120, 0), _synthetic(d / "long", 30, 200, 900, 1),
            _synthetic(d / "val", 40, 5, 120, 2))


def _cfg(**train):
    t = {"total_tokens": 1024 * 40, "tokens_per_step": 1024, "mb_tokens": 512, "lr": 3e-3, "warmup_steps": 5,
         "precision": "fp32", "grad_ckpt": "layer",
         "curriculum": [{"until": 0.5, "max_len": 64, "long_doc_frac": 0.15},
                        {"until": 0.8, "max_len": 256, "long_doc_frac": 0.5},
                        {"until": 1.0, "max_len": 512, "long_doc_frac": 0.7}]}
    t.update(train)
    return tiny_config(train=t)


def _trainer(cfg, corpora, out_dir=None, **kw):
    main, long, val = corpora
    torch.manual_seed(0)
    model = RexModel(cfg)
    t, m = cfg.train, cfg.model
    ld = CurriculumLoader(main, long, curriculum=t.curriculum, total_tokens=t.total_tokens,
                          tokens_per_step=t.tokens_per_step, mb_tokens=t.mb_tokens, bos_id=m.bos_id,
                          eot_id=m.eot_id, seed=0)
    ev = fixed_batches(val, None, max_len=64, n_batches=2, mb_tokens=256, long_doc_frac=0.0, bos_id=BOS, eot_id=EOT)
    return Trainer(cfg, model, ld, out_dir=out_dir, eval_batches=ev, echo=False, **kw)


def test_schedule():
    t = _cfg(warmup_steps=10, total_tokens=1024 * 100).train
    n = total_steps(t)
    assert n == 100
    assert lr_at(0, t) == pytest.approx(t.lr / 10)
    assert lr_at(9, t) == pytest.approx(t.lr)
    assert lr_at(n - 1, t) == pytest.approx(t.lr * t.lr_min_ratio)
    vals = [lr_at(s, t) for s in range(10, n)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    from rex.config import RexConfig
    assert total_steps(RexConfig().train) == 22888                  # §9
    st = [CurriculumStage(0.82, 8192, 0.15), CurriculumStage(0.94, 65536, 0.5), CurriculumStage(1.0, 262144, 0.7)]
    assert [stage_index(st, 100, x) for x in (0, 81.9, 82, 93.9, 94, 100)] == [0, 0, 1, 1, 2, 2]


def test_smoke_config_matches_ruling():
    from conftest import CONFIGS
    from rex.config import load_config
    c = load_config(CONFIGS / "smoke.yaml")
    assert total_steps(c.train) == 190 and c.train.warmup_steps == 20
    assert [(s.until, s.max_len) for s in c.train.curriculum] == [(1.0, 8192)]


def test_parse_overrides():
    assert parse_overrides(["train.lr=2e-4", "model.attn.backend=reference", "train.tok_chunk=null"]) == {
        "train": {"lr": "2e-4", "tok_chunk": None}, "model": {"attn": {"backend": "reference"}}}
    c = tiny_config().replace(**parse_overrides(["train.lr=2e-4"]))
    assert c.train.lr == 2e-4


def test_loss_falls_and_metrics_emit(corpora, tmp_path):
    tr = _trainer(_cfg(), corpora, out_dir=tmp_path, log_every=10, eval_every=20, ckpt_every=20)
    tr.fit()
    recs = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    train = [r for r in recs if r["kind"] == "train"]
    evals = [r for r in recs if r["kind"] == "eval"]
    assert len(train) == 4 and len(evals) == 2
    assert train[0]["loss"] > train[-1]["loss"] + 1.0, [r["loss"] for r in train]
    assert {r["stage"] for r in train} == {0, 1, 2}
    for k in ("bpb", "grad_norm", "clip_frac", "softcap_saturation", "tok_s", "peak_mem_bytes", "seq_len_hist",
              "opt/update_ratio/decay", "opt/update_ratio/no_decay", "attn/qk_scale_mean", "moe/load_entropy",
              "moe/max_min_ratio", "moe/dead_experts", "moe/router_confidence", "moe/lb_loss", "moe/z_loss",
              "moe/load_entropy_per_pass", "aux_loss"):
        assert k in train[-1], k
    e = evals[-1]
    for k in ("val/nats", "val/bpb", "val/loss_T1", "val/loss_T2", "val/loss_T3", "val/loss_no_message",
              "val/pos_bucket/0-1024", "loop/token_corr/p3", "loop/pass_change_cos/p2", "block_rms/V1",
              "running_rms/L0", "attn/out_gate_mean/L0", "moe/pass_overlap/L2/p1-p2", "alpha"):
        assert k in e, k
    assert all(0 <= v <= 1 for k, v in e.items() if k.startswith("moe/pass_overlap"))
    assert latest_checkpoint(tmp_path) is not None
    assert len(list((tmp_path / "checkpoints").glob("step_*.pt"))) == 2


def test_exact_resume(corpora, tmp_path):
    cfg = _cfg(total_tokens=1024 * 12)
    a = _trainer(cfg, corpora, log_every=4, eval_every=0)
    a.fit()
    b = _trainer(cfg, corpora, out_dir=tmp_path, log_every=4, eval_every=0, ckpt_every=5, max_steps=5)
    b.fit()
    assert b.step == 5
    c = _trainer(cfg, corpora, out_dir=tmp_path, log_every=4, eval_every=0, ckpt_every=5)
    c.load(latest_checkpoint(tmp_path))
    assert c.step == 5 and c.tokens == b.tokens
    c.fit()
    assert c.step == a.step == 12 and c.tokens == a.tokens
    sa, sc = a.model.state_dict(), c.model.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sc[k]), k


def test_balance_bias_and_uniform_passes(corpora):
    cfg = _cfg(total_tokens=1024 * 4, pass_sampling="uniform")
    cfg = cfg.replace(loss={"balance": "bias", "lb_coef": 1e-4})
    tr = _trainer(cfg, corpora, log_every=100, eval_every=0)
    tr.fit()
    bias = torch.stack([l.ffn.balance_bias for l in tr.model.layers])
    assert bias.abs().max() > 0
    # 4 steps at speed 1e-3: each entry moved by a multiple of 1e-3, at most 4e-3
    assert torch.allclose(bias / 1e-3, (bias / 1e-3).round(), atol=1e-4) and bias.abs().max() <= 4e-3 + 1e-9


def test_bf16_autocast_cpu_step(corpora):
    tr = _trainer(_cfg(total_tokens=1024 * 2, precision="bf16"), corpora, log_every=100, eval_every=0)
    rec = tr.train_step()
    assert math.isfinite(rec["loss"]) and math.isfinite(rec["grad_norm"])


def test_muon_and_dense_baseline_train(corpora):
    tr = _trainer(_cfg(total_tokens=1024 * 2, optimizer="muon"), corpora, log_every=100, eval_every=0)
    assert math.isfinite(tr.train_step()["loss"])
    cfg = _cfg(total_tokens=1024 * 2).replace(model={"residual": "stream", "n_passes": 1, "ffn": {"type": "dense"}})
    tr = _trainer(cfg, corpora, log_every=1, eval_every=0)
    assert math.isfinite(tr.train_step()["loss"])
    assert tr.run_eval()["val/nats"] > 0
