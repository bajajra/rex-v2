"""Training loop (§9): micro-batch accumulation, mixed precision, chunked head, auxiliary losses,
gradient clipping, LR schedule and curriculum, MoE balance-bias updates, logging, evaluation,
checkpoint save/resume.

Loss normalisation over a step of several micro-batches:
  * cross-entropy: summed over all target positions of the step and divided by the step's total
    target count, so a step equals one large batch (L_CE = mean over all positions, §5.8);
  * auxiliary losses (computed per micro-batch by the model, §5.7): weighted by the micro-batch's
    share of the step's tokens.
"""

from __future__ import annotations

import json
import math
import os
import random
import resource
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from ..eval.metrics import (DeadExpertTracker, RoutingAccumulator, length_histogram, parameter_stats,
                            snapshot_groups, update_ratios)
from ..model.model import next_token_targets
from .losses import HeadStats, bits_per_byte, chunked_lm_backward
from .optim import build_optimizer
from .schedule import lr_at, total_steps

PRECISIONS = ("bf16", "fp32")


def autocast_ctx(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


class JsonlLogger:
    def __init__(self, path: Optional[Path], echo: bool = True):
        self.path, self.echo = path, echo
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, rec: dict) -> None:
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        if self.echo:
            keys = [k for k in ("kind", "step", "stage", "loss", "bpb", "lr", "grad_norm", "tok_s",
                                "val/nats", "val/loss_T1", "val/loss_T3", "val/loss_no_message",
                                "moe/load_entropy_min", "moe/dead_experts_total") if k in rec]
            print(" ".join(f"{k}={_fmt(rec[k])}" for k in keys), flush=True)


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # Linux: KB


class Trainer:
    def __init__(self, cfg, model, loader, *, out_dir: Optional[str | Path] = None, device="cpu",
                 eval_batches=None, attn_eval_batches=None, log_every: int = 100, eval_every: int = 1000,
                 ckpt_every: int = 1000, keep_ckpts: int = 2, seed: int = 0, max_steps: Optional[int] = None,
                 echo: bool = True):
        self.cfg = cfg
        self.t = cfg.train
        if self.t.precision not in PRECISIONS:
            raise ValueError(f"train.precision must be one of {PRECISIONS}")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loader = loader
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.eval_batches = eval_batches
        self.attn_eval_batches = attn_eval_batches
        self.log_every, self.eval_every, self.ckpt_every = log_every, eval_every, ckpt_every
        self.keep_ckpts = keep_ckpts
        self.n_steps = total_steps(self.t)
        self.max_steps = self.n_steps if max_steps is None else min(max_steps, self.n_steps)
        self.opt = build_optimizer(self.model, self.t)
        self.step = 0
        self.tokens = 0
        self.rng = np.random.default_rng(seed)     # pass sampling (uniform ablation)
        self.seed = seed
        m = cfg.model
        self.is_moe = m.ffn.type == "moe"
        self.routing = RoutingAccumulator(m.n_layers, m.ffn.n_experts, m.ffn.route_scale)
        self.dead = DeadExpertTracker(m.n_layers, m.ffn.n_experts, self.t.warmup_steps)
        self.logger = JsonlLogger(self.out_dir / "metrics.jsonl" if self.out_dir else None, echo)
        self._win = self._new_window()
        if cfg.loss.balance == "bias" and cfg.loss.lb_coef != 1e-4:
            warnings.warn("§5.7: balance: bias sets lb_coef to 1e-4; config has "
                          f"lb_coef={cfg.loss.lb_coef} (not overridden)")
        self.hooks: List[Callable[[dict], None]] = []

    # ------------------------------------------------------------------ window stats
    def _new_window(self) -> dict:
        return {"head": HeadStats(), "bytes": 0.0, "tokens": 0, "steps": 0, "clipped": 0, "gn_sum": 0.0,
                "t0": time.time(), "lens": {}, "aux": 0.0, "n_mb": 0}

    # ------------------------------------------------------------------ one step
    def train_step(self) -> dict:
        model, t = self.model, self.t
        model.train()
        sb = self.loader.next_step(self.tokens)
        lr = lr_at(self.step, t, self.n_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        head = HeadStats()
        f_step = None
        w_tok = 0
        aux_total = 0.0
        for mb in sb.micro:
            mb = mb.to(self.device)
            T = None
            if t.pass_sampling == "uniform":
                T = int(self.rng.integers(1, model.n_passes + 1))   # T ~ U{1..n_passes} per micro-batch
            with autocast_ctx(self.device, t.precision):
                out = model(mb.ids, mb.cu, mb.pos, T)
                tgt = next_token_targets(mb.ids, mb.cu)
                st = chunked_lm_backward(model, out.hidden, tgt, chunk_tokens=self.cfg.loss.loss_chunk_tokens,
                                         ce_scale=1.0 / max(sb.n_targets, 1), aux_loss=out.aux_loss,
                                         aux_scale=mb.n_tokens / sb.n_tokens)
            head.add(st)
            aux_total += float(out.aux_loss.detach()) * mb.n_tokens / sb.n_tokens
            self._win["bytes"] += mb.bytes_est
            self._win["lens"].setdefault(sb.stage, []).extend(mb.seq_lens)
            if out.load is not None:
                self.routing.add(out)
                fl = out.load.detach().double() * mb.n_tokens
                f_step = fl if f_step is None else f_step + fl
                w_tok += mb.n_tokens
            del out
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip))
        log_now = (self.step + 1) % self.log_every == 0
        before = snapshot_groups(self.opt) if log_now else None
        self.opt.step()
        if self.cfg.loss.balance == "bias" and f_step is not None:
            model.update_balance_biases((f_step / w_tok).to(self.device))
        ratios = update_ratios(self.opt, before) if log_now else {}
        self.step += 1
        self.tokens += sb.n_tokens
        w = self._win
        w["head"].add(head)
        w["tokens"] += sb.n_tokens
        w["steps"] += 1
        w["clipped"] += int(gn > t.grad_clip)
        w["gn_sum"] += gn
        w["aux"] += aux_total
        rec = {"step": self.step, "loss": head.ce_sum / max(head.n_targets, 1), "lr": lr, "grad_norm": gn,
               "stage": sb.stage, "max_len": sb.max_len, "tokens": self.tokens, "n_micro": len(sb.micro)}
        if log_now:
            self._log_window(rec, ratios)
        return rec

    def _log_window(self, rec: dict, ratios: dict) -> None:
        w = self._win
        dt = max(time.time() - w["t0"], 1e-9)
        nats = w["head"].ce_sum / max(w["head"].n_targets, 1)
        out = {"kind": "train", "step": self.step, "stage": rec["stage"], "max_len": rec["max_len"],
               "tokens": self.tokens, "loss": nats, "bpb": bits_per_byte(w["head"].ce_sum, w["bytes"]),
               "aux_loss": w["aux"] / max(w["steps"], 1), "lr": rec["lr"],
               "grad_norm": w["gn_sum"] / max(w["steps"], 1), "clip_frac": w["clipped"] / max(w["steps"], 1),
               "softcap_saturation": w["head"].n_saturated / max(w["head"].n_logits, 1),
               "tok_s": w["tokens"] / dt, "peak_mem_bytes": peak_memory_bytes(self.device),
               "seq_len_hist": {f"stage{s}": length_histogram(v) for s, v in w["lens"].items()}}
        out.update(ratios)
        out.update(parameter_stats(self.model))
        if self.is_moe:
            out.update(self.routing.summary())
            stale = self.dead.update(self.step, self.routing.dead_mask())
            if stale:
                out["moe/dead_alarm"] = stale  # §5.7: restart with lb_coef 0.03, then balance: bias
            self.routing.reset()
        self.logger.log(out)
        for h in self.hooks:
            h(out)
        self._win = self._new_window()

    # ------------------------------------------------------------------ eval
    def run_eval(self) -> Optional[dict]:
        if not self.eval_batches:
            return None
        from ..eval.probes import evaluate  # lazy: rex.eval.probes imports rex.train.losses

        res = evaluate(self.model, self.eval_batches, chunk_tokens=self.cfg.loss.loss_chunk_tokens,
                       device=self.device, precision=self.t.precision, attn_batches=self.attn_eval_batches)
        rec = {"kind": "eval", "step": self.step, "tokens": self.tokens, **res}
        self.logger.log(rec)
        for h in self.hooks:
            h(rec)
        return rec

    # ------------------------------------------------------------------ loop
    def fit(self) -> None:
        while self.step < self.max_steps:
            self.train_step()
            if self.eval_every and self.step % self.eval_every == 0:
                self.run_eval()
            if self.out_dir is not None and self.ckpt_every and self.step % self.ckpt_every == 0:
                self.save()
        if self.out_dir is not None and (not self.ckpt_every or self.step % self.ckpt_every != 0):
            self.save()

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(), "step": self.step,
                "tokens": self.tokens, "loader": self.loader.state_dict(),
                "rng": {"np": self.rng.bit_generator.state, "torch": torch.get_rng_state(),
                        "python": random.getstate()},
                "dead": self.dead.state_dict(), "config": self.cfg.to_dict(), "n_steps": self.n_steps}

    def load_state_dict(self, s: dict) -> None:
        if s["config"]["model"] != self.cfg.to_dict()["model"]:
            raise ValueError("checkpoint model config differs from the current config")
        if s["config"] != self.cfg.to_dict():
            warnings.warn("checkpoint config differs from the current config outside the model section")
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["optimizer"])
        self.step, self.tokens = int(s["step"]), int(s["tokens"])
        self.loader.load_state_dict(s["loader"])
        self.rng.bit_generator.state = s["rng"]["np"]
        torch.set_rng_state(s["rng"]["torch"])
        random.setstate(s["rng"]["python"])
        self.dead.load_state_dict(s["dead"])

    def save(self, path: Optional[Path] = None) -> Path:
        assert self.out_dir is not None or path is not None
        d = self.out_dir / "checkpoints" if path is None else Path(path).parent
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"step_{self.step:07d}.pt" if path is None else Path(path)
        tmp = path.with_suffix(".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)
        (d / "latest").write_text(path.name)
        if self.keep_ckpts:
            old = sorted(d.glob("step_*.pt"))[:-self.keep_ckpts]
            for p in old:
                p.unlink()
        self.logger.log({"kind": "checkpoint", "step": self.step, "path": str(path)})
        return path

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))


def latest_checkpoint(out_dir: str | Path) -> Optional[Path]:
    d = Path(out_dir) / "checkpoints"
    f = d / "latest"
    if f.is_file() and (d / f.read_text().strip()).is_file():
        return d / f.read_text().strip()
    return None
