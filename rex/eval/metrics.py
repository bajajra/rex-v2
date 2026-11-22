"""§10 training-time metrics: MoE load statistics, dead experts, router confidence, auxiliary
losses, optimisation statistics. Forward-based evaluation probes are in probes.py."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import torch


def load_entropy(counts: torch.Tensor) -> float:
    """H(f)/ln E for a count vector."""
    E = counts.numel()
    tot = float(counts.sum())
    if tot <= 0 or E < 2:
        return float("nan")
    f = counts.double() / tot
    h = -(f[f > 0] * f[f > 0].log()).sum()
    return float(h / math.log(E))


def max_min_ratio(counts: torch.Tensor) -> float:
    mn = float(counts.min())
    return float("inf") if mn <= 0 else float(counts.max()) / mn


class RoutingAccumulator:
    """Accumulates MoE statistics over a window of micro-batches (log or eval window)."""

    def __init__(self, n_layers: int, n_experts: int, route_scale: float):
        self.L, self.E, self.route_scale = n_layers, n_experts, route_scale
        self.reset()

    def reset(self) -> None:
        self.counts = torch.zeros(self.L, self.E, dtype=torch.float64)
        self.counts_pass: Dict[tuple, torch.Tensor] = defaultdict(lambda: torch.zeros(self.E, dtype=torch.float64))
        self.maxw_sum = torch.zeros(self.L, dtype=torch.float64)
        self.maxw_n = torch.zeros(self.L, dtype=torch.float64)
        self.lb = torch.zeros(self.L, dtype=torch.float64)
        self.z = torch.zeros(self.L, dtype=torch.float64)
        self.n_mb = 0

    def add(self, out) -> None:
        if out.load is None:
            return
        for r in out.routing:
            l, p = r["layer"], r["pass"]
            c = r["counts"].detach().double().cpu()
            self.counts[l] += c
            if p is not None:
                self.counts_pass[(l, p)] += c
            n = float(c.sum())  # assignments = tokens·k: token weight for the mean
            self.maxw_sum[l] += float(r["max_weight_mean"]) * n
            self.maxw_n[l] += n
        self.lb += out.lb_loss.detach().double().cpu()
        self.z += out.z_loss.detach().double().cpu()
        self.n_mb += 1

    def load(self) -> torch.Tensor:
        """f over the window [L, E] (fractions of assignments)."""
        return self.counts / self.counts.sum(-1, keepdim=True).clamp_min(1)

    def dead_mask(self) -> torch.Tensor:
        """§5.7: an expert is dead if its load over the window is below 0.1/E."""
        return self.load() < 0.1 / self.E

    def summary(self) -> dict:
        if self.n_mb == 0:
            return {}
        ent = [load_entropy(self.counts[l]) for l in range(self.L)]
        ratio = [max_min_ratio(self.counts[l]) for l in range(self.L)]
        dead = self.dead_mask().sum(-1).tolist()
        conf = (self.maxw_sum / self.maxw_n.clamp_min(1) / self.route_scale).tolist()
        out = {
            "moe/load_entropy": ent,
            "moe/load_entropy_min": min(ent),
            "moe/max_min_ratio": ratio,
            "moe/max_min_ratio_max": max(ratio),
            "moe/dead_experts": dead,
            "moe/dead_experts_total": int(sum(dead)),
            "moe/router_confidence": conf,
            "moe/router_confidence_mean": sum(conf) / len(conf),
            "moe/lb_loss": (self.lb / self.n_mb).tolist(),
            "moe/z_loss": (self.z / self.n_mb).tolist(),
        }
        if self.counts_pass:
            out["moe/load_entropy_per_pass"] = {f"L{l}.p{p}": load_entropy(c)
                                                for (l, p), c in sorted(self.counts_pass.items())}
        return out


class DeadExpertTracker:
    """§5.7: flags experts that stay dead for ``patience`` steps after warm-up (operator action:
    restart with lb_coef 0.03, then balance: bias; no revival mechanism is built)."""

    def __init__(self, n_layers: int, n_experts: int, warmup_steps: int, patience: int = 2000):
        self.warmup, self.patience = warmup_steps, patience
        self.last_alive = torch.zeros(n_layers, n_experts, dtype=torch.long)

    def update(self, step: int, dead: torch.Tensor) -> List[tuple]:
        self.last_alive[~dead] = step
        if step < self.warmup:
            self.last_alive[:] = step
            return []
        stale = (step - torch.clamp(self.last_alive, min=self.warmup)) >= self.patience
        return [tuple(x) for x in stale.nonzero().tolist()]

    def state_dict(self) -> dict:
        return {"last_alive": self.last_alive.clone()}

    def load_state_dict(self, s: dict) -> None:
        self.last_alive = s["last_alive"].clone()


@torch.no_grad()
def parameter_stats(model) -> dict:
    """QK scales per layer (§10 "QK scales"). Output-gate means need activations: see probes."""
    return {"attn/qk_scale_mean": [float(l.attn.qk_scale.float().mean()) for l in model.layers]}


@torch.no_grad()
def snapshot_groups(optimizer) -> List[List[torch.Tensor]]:
    return [[p.detach().clone() for p in g["params"]] for g in optimizer.param_groups]


@torch.no_grad()
def update_ratios(optimizer, before: List[List[torch.Tensor]]) -> Dict[str, float]:
    """‖Δθ‖ / ‖θ‖ per parameter group (§10 "update-to-weight ratio per parameter group")."""
    out = {}
    for g, ps in zip(optimizer.param_groups, before):
        num = sum(float((p.detach().float() - b.float()).pow(2).sum()) for p, b in zip(g["params"], ps))
        den = sum(float(b.float().pow(2).sum()) for b in ps)
        out[f"opt/update_ratio/{g.get('name', '?')}"] = math.sqrt(num) / max(math.sqrt(den), 1e-30)
    return out


def length_histogram(lengths: Iterable[int]) -> Dict[str, int]:
    """Sequence-length histogram with power-of-two bins (§10)."""
    h: Dict[str, int] = defaultdict(int)
    for L in lengths:
        b = 1 << max(0, (int(L) - 1).bit_length())
        h[f"<={b}"] += 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0][2:])))
