"""Parameter groups, AdamW and Muon (§9).

Weight decay 0.1: all matrices — ``Emb`` (and an untied output head), attention W_Q/W_K/W_V/W_O,
experts W_in/W_out, router weight, output-gate weight (and the dense-FFN matrices of the
baselines). Weight decay 0: RMSNorm gains, biases, QK scales, depth queries, cross-pass u, P, F, G.

Muon (ablation, ``optimizer: muon``): attention W_Q/W_K/W_V/W_O and expert W_in/W_out only, per
expert slice; Nesterov momentum 0.95; five Newton-Schulz steps with (3.4445, -4.7750, 2.0315);
update = 0.2·sqrt(max(rows, cols))·NS(·), with the AdamW learning rate and weight decay 0.1.
Everything else stays on AdamW.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# (regex on the parameter name, group). Names follow rex/model; unknown names raise.
_RULES: List[Tuple[str, str]] = [
    (r"^emb\.weight$", "decay"),
    (r"^lm_head\.weight$", "decay"),
    (r"^layers\.\d+\.attn\.w[qkvo]\.weight$", "decay_matrix"),       # Muon-eligible
    (r"^layers\.\d+\.attn\.gate\.weight$", "decay"),
    (r"^layers\.\d+\.attn\.gate\.bias$", "no_decay"),
    (r"^layers\.\d+\.attn\.qk_scale$", "no_decay"),
    (r"^layers\.\d+\.ffn\.w_(in|out)$", "decay_matrix"),             # MoE experts, Muon-eligible
    # SPEC: §9 Muon names "attention and expert matrices only"; dense-FFN matrices (baselines)
    # are not named, so they stay on AdamW (with decay, as matrices).
    (r"^layers\.\d+\.ffn\.w_(in|out)\.weight$", "decay"),            # dense FFN (baselines)
    (r"^layers\.\d+\.ffn\.router_weight$", "decay"),
    (r"^layers\.\d+\.ffn\.router_bias$", "no_decay"),
    (r"^(emb_norm|final_norm|layers\.\d+\.(attn_norm|ffn_norm))\.weight$", "no_decay"),
    (r"^(mix_out|layers\.\d+\.mix_(attn|q|k|v|moe))\.q$", "no_decay"),
    (r"^layers\.\d+\.cross\.[QKVM]\.(u|P\.weight|F\.weight|G\.weight)$", "no_decay"),
]


def classify(name: str) -> str:
    for pat, g in _RULES:
        if re.match(pat, name):
            return g
    raise KeyError(f"parameter {name!r} matches no parameter-group rule (§9)")


def param_groups(model: nn.Module, weight_decay: float, use_muon: bool = False) -> List[dict]:
    """Every parameter (frozen ones included) lands in exactly one group:
    ``decay``, ``no_decay`` and, with Muon, ``muon``."""
    groups: Dict[str, dict] = {
        "decay": {"name": "decay", "params": [], "names": [], "weight_decay": weight_decay},
        "no_decay": {"name": "no_decay", "params": [], "names": [], "weight_decay": 0.0},
    }
    if use_muon:
        groups["muon"] = {"name": "muon", "params": [], "names": [], "weight_decay": weight_decay}
    for n, p in model.named_parameters():
        g = classify(n)
        if g == "decay_matrix":
            g = "muon" if use_muon else "decay"
        groups[g]["params"].append(p)
        groups[g]["names"].append(n)
    return [g for g in groups.values() if g["params"]]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalise the last two dims (batched over leading dims, i.e. per expert slice)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float = 0.1, momentum: float = 0.95, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            mu = g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.get("momentum_buffer")
                if buf is None:
                    buf = st["momentum_buffer"] = torch.zeros_like(p)
                buf.mul_(mu).add_(p.grad)
                upd = p.grad.add(buf, alpha=mu)                         # Nesterov
                o = zeropower_via_newtonschulz5(upd, g["ns_steps"])
                rows, cols = p.shape[-2], p.shape[-1]
                o = o * (0.2 * math.sqrt(max(rows, cols)))
                p.mul_(1.0 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"])


class CombinedOptimizer:
    """Several optimisers behind one interface (step / zero_grad / state_dict / param_groups)."""

    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self) -> List[dict]:
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self) -> None:
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, s: dict) -> None:
        for o, sd in zip(self.optimizers, s["optimizers"], strict=True):
            o.load_state_dict(sd)


def build_optimizer(model: nn.Module, train_cfg) -> CombinedOptimizer:
    t = train_cfg
    if t.optimizer not in ("adamw", "muon"):
        raise ValueError(f"optimizer must be adamw | muon, got {t.optimizer!r}")
    groups = param_groups(model, t.weight_decay, use_muon=(t.optimizer == "muon"))
    adam_groups = [g for g in groups if g["name"] != "muon"]
    muon_groups = [g for g in groups if g["name"] == "muon"]
    opts: List[torch.optim.Optimizer] = [
        torch.optim.AdamW(adam_groups, lr=t.lr, betas=tuple(t.betas), eps=t.adam_eps, weight_decay=t.weight_decay)
    ]
    if muon_groups:
        opts.append(Muon(muon_groups, lr=t.lr, weight_decay=t.weight_decay, momentum=0.95))
    return CombinedOptimizer(opts)
