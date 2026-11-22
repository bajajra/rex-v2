"""Chunked output head and loss (§5.8), auxiliary-loss weighting.

§5.8: "detach hfinal into a leaf; for each chunk of loss_chunk_tokens compute logits and the
summed loss, call backward to accumulate gradients into Emb and into the leaf; then
backpropagate the leaf's gradient into the body. This is exact."

The auxiliary losses (L_LB, L_Z) come from the model's forward (``RexOutput.aux_loss``,
already multiplied by ``lb_coef`` / ``z_coef``); they are backpropagated together with the
leaf's gradient in one ``torch.autograd.backward`` call so the body graph is traversed once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

IGNORE = -100


@dataclass
class HeadStats:
    ce_sum: float = 0.0          # Σ nats over targets
    n_targets: int = 0
    n_saturated: int = 0         # logits with |x| > saturation threshold (softcap saturation, §10)
    n_logits: int = 0

    def add(self, o: "HeadStats") -> None:
        self.ce_sum += o.ce_sum
        self.n_targets += o.n_targets
        self.n_saturated += o.n_saturated
        self.n_logits += o.n_logits


def chunked_lm_backward(model, hidden: torch.Tensor, targets: torch.Tensor, *, chunk_tokens: int,
                        ce_scale: float, aux_loss: Optional[torch.Tensor] = None, aux_scale: float = 1.0,
                        saturation: float = 25.0) -> HeadStats:
    """Backpropagate ``ce_scale * Σ CE + aux_scale * aux_loss`` without materialising all logits.

    ``hidden`` is the body's output (requires grad). ``targets`` [N] uses ``IGNORE`` for positions
    without a next token in the same sequence. Returns detached statistics.
    """
    st = HeadStats()
    leaf = hidden.detach().requires_grad_(True)
    N = hidden.shape[0]
    for a in range(0, N, chunk_tokens):
        b = min(a + chunk_tokens, N)
        t = targets[a:b]
        nt = int((t != IGNORE).sum())
        if nt == 0:
            continue
        logits = model.logits(leaf[a:b])
        ce = F.cross_entropy(logits, t, ignore_index=IGNORE, reduction="sum")
        (ce * ce_scale).backward()
        with torch.no_grad():
            st.ce_sum += float(ce.detach())
            st.n_targets += nt
            st.n_saturated += int((logits.detach().abs() > saturation).sum())
            st.n_logits += logits.numel()
        del logits, ce
    tensors, grads = [], []
    if leaf.grad is not None and hidden.requires_grad:
        tensors.append(hidden)
        grads.append(leaf.grad)
    if aux_loss is not None and aux_loss.requires_grad:
        tensors.append(aux_loss * aux_scale)
        grads.append(torch.ones_like(aux_loss))
    if tensors:
        torch.autograd.backward(tensors, grads)
    return st


@torch.no_grad()
def chunked_lm_eval(model, hidden: torch.Tensor, targets: torch.Tensor, *, chunk_tokens: int,
                    per_token: bool = False, saturation: float = 25.0):
    """Forward-only chunked head. Returns (HeadStats, per-token nll [N] float32 or None);
    ignored positions get nll 0 in the per-token vector."""
    st = HeadStats()
    nll = torch.zeros(hidden.shape[0], dtype=torch.float32, device=hidden.device) if per_token else None
    for a in range(0, hidden.shape[0], chunk_tokens):
        b = min(a + chunk_tokens, hidden.shape[0])
        t = targets[a:b]
        mask = t != IGNORE
        if not bool(mask.any()):
            continue
        logits = model.logits(hidden[a:b])
        l = F.cross_entropy(logits, t.clamp_min(0), reduction="none") * mask
        st.ce_sum += float(l.sum())
        st.n_targets += int(mask.sum())
        st.n_saturated += int((logits.abs() > saturation).sum())
        st.n_logits += logits.numel()
        if per_token:
            nll[a:b] = l.float()
    return st, nll


def bits_per_byte(nats: float, n_bytes: float) -> float:
    return float("nan") if n_bytes <= 0 else nats / (math.log(2) * n_bytes)
