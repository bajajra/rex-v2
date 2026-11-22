"""Parameter accounting (§3): analytic per-component formulas and the same breakdown read
from an instantiated model (instantiate on ``torch.device("meta")`` to avoid allocation)."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict

import torch.nn as nn

from ..config import RexConfig

EMBEDDING_ROWS = ("token_embedding", "lm_head")

ROWS = (
    "token_embedding", "lm_head", "embedding_norm",
    "attn_wq", "attn_wk", "attn_wv", "attn_wo", "qk_scale", "out_gate", "sublayer_norms",
    "depth_readers_attn", "depth_readers_moe", "depth_reader_out", "final_norm",
    "router_weight", "router_bias", "expert_w_in", "expert_w_out", "dense_w_in", "dense_w_out",
    "cross_u", "cross_P", "cross_F", "cross_G",
)


def analytic_counts(cfg: RexConfig) -> "OrderedDict[str, int]":
    m, a, f = cfg.model, cfg.model.attn, cfg.model.ffn
    d, V, L = m.d_model, m.vocab_size, m.n_layers
    hq, hkv, dh = a.n_q_heads, a.n_kv_heads, a.head_dim
    E, h, r = f.n_experts, f.expert_hidden, m.gate_rank
    glob = set(a.global_layers)
    attnres = m.residual == "attnres"
    C = (m.core_end - m.core_start) if (attnres and m.n_passes > 1) else 0
    moe = f.type == "moe"
    # attention readers: 3 in a full-attention layer, none for layer 0 (single source), else 1.
    # For the base config this is (L - 1 + 2G) = 31.
    n_attn_readers = sum(3 if l in glob else (0 if l == 0 else 1) for l in range(L))
    c = OrderedDict()
    c["token_embedding"] = V * d
    c["lm_head"] = 0 if m.tie_embeddings else V * d
    c["embedding_norm"] = d
    c["attn_wq"] = L * d * hq * dh
    c["attn_wk"] = L * d * hkv * dh
    c["attn_wv"] = L * d * hkv * dh
    c["attn_wo"] = L * hq * dh * d
    c["qk_scale"] = L * hq
    c["out_gate"] = L * (hq * d + hq) if a.out_gate else 0
    c["sublayer_norms"] = 2 * L * d
    c["depth_readers_attn"] = n_attn_readers * d if attnres else 0
    c["depth_readers_moe"] = L * d if attnres else 0
    c["depth_reader_out"] = d if attnres else 0
    c["final_norm"] = d
    c["router_weight"] = L * E * d if moe else 0
    c["router_bias"] = L * E if (moe and f.router_bias) else 0
    c["expert_w_in"] = L * E * 2 * h * d if moe else 0
    c["expert_w_out"] = L * E * d * h if moe else 0
    c["dense_w_in"] = 0 if moe else L * 2 * f.dense_hidden * d
    c["dense_w_out"] = 0 if moe else L * d * f.dense_hidden
    c["cross_u"] = C * 4 * d
    c["cross_P"] = C * 4 * r * 2 * d
    c["cross_F"] = C * 4 * d * r
    c["cross_G"] = C * 4 * d * r
    return c


_PATTERNS = [
    (r"^emb\.weight$", "token_embedding"),
    (r"^lm_head\.weight$", "lm_head"),
    (r"^emb_norm\.weight$", "embedding_norm"),
    (r"\.attn\.wq\.weight$", "attn_wq"),
    (r"\.attn\.wk\.weight$", "attn_wk"),
    (r"\.attn\.wv\.weight$", "attn_wv"),
    (r"\.attn\.wo\.weight$", "attn_wo"),
    (r"\.attn\.qk_scale$", "qk_scale"),
    (r"\.attn\.gate\.(weight|bias)$", "out_gate"),
    (r"\.(attn_norm|ffn_norm)\.weight$", "sublayer_norms"),
    (r"\.mix_(attn|q|k|v)\.q$", "depth_readers_attn"),
    (r"\.mix_moe\.q$", "depth_readers_moe"),
    (r"^mix_out\.q$", "depth_reader_out"),
    (r"^final_norm\.weight$", "final_norm"),
    (r"\.ffn\.router_weight$", "router_weight"),
    (r"\.ffn\.router_bias$", "router_bias"),
    (r"\.ffn\.w_in$", "expert_w_in"),
    (r"\.ffn\.w_out$", "expert_w_out"),
    (r"\.ffn\.w_in\.weight$", "dense_w_in"),
    (r"\.ffn\.w_out\.weight$", "dense_w_out"),
    (r"\.cross\.[QKVM]\.u$", "cross_u"),
    (r"\.cross\.[QKVM]\.P\.weight$", "cross_P"),
    (r"\.cross\.[QKVM]\.F\.weight$", "cross_F"),
    (r"\.cross\.[QKVM]\.G\.weight$", "cross_G"),
]


def classify(name: str) -> str:
    for pat, row in _PATTERNS:
        if re.search(pat, name):
            return row
    raise KeyError(f"parameter {name!r} is not in the §3 accounting")


def model_counts(model: nn.Module) -> "OrderedDict[str, int]":
    c = OrderedDict((k, 0) for k in ROWS)
    for name, p in model.named_parameters():
        c[classify(name)] += p.numel()
    return c


def body_total(counts: Dict[str, int]) -> int:
    return sum(v for k, v in counts.items() if k not in EMBEDDING_ROWS)


def embedding_total(counts: Dict[str, int]) -> int:
    return sum(counts[k] for k in EMBEDDING_ROWS)


def active_counts(cfg: RexConfig) -> Dict[str, int]:
    """§1/§3 active-parameter accounting.

    stored_active = body - L·(E-k)·3·d·h (inactive experts).
    effective_active (§3): weight matrices only, each counted once per layer application
    (L + (T-1)·C of them): attention matrices, output-gate weight, router weight, the k active
    experts (or the dense FFN), plus P, F, G of every gated application. All vectors (biases,
    gains, QK scales, queries) are excluded.
    """
    m, a, f = cfg.model, cfg.model.attn, cfg.model.ffn
    c = analytic_counts(cfg)
    body = body_total(c)
    d, L = m.d_model, m.n_layers
    moe = f.type == "moe"
    if moe:
        inactive = L * (f.n_experts - f.top_k) * 3 * d * f.expert_hidden
        ffn_app = f.n_experts * d + f.top_k * 3 * d * f.expert_hidden
    else:
        inactive = 0
        ffn_app = 3 * d * f.dense_hidden
    attn_app = d * (2 * a.n_q_heads + 2 * a.n_kv_heads) * a.head_dim + (a.n_q_heads * d if a.out_gate else 0)
    C = m.core_end - m.core_start
    apps = L + (m.n_passes - 1) * C
    gated_apps = (m.n_passes - 1) * C if (m.residual == "attnres" and m.n_passes > 1) else 0
    pfg = 4 * (m.gate_rank * 2 * d + 2 * d * m.gate_rank)
    return {
        "stored_active": body - inactive,
        "effective_active": apps * (attn_app + ffn_app) + gated_apps * pfg,
        "layer_applications": apps,
    }
