"""Configuration dataclasses mirroring the YAML schema of docs/DESIGN.md §2.

Defaults equal the rex-v2-base values, so ``RexConfig()`` is the base config.
YAML loading is strict: unknown keys raise, so every key in a config file is consumed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
import typing
from typing import Any, List, Optional

import yaml


@dataclass
class AttnConfig:
    head_dim: int = 128
    n_q_heads: int = 6
    n_kv_heads: int = 2
    window: int = 1024
    global_layers: List[int] = field(default_factory=lambda: [5, 11, 17, 23])
    local_pos: str = "rope"
    rope_base: float = 10000.0
    global_pos: str = "none"  # none | rope
    global_rope_base: float = 1.0e6
    qk_mode: str = "norm"  # norm | l2clip
    qk_scale_init: float = 1.44
    log_scale_alpha: float = 0.1
    log_scale_n_floor: int = 8192
    out_gate: bool = True
    backend: str = "flash"  # flash | reference


@dataclass
class FFNConfig:
    type: str = "moe"  # moe | dense
    n_experts: int = 28
    top_k: int = 8
    expert_hidden: int = 144
    route_scale: float = 8.0
    router_bias: bool = True
    dense_hidden: int = 1152
    dispatch: str = "sorted"  # sorted | dense_ref


@dataclass
class ModelConfig:
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    bos_id: int = 200028
    eot_id: int = 199999
    tie_embeddings: bool = True
    d_model: int = 576
    n_layers: int = 24
    block_size: int = 6
    core_start: int = 6
    core_end: int = 18
    n_passes: int = 3
    residual: str = "attnres"  # attnres | stream
    gate_rank: int = 16
    rms_eps: float = 1.0e-6
    attn: AttnConfig = field(default_factory=AttnConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    final_logit_softcap: float = 30.0


@dataclass
class LossConfig:
    lb_coef: float = 0.01
    lb_load_ema: float = 0.9
    z_coef: float = 0.001
    balance: str = "aux"  # aux | bias
    bias_update_speed: float = 0.001
    loss_chunk_tokens: int = 2048


@dataclass
class CurriculumStage:
    until: float
    max_len: int
    long_doc_frac: float


def _default_curriculum() -> List[CurriculumStage]:
    return [
        CurriculumStage(0.82, 8192, 0.15),
        CurriculumStage(0.94, 65536, 0.50),
        CurriculumStage(1.00, 262144, 0.70),
    ]


@dataclass
class TrainConfig:
    total_tokens: float = 6.0e9
    tokens_per_step: int = 262144
    mb_tokens: int = 65536
    grad_ckpt: str = "layer"  # none | layer
    tok_chunk: Optional[int] = None
    optimizer: str = "adamw"  # adamw | muon
    lr: float = 4.0e-4
    lr_min_ratio: float = 0.1
    warmup_steps: int = 1000
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    adam_eps: float = 1.0e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "bf16"
    pass_sampling: str = "fixed"  # fixed | uniform
    curriculum: List[CurriculumStage] = field(default_factory=_default_curriculum)

    @property
    def max_len(self) -> int:
        return max(s.max_len for s in self.curriculum)


@dataclass
class DataConfig:
    main: str = "HuggingFaceFW/fineweb-edu (sample-10BT)"
    long_docs: str = "deepmind/pg19"


@dataclass
class RexConfig:
    name: str = "rex-v2-base"
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        validate(self)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "RexConfig":
        return _build(cls, d, "")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RexConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def replace(self, **sections: Any) -> "RexConfig":
        """Return a validated copy; ``sections`` are nested dicts of overrides,
        e.g. ``cfg.replace(model={"n_passes": 1, "attn": {"backend": "reference"}})``."""
        d = self.to_dict()
        _deep_update(d, sections)
        return RexConfig.from_dict(d)


def load_config(path: str | Path) -> RexConfig:
    return RexConfig.from_yaml(path)


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _build(cls, d: Any, where: str):
    if not dataclasses.is_dataclass(cls):
        return d
    if not isinstance(d, dict):
        raise TypeError(f"config section {where or '<root>'} must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise KeyError(f"unknown config keys in {where or '<root>'}: {sorted(unknown)}")
    kwargs = {}
    hints = {
        (RexConfig, "model"): ModelConfig,
        (RexConfig, "loss"): LossConfig,
        (RexConfig, "train"): TrainConfig,
        (RexConfig, "data"): DataConfig,
        (ModelConfig, "attn"): AttnConfig,
        (ModelConfig, "ffn"): FFNConfig,
    }
    for k, v in d.items():
        sub = hints.get((cls, k))
        if sub is not None:
            kwargs[k] = _build(sub, v, f"{where}.{k}".lstrip("."))
        elif cls is TrainConfig and k == "curriculum":
            kwargs[k] = [s if isinstance(s, CurriculumStage) else CurriculumStage(**s) for s in v]
        else:
            kwargs[k] = _coerce(typing.get_type_hints(cls)[k], v, f"{where}.{k}".lstrip("."))
    return cls(**kwargs)


def _coerce(hint, v, where: str):
    """YAML 1.1 (PyYAML) reads e.g. ``1.0e6`` as a string; coerce values of float fields."""
    if hint is float and isinstance(v, (int, str)) and not isinstance(v, bool):
        try:
            return float(v)
        except ValueError:
            raise TypeError(f"config key {where} must be a number, got {v!r}") from None
    if typing.get_origin(hint) in (list, List) and isinstance(v, list):
        (arg,) = typing.get_args(hint)
        return [_coerce(arg, x, where) for x in v]
    return v


def validate(cfg: RexConfig) -> None:
    m, a, f = cfg.model, cfg.model.attn, cfg.model.ffn

    def req(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"invalid config {cfg.name!r}: {msg}")

    # §5.4 config validation
    req(m.n_layers == 4 * m.block_size, "n_layers must equal 4 * block_size")
    req(m.core_start == m.block_size, "core_start must equal block_size")
    req(m.core_end == 3 * m.block_size, "core_end must equal 3 * block_size")
    req(m.n_passes >= 1, "n_passes must be >= 1")
    req(m.residual in ("attnres", "stream"), "residual must be attnres | stream")
    # §2 / §13: stream only for baseline B0, requires n_passes: 1
    req(m.residual != "stream" or m.n_passes == 1, "residual: stream requires n_passes: 1")
    req(m.unpadded_vocab_size <= m.vocab_size, "unpadded_vocab_size must be <= vocab_size")
    req(0 <= m.bos_id < m.unpadded_vocab_size and 0 <= m.eot_id < m.unpadded_vocab_size,
        "bos_id / eot_id must be valid (unpadded) ids")
    req(a.n_q_heads % a.n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads")
    req(a.head_dim % 2 == 0, "head_dim must be even (RoPE)")
    req(a.window >= 1, "window must be >= 1")
    req(all(0 <= l < m.n_layers for l in a.global_layers), "global_layers out of range")
    req(len(set(a.global_layers)) == len(a.global_layers), "global_layers has duplicates")
    req(a.local_pos == "rope", "local_pos must be rope")
    req(a.global_pos in ("none", "rope"), "global_pos must be none | rope")
    req(a.qk_mode in ("norm", "l2clip"), "qk_mode must be norm | l2clip")
    req(a.backend in ("flash", "reference"), "backend must be flash | reference")
    req(f.type in ("moe", "dense"), "ffn.type must be moe | dense")
    req(f.dispatch in ("sorted", "dense_ref"), "ffn.dispatch must be sorted | dense_ref")
    req(1 <= f.top_k <= f.n_experts, "top_k must be in [1, n_experts]")
    req(cfg.loss.balance in ("aux", "bias"), "loss.balance must be aux | bias")
    req(0.0 <= cfg.loss.lb_load_ema < 1.0, "lb_load_ema must be in [0, 1)")
    t = cfg.train
    req(t.grad_ckpt in ("none", "layer"), "grad_ckpt must be none | layer")
    req(t.tok_chunk is None or t.tok_chunk > 0, "tok_chunk must be null or positive")
    req(t.pass_sampling in ("fixed", "uniform"), "pass_sampling must be fixed | uniform")
    req(len(t.curriculum) > 0, "curriculum must be non-empty")


def tiny_config(**sections: Any) -> RexConfig:
    """The "tiny" model of §14: d 64, 8 layers in blocks of 2 (core 2..5),
    2 query / 1 KV heads x 32, 6 experts top-2 hidden 16, rank 4, window 8, vocabulary 512.

    §14 does not state the tiny model's full-attention layers, unpadded vocabulary or
    backend; following §2 ("last of each block") they are [1, 3, 5, 7]. The table has 512
    rows of which 500 are used, so padded-row behaviour is exercised. Backend: reference.
    """
    base = RexConfig(
        name="tiny",
        model=ModelConfig(
            vocab_size=512, unpadded_vocab_size=500, bos_id=498, eot_id=499,
            d_model=64, n_layers=8, block_size=2, core_start=2, core_end=6, n_passes=3,
            gate_rank=4,
            attn=AttnConfig(head_dim=32, n_q_heads=2, n_kv_heads=1, window=8,
                            global_layers=[1, 3, 5, 7], backend="reference"),
            ffn=FFNConfig(n_experts=6, top_k=2, expert_hidden=16, dense_hidden=32),
        ),
        train=TrainConfig(grad_ckpt="none"),
    )
    return base.replace(**sections) if sections else base
