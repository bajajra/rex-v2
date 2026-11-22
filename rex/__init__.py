"""rex-v2: a looped mixture-of-experts decoder with attention residuals (see docs/DESIGN.md)."""

from .config import RexConfig, load_config, tiny_config

__all__ = ["RexConfig", "load_config", "tiny_config"]
