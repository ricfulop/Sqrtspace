from __future__ import annotations

from typing import Dict


def plan_defaults(cfg: Dict) -> Dict:
    # placeholder: return cfg with sane defaults
    out = dict(cfg)
    out.setdefault("checkpoint_every", 16)
    out.setdefault("lru_bytes", 64 * 1024 * 1024)
    out.setdefault("max_workers", 8)
    out.setdefault("input_buffer_blocks", 64)
    return out
