from __future__ import annotations

import os
from typing import Dict, Any


def detect_llc_bytes() -> int:
    # Minimal, conservative fallback with env override
    env = os.environ.get("LLC_BYTES")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    # Default 32 MiB when unknown
    return 32 << 20


def dependency_span(window: int, hop: int) -> int:
    if hop <= 0:
        return 1
    # Number of hops covering one window (ceil(window / hop))
    return max(1, (window + hop - 1) // hop)


def plan(cfg: Dict[str, Any], windows, num_stages: int) -> Dict[str, Any]:
    llc = detect_llc_bytes()
    per_win_bytes = getattr(windows, "dtype", None)
    if per_win_bytes is not None:
        per_win_bytes = windows.dtype.itemsize * (windows.shape[1] if windows.ndim >= 2 else 0)
    else:
        per_win_bytes = 4 * (windows.shape[1] if windows.ndim >= 2 else 0)

    est_live = per_win_bytes * max(1, num_stages)
    small_work = windows.shape[0] <= 32
    fits_cache = est_live < 0.6 * llc

    want_classic = small_work or fits_cache

    # Choose k and LRU
    win = int(cfg.get("window", 4096))
    hop = int(cfg.get("hop", 1024))
    span = dependency_span(win, hop)
    k_cfg = int(cfg.get("checkpoint_every", 16))
    k = max(span, k_cfg)
    lru_cfg = int(cfg.get("lru_bytes", 64 * 1024 * 1024))
    lru = min(lru_cfg, int(0.6 * llc))

    return {
        "mode": "classic" if want_classic else "sqrt",
        "checkpoint_every": k,
        "lru_bytes": lru,
        "llc_bytes": llc,
        "dependency_span": span,
    }


def plan_defaults(cfg: Dict) -> Dict:
    out = dict(cfg)
    out.setdefault("checkpoint_every", 16)
    out.setdefault("lru_bytes", 64 * 1024 * 1024)
    out.setdefault("max_workers", 8)
    out.setdefault("input_buffer_blocks", 64)
    return out
