from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class RunMetrics:
    peak_rss_bytes: Optional[int] = None
    read_mb: Optional[float] = None
    write_mb: Optional[float] = None


def snapshot() -> RunMetrics:
    p = psutil.Process()
    mem = p.memory_info()
    io = p.io_counters() if hasattr(p, "io_counters") else None
    return RunMetrics(
        peak_rss_bytes=mem.rss,
        read_mb=(io.read_bytes / (1024 * 1024)) if io else None,
        write_mb=(io.write_bytes / (1024 * 1024)) if io else None,
    )
