from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class Row:
    k: int
    lru_mb: int
    time_classic_s: float
    time_sqrt_s: float
    speedup: float


def read_csv(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                Row(
                    k=int(d["k"]),
                    lru_mb=int(d["lru_mb"]),
                    time_classic_s=float(d["time_classic_s"]),
                    time_sqrt_s=float(d["time_sqrt_s"]),
                    speedup=float(d["speedup"]),
                )
            )
    return rows


def plot(rows: list[Row], outdir: str = "bench_out") -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Plot time by LRU for each k
    ks = sorted({r.k for r in rows})
    for k in ks:
        sub = [r for r in rows if r.k == k]
        sub.sort(key=lambda r: r.lru_mb)
        lru = [r.lru_mb for r in sub]
        tc = [r.time_classic_s for r in sub]
        ts = [r.time_sqrt_s for r in sub]
        sp = [r.speedup for r in sub]

        plt.figure(figsize=(6, 4))
        plt.plot(lru, tc, marker="o", label="classic")
        plt.plot(lru, ts, marker="o", label="sqrt")
        plt.xlabel("LRU (MB)")
        plt.ylabel("time (s)")
        plt.title(f"Logs benchmark times (k={k})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/times_k{k}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(lru, sp, marker="o", color="green")
        plt.xlabel("LRU (MB)")
        plt.ylabel("speedup classic/sqrt (×)")
        plt.title(f"Logs benchmark speedup (k={k})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/speedup_k{k}.png", dpi=150)
        plt.close()


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bench/plot_bench.py results.csv")
        sys.exit(1)
    rows = read_csv(sys.argv[1])
    plot(rows)

