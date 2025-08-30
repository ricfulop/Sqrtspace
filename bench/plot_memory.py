from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_extreme_csv(path: str):
    modes = []
    time_s = []
    peak_mb = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            modes.append(d["mode"])  # classic, sqrt
            time_s.append(float(d["time_s"]))
            peak_mb.append(float(d["peak_rss_mb"]))
    return modes, time_s, peak_mb


def plot_memory(csv_path: str, outdir: str = "bench_out") -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    modes, time_s, peak_mb = read_extreme_csv(csv_path)

    # Bar chart for memory
    plt.figure(figsize=(5, 4))
    x = range(len(modes))
    colors = ["tab:blue" if m == "classic" else "tab:green" for m in modes]
    plt.bar(x, peak_mb, color=colors)
    plt.xticks(x, modes)
    plt.ylabel("peak RSS (MB)")
    plt.title("Extreme benchmark peak memory")
    plt.tight_layout()
    plt.savefig(f"{outdir}/peak_memory.png", dpi=150)
    plt.close()

    # Bar chart for time
    plt.figure(figsize=(5, 4))
    plt.bar(x, time_s, color=colors)
    plt.xticks(x, modes)
    plt.ylabel("time (s)")
    plt.title("Extreme benchmark runtime")
    plt.tight_layout()
    plt.savefig(f"{outdir}/runtime_extreme.png", dpi=150)
    plt.close()


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bench/plot_memory.py results_extreme.csv")
        sys.exit(1)
    plot_memory(sys.argv[1])
