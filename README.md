## SqrtSpace Streams

A streaming pipeline engine that lowers peak memory for long-running array/tensor pipelines by recomputing on demand (√t-style), with selective checkpointing, a tiny LRU cache, and an auto-fallback to classic order when memory savings won’t help.

### Badges

[![status](https://img.shields.io/badge/status-alpha-orange.svg)](#)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![license](https://img.shields.io/badge/license-MIT-lightgray.svg)](LICENSE)

### Install

```bash
pip install -e .
```

### CLI

```bash
sqrtspace demo audio --mode auto    # or: classic | sqrt
```

Prints a short report with output shape and process RSS.

### Python API (quickstart)

```python
import numpy as np
from sqrtspace_streams import SqrtSpacePipeline

pipe = SqrtSpacePipeline(
    window=4096,
    hop=1024,
    checkpoint_every=16,
    lru_bytes=64*1024*1024,
    mode="auto",  # "auto" | "classic" | "sqrt"
)

@pipe.stage("fft", inputs=["input"])  # "input" = raw window source
def fft(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(x, axis=1))

@pipe.stage("features", inputs=["fft"])
def features(X: np.ndarray) -> np.ndarray:
    return np.argmax(X, axis=1, keepdims=True).astype(np.int32)

# Push samples, then evaluate
sr = 48_000
t = np.arange(sr * 2, dtype=np.float32)
sig = np.sin(2*np.pi*440.0*t/sr).astype(np.float32)
pipe.push_samples(sig)
out = pipe.evaluate("features")
print(out.shape)
```

### Modes
- **classic**: Straightforward forward pass for best latency when memory is plentiful.
- **sqrt**: Depth-first evaluation per time window with selective checkpointing every k windows and an LRU cache to cap live memory.
- **auto**: Simple heuristic to choose between classic and sqrt based on window count and estimated live bytes.

Knobs:
- **checkpoint_every (k)**: checkpoint interval (windows). For safety on overlapped windows, choose k ≥ the temporal dependency span implied by window/hop.
- **lru_bytes**: cap for hot entries held in RAM.

### Metrics
- CLI prints process RSS and basic I/O counters after execution.

### Plotting
- Time/speedup charts from CSV:
```bash
sqrtspace bench logs --csv logs_results.csv
python bench/plot_bench.py logs_results.csv
# outputs PNGs in bench_out/
```
- Extreme peak memory/time charts:
```bash
sqrtspace bench logs --extreme --n 20000000 --win 32768 --hop 4096 --csv extreme_logs.csv
python bench/plot_memory.py extreme_logs.csv
# outputs PNGs in bench_out/
```

### Docker demo (resource-limited, apples-to-apples)
- Build image:
```bash
make build
```
- Barely-fit time/speedup (RAD750/GR740-like caps):
```bash
make demo-barely
```
- Spill with peak memory/time:
```bash
make demo-spill
```
- Satcom deeper pipeline:
```bash
make satcom-demo
```
These use cgroup limits (RAM/CPU) and `LLC_BYTES` to mirror constrained OBCs. Classic will page on spill; sqrt keeps a tiny live set.

### Target OBC profiles (what we emulate)
- RAD750 (BAE Systems, PowerPC 750)
  - Flown on: Mars Reconnaissance Orbiter (MRO), Mars Science Laboratory Curiosity, Mars 2020 Perseverance, Juno, Lunar Reconnaissance Orbiter (LRO), Kepler, Fermi (GLAST), others.
  - Representative constraints: single core, a few hundred MB RAM, modest cache.
- LEON/GR family (Cobham Gaisler/Microchip, SPARC)
  - GR712 (LEON3‑FT) and GR740 (LEON4‑FT) across many ESA missions (e.g., Solar Orbiter, Sentinel/Copernicus, JUICE‑class avionics).
  - Representative constraints: 1–4 cores, 256–512 MB RAM, modest cache.

### Dev
- Run demo without installing (local): `PYTHONPATH=src python3 -m sqrtspace_streams.cli demo audio`
- Bench (toy): `PYTHONPATH=src python3 bench/bench_audio.py`
- Tests (example): add your runner, e.g., `pytest` (skeletons included under `tests/`).
- IDE: install optional dev deps for linting/typing: `pip install -e .[dev]`.

### Contributing
- Issues and PRs welcome. Please keep PRs focused and small when possible.
- Style: Black + Ruff; type-check public APIs with mypy.
- Before submitting:
  - Run unit tests (add `pytest` and a runner if not present).
  - Ensure `pip install -e .` succeeds and `sqrtspace demo audio` runs.
  - Match code style and add concise docstrings for public functions.
