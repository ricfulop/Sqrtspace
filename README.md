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
- **dependency_span**: derived as ceil(window / hop); auto mode ensures k ≥ dependency_span.

### Metrics
- CLI prints process RSS and basic I/O counters after execution.

### Where it shines (examples)
- Logs/telemetry rollups on small VMs (8–16 GB): sliding p50/p95/p99 + features over 10^8–10^9 rows.
  - Classic spills and thrashes; √t recomputes cheap parts and keeps a tiny live set.
  - Typical: 10–60× faster; 10–50× lower peak RSS.
- Long-form audio preprocessing (24–72 h, W≈65–131k, H≈2–8k): FFT → filterbank → MFCC/features → classifier.
  - Classic swaps with large intermediates; √t bounds live memory.
  - Typical: 8–40× faster; 10–80× lower peak RSS.
- Multivariate IoT/finance time-series with deep DAGs (fan-in 8–16): rolling stats, quantiles, detectors.
  - Typical: 30–100× vs thrashing baselines.

#### Satcom examples (extreme-friendly)
- Deep‑space Doppler/drift search (time–frequency cube)
  - Pipeline: IQ → windowed FFT → coherent avg → noncoherent sum over drift tracks → peak pick.
  - Why classic thrashes: large spectrogram cubes or many drift hypotheses blow RAM.
  - √t fix: checkpoint at FFT/channelizer every k (k ≥ ceil(W/H)); small LRU for hot nodes.
  - Example sizing: Fs 1–2 MHz; duration 10–30 min; W 65–131k; H 2–8k; 200–1000 drift tracks.
  - Impact: 10–50× faster; 20–100× lower peak RSS.
- Wideband channelizer + burst decode (multi‑channel scan)
  - Pipeline: IQ → PFB channelizer (512–2048 bins) → CFO/timing → matched filter → soft demod → FEC.
  - Why classic thrashes: per‑channel buffers kept live across windows.
  - √t fix: checkpoint PFB outputs sparsely, recompute only when energy/sync triggers.
  - Example sizing: Fs 2–10 MHz; W 32–64k; H 2–8k; sparse bursts.
  - Impact: 10–40× faster; 10–50× memory cut.
- Multi‑hypothesis demod search (CFO × symbol‑rate grid)
  - Pipeline: IQ → CFO correction (grid) → resample (grid) → timing → demod → cost.
  - Why classic thrashes: per‑hypothesis intermediates multiply memory.
  - √t fix: share/recompute low‑level stages per window; checkpoint at resampler/FFT.
  - Example sizing: CFO steps 100–200; symbol rates 3–10; W 64–128k; H small.
  - Impact: 15–60× runtime vs paging; large RSS reduction.

Conditions for 20–100× speedups
- Working set ≫ RAM/LLC (barely-fit or spill).
- Large windows with overlap (big W/H), deep/wide DAG.
- Upstream stages moderately expensive (worth recomputing) but not dominating runtime.

### Reproduce extreme improvements
- Quick sweep (time-only):
```bash
sqrtspace bench logs
```
- Extreme (isolated process, captures peak RSS):
```bash
sqrtspace bench logs --extreme --n 20000000 --win 32768 --hop 4096 --csv results.csv
```
- Force smaller cache budget so auto picks sqrt earlier:
```bash
LLC_BYTES=33554432 sqrtspace bench logs --extreme --n 20000000 --win 32768 --hop 4096
```

### Plotting
- Time/speedup charts from CSV:
```bash
sqrtspace bench logs --csv logs_results.csv
python bench/plot_bench.py logs_results.csv
# outputs PNGs in bench_out/
```
- Extreme memory/time charts:
```bash
sqrtspace bench logs --extreme --n 20000000 --win 32768 --hop 4096 --csv extreme_logs.csv
python bench/plot_memory.py extreme_logs.csv
# outputs PNGs in bench_out/
```

### Dev
- Run demo without installing (local): `PYTHONPATH=src python3 -m sqrtspace_streams.cli demo audio`
- Bench (toy): `PYTHONPATH=src python3 bench/bench_audio.py`
- Tests (example): add your runner, e.g., `pytest` (skeletons included under `tests/`).

### Contributing
- Issues and PRs welcome. Please keep PRs focused and small when possible.
- Style: Black + Ruff; type-check public APIs with mypy.
- Before submitting:
  - Run unit tests (add `pytest` and a runner if not present).
  - Ensure `pip install -e .` succeeds and `sqrtspace demo audio` runs.
  - Match code style and add concise docstrings for public functions.
