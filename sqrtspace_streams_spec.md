
# sqrtspace-streams — Zero‑Shot Spec (MVP → v0.1.0)

**Purpose:** A small, boring-to-use **streaming pipeline wrapper** that cuts peak memory for long pipelines by evaluating in a √t-style *recompute‑on‑demand* order with **selective checkpointing** + **LRU caching**, and **auto‑fallback** when memory savings don’t help.

This spec is written for a **zero‑shot IDE** (Zed, Cursor, etc.). You can hand it this file and ask it to scaffold the repo and implement the MVP. It contains folder layout, API, acceptance tests, example notebooks/scripts, and a bench harness.

---

## Language Choice

- **Primary reference implementation:** **Python 3.10+**  
  Rationale: fastest to prototype, widest adoption for DSP/ML/data (NumPy/Numba/Pandas/PyTorch/CuPy), easy packaging, easy to run on laptops/edge devices.
- **Design for porting:** the core abstractions (Stage, Kernel, Planner, Executor, Checkpoint/Cache) are language-agnostic. We keep I/O as plain arrays/tensors to ease a later **Rust** re‑implementation with **PyO3** bindings if desired.
- **GPU (optional):** CuPy for drop-in GPU support; keep the API identical.

> Start in Python for your Zed test. Later, a Rust core (with Python bindings) can drop into the same API if you need max throughput / predictable latency.

---

## Outcomes & Non‑Goals

### Outcomes (v0.1.0 MVP)
1. **Drop‑in stages** for array pipelines: declare stages with a decorator; run pull or push streaming.
2. **√t‑mode executor**: DFS order, **checkpoint ring buffer** every *k* windows, **tiny LRU** for hot nodes.
3. **Auto‑fallback:** if the baseline fits comfortably in RAM/LLC or the wrapper would hurt latency, run the classic order.
4. **Deterministic outputs**, **bit‑identical** to baseline for pure stages; finite‑state stage support.
5. **Metrics:** peak RSS, page faults, I/O MB (Linux/macOS/Win), optional LLC‑misses on Linux.
6. **Examples:** Audio DSP, Logs, Small ML preprocessing.
7. **Bench harness** to compare classic vs √t on fit / barely‑fit / spill datasets.

### Non‑Goals (MVP)
- No distributed execution; single host, multi‑threaded/multi‑process only.
- No dynamic graph rewrite beyond stage tiling/ordering and operator fusion within a block.
- No custom tensor type; always pass native arrays/tensors through.

---

## Repository Layout

```
sqrtspace-streams/
  pyproject.toml
  README.md
  LICENSE
  src/sqrtspace_streams/
    __init__.py
    pipeline.py         # SqrtSpacePipeline, Stage/Kernel base, Planner, Executor
    stages.py           # common kernels (fft, bandpass, resample, features)
    cache.py            # LRU cache + checkpoint ring buffer
    autotune.py         # probe-based tuner (window, k, lru_bytes, workers)
    adapters/
      numpy_backend.py
      cupy_backend.py   # optional
      pandas_backend.py # phase B
      pytorch_backend.py# phase B
    metrics.py          # telemetry, perf counters, reporting
    cli.py              # `sqrtspace` CLI (run demos, profile, bench)
  tests/
    test_equivalence.py # property tests vs baseline
    test_pipeline.py    # pipeline API tests
    test_cache.py       # LRU + checkpoint correctness
    test_autotune.py    # tuner guards and bounds
  examples/
    audio_demo.ipynb
    logs_demo.ipynb
    ml_preproc_demo.ipynb
    data/               # small bundled datasets (or scripts to generate)
  bench/
    bench_audio.py
    bench_logs.py
    bench_utils.py
```

---

## Installation & Runtime

- **Python:** 3.10+
- **Dependencies (MVP):** `numpy`, `numba` (optional), `psutil`, `platformdirs`, `tqdm`, `rich`, `typing-extensions`.  
- **Optional:** `cupy-cuda12x`, `pandas`, `pyarrow`, `scipy`, `soundfile`, `pyperf` (Linux perf integration optional).

```bash
pip install -e .
sqrtspace demo audio           # run audio demo with report
sqrtspace bench audio          # run audio benchmark matrix
```

Packagers: expose the CLI via `console_scripts` so `sqrtspace` is available after install:

```toml
[project.scripts]
sqrtspace = "sqrtspace_streams.cli:main"
```

---

## Core Concepts

### 1) Stage (pure function)
A **Stage** transforms an array/tensor block to another array/tensor block. No side effects.

```python
@pipe.stage("fft", inputs=["input"])
def fft(x_block: np.ndarray, *, sr: int) -> np.ndarray: ...
```

- **Purity check:** default true. If non‑pure, declare as **Kernel** (below).
- **Signature:** positional array(s), keyword params only; return arrays. No global state.

### 2) Kernel (finite‑state operator)
Explicit state in/out. Useful for IIR filters, trackers, Viterbi decoders, etc.

```python
class BandpassKernel(Kernel):
    def init_state(self, shape, dtype, **kw) -> dict: ...
    def step(self, x_block, state, **kw) -> tuple[np.ndarray, dict]: ...
```

- **Checkpointability:** state must be **JSON‑serializable** or `np.savez` compatible.
- **Determinism:** same inputs + state → same outputs + next state.

### 3) Planner
Chooses `window`, `hop`, `k` (checkpoint interval), `lru_bytes`, `workers` to keep:
```
hot_bytes + checkpoint_bytes ≤ 0.6 × LLC_bytes
```
- Probes system cache sizes (via `psutil`/`/sys` on Linux; fallbacks elsewhere).
- Can be overridden via user config.

### 4) Executor
- **Classic mode:** straightforward forward pass (vectorized).
- **√t‑mode:** depth‑first traversal of the stage graph × time blocks; selective **checkpoint every k windows**; **LRU** small cache for hot nodes.
- **Heuristic switch:** run a quick dry‑run estimate; if fit is good or predicted slowdown >10%, fallback to classic.

### 5) Storage
- **Checkpoint ring buffer:** size `k`; stores `(t_idx, stage_id, output_block, state)` at coarse granularity.
- **LRU:** capacity `lru_bytes`; keyed by `(stage_id, t_idx)` to retain hot subtrees.
- **In-memory only (MVP):** checkpoints and LRU live in RAM; no on-disk persistence across runs.
- **Checkpoint mapping:** for each `stage_id`, store entries at `t_idx % k` (ring behavior). On collision, the older entry is discarded after ensuring all dependents have consumed it.
- **Kernel state format:** kernel `state` is serialized as NumPy-compatible structures; if needed, `np.savez`-backed bytes are stored in-memory alongside the `output_block`.

---

## Public API (Python)

```python
from sqrtspace_streams import SqrtSpacePipeline, Kernel, Config

pipe = SqrtSpacePipeline(
    window=4096, hop=1024, dtype="float32",
    checkpoint_every=16, lru_bytes=64*1024*1024,
    max_workers=8, device="cpu", backend="numpy",
    mode="auto"  # "auto" | "classic" | "sqrt"
)

@pipe.stage("fft", inputs=["input"])  # "input" is the implicit raw window source
def fft(x: np.ndarray, *, sr: int) -> np.ndarray: ...

@pipe.stage("bandpass", inputs=["fft"])
def bandpass(X: np.ndarray, *, f_lo: float, f_hi: float, sr: int) -> np.ndarray: ...

@pipe.stage("features", inputs=["bandpass"])
def features(Y: np.ndarray) -> np.ndarray: ...

@pipe.stage("classify", inputs=["features"])
def classify(F: np.ndarray) -> np.ndarray: ...

# Pull evaluation
y = pipe.evaluate("classify", t_range=(t0, t1), params={"sr": 48000, "f_lo": 80, "f_hi": 2000})

# Push streaming
pipe.push_samples(x_chunk)       # enqueue raw data
out = pipe.pull_until(t_target)  # get outputs up to time
```

#### Graph wiring (DAG)

- Stage decorator signature:

```python
@pipe.stage(name: str, inputs: list[str] | None = None, pure: bool = True)
```

- inputs: upstream stage names in order. Use the reserved name "input" to refer to the raw windowed source provided by `push_samples`.
- Function positional arguments correspond one-to-one with outputs of inputs in the same order. Stage configuration remains via keyword-only params.
- Cycles are invalid and rejected at registration.
- Multi-input example:

```python
@pipe.stage("mix", inputs=["bandpass_left", "bandpass_right"]) 
def mix(l: np.ndarray, r: np.ndarray, *, gain: float = 0.5) -> np.ndarray:
    return gain * (l + r)
```

#### Streaming semantics (push/pull)

- Time base: `t_idx` counts complete analysis windows. `t_range=(t0, t1)` is half-open on window indices `[t0, t1)`.
- `push_samples(x: np.ndarray) -> bool`: accepts 1-D arrays of dtype matching `Config.dtype`. Samples are accumulated; windows are formed internally with `window`/`hop`.
- Input buffer capacity: `Config.input_buffer_blocks` windows (default 64). If enqueuing would exceed capacity, `push_samples` returns `False` and no samples are enqueued.
- `pull_until(t_target: int) -> np.ndarray`: synchronously evaluates the target stage and returns outputs up to window index `t_target` (exclusive). If insufficient input is available for a full window, it returns outputs up to the last fully available window.
- Backpressure: call sites should honor a `False` return from `push_samples` (retry or slow producer).

#### Evaluate API

```python
def SqrtSpacePipeline.evaluate(self, target: str, t_range: tuple[int, int] | None = None, params: dict | None = None) -> np.ndarray: ...
```

### Config
```python
class Config(TypedDict, total=False):
    window: int
    hop: int
    checkpoint_every: int
    lru_bytes: int
    max_workers: int
    input_buffer_blocks: int  # default: 64
    device: Literal["cpu", "cuda"]
    backend: Literal["numpy", "cupy", "pandas", "pytorch"]
    mode: Literal["auto", "classic", "sqrt"]
```

---

## Algorithmic Details (√t‑mode)

**Goal:** minimize live memory while guaranteeing identical outputs to classic mode.

1. **Time blocking:** split stream into windows of `W` samples with hop `H`.
2. **Tree:** each `stage` × `t_idx` is a node; parents depend on children at same `t_idx` (or neighbor windows if overlap).
3. **DFS traversal:** evaluate from requested output node downwards, depth‑first.
4. **Checkpointing:** every `k` windows, persist the full `(stage_out, state)` for selected stages (by default, the **lowest expensive producer** stage to bound recompute depth).
5. **LRU caching:** keep most recently used nodes in RAM; small cap (e.g., 32–128MB) to avoid thrash.
6. **Autotune:** pick `W,k,LRU` so `hot_bytes+saved_bytes ≤ 0.6 × LLC` and **estimated recompute** < 20% time overhead. If not satisfied → fallback.

Pseudo‑code:
```python
def eval_node(stage_id, t_idx):
    if cache.has(stage_id, t_idx): return cache.get(...)
    if checkpoint.has(stage_id, t_idx): return checkpoint.load(...)
    inputs = [eval_node(pred, t_idx) for pred in preds(stage_id)]
    out = stage_fn[stage_id](*inputs)
    maybe_cache(stage_id, t_idx, out)
    if t_idx % k == 0 and is_checkpoint_stage(stage_id):
        checkpoint.save(stage_id, t_idx, out)
    return out
```

---

## Correctness & Equivalence

- **Pure stages:** integers must be bit‑identical. For floating point, use dtype-aware tolerances by default: 
  - float32: `np.allclose(a, b, rtol=1e-5, atol=1e-6)`
  - float64: `np.allclose(a, b, rtol=1e-7, atol=1e-9)`
  - complex: apply the same per-component tolerances.
  Individual tests may tighten/loosen tolerances for known numerically-sensitive ops (e.g., FFT windowing).
- **Kernels:** equality on `(output, next_state)` vs baseline classic order.  
- **Seeds:** set global seeds; stage functions must be deterministic given inputs.

---

## Acceptance Criteria (MVP)

1. **API works:** users can declare 4–6 stages, run pull or push modes.
2. **Equivalence tests pass:** classic vs √t produce identical results on demos.
3. **Memory savings:** on “barely‑fit” dataset, **peak RSS ↓ ≥5×**, **page faults ↓**.
4. **No regressions:** on “fits easily” dataset, **runtime within ±10%** due to auto‑fallback.
5. **Docs & Demos:** `sqrtspace demo audio` runs end‑to‑end, printing a concise report.

---

## CLI

```bash
sqrtspace demo audio          # run audio pipeline demo with report
sqrtspace demo logs           # run logs demo
sqrtspace bench audio         # run a benchmark matrix (window,k,LRU,workers)
sqrtspace plan --print        # print chosen plan for given config
```

---

## Examples

### Audio DSP (48 kHz)
Stages:
- `fft`: windowed FFT (hann), complex spectrum
- `bandpass`: retain bins in [80, 2000] Hz
- `features`: spectral peaks, centroid, kurtosis
- `classify`: tiny MLP/GBDT (CPU)

Script sketch (provided in `examples/audio_demo.ipynb`):
```python
pipe = SqrtSpacePipeline(window=4096, hop=1024, dtype="float32",
                         checkpoint_every=16, lru_bytes=64<<20, mode="auto")
# ... declare stages ...
y = pipe.evaluate("classify", t_range=(0, 120))  # 120 s stream
```

### Logs (rolling analytics)
Stages: regex filter → rolling window aggregates → quantiles → alert flags.  
Use `pandas_backend` (Phase B).

### ML Preprocessing (image/audio shingling)
Stages: tile → augment → features → enqueue to DataLoader.  
Targets stable memory + zero OOM.

---

## Autotuner (MVP)

- **Inputs:** sample of input stats (row size, dtype, bandwidth), stage memory multipliers (dev notes below).
- **Outputs:** (`window`, `hop`, `k`, `lru_bytes`, `workers`).
- **Procedure:** probe small runs at 2–3 candidate plans over a subset of the stream (fraction `0.03`, capped at `probe_max_windows=64`); pick best predicted (`peak_RSS`, `page_faults`, `ETA`).  
- **Guardrails:** if estimated slow‑down `> 0.10` and baseline fits cache comfortably → choose classic. Enforce recompute overhead budget `< 0.20`.
- **LLC detection:** use platform probes when available; if unknown, assume `LLC_bytes = 32 << 20` (32 MiB). Respect `hot_bytes + checkpoint_bytes ≤ 0.6 × LLC_bytes`.

---

## Metrics

- Peak RSS (psutil), Major page faults, Read/Write MB (proc IO counters).
- Optional Linux perf: LLC load misses.
- Report template (Rich):
```
Mode: sqrt (auto)   Window: 4096   k: 16   LRU: 64MB   Workers: 8
Peak RSS: 1.1 GB  (baseline 6.8 GB)   Major Faults: 0  (baseline 4231)
LLC Misses: 1.4e8  (baseline 5.9e8)    Time: 0.83× baseline (barely-fit set)
Outputs: bit-identical (pure stages)
```

---

## Implementation Notes

- **Backends:** start with **NumPy** (CPU). Optional **CuPy**: maintain identical API; careful with tile sizes to amortize kernel launch overhead and H2D/D2H copies.
- **Operator fusion:** within a stage block, avoid transient allocations (in‑place ops where safe).
- **Threading:** thread pool with per‑worker scratch buffers sized to L2/L3; cap workers by `LLC / worker_hot_bytes`.
- **I/O:** prefer memory‑mapped reads for large inputs (logs); contiguous tiles for audio.
- **Stateful kernels:** make state explicit; snapshot at checkpoints; restore on recompute paths.
- **Determinism:** fix seeds; avoid nondeterministic BLAS if possible (note in docs).

---

## Testing Strategy

- **Property tests:** equality classic vs √t on random seeds & sizes.
- **Edge cases:** tiny window, large window, odd hop; empty input; NaNs/inf pass‑through.
- **Cache/Checkpoint:** evict/restore correctness; ring buffer wraparound.
- **Autotune:** respects RAM/LLC bounds; falls back when estimates say “no benefit”.

---

## Bench Harness

- Datasets: (a) fits in RAM/LLC; (b) barely fits; (c) spills.  
- Run matrix over window ∈ {2k,4k,8k}, k ∈ {8,16,32}, LRU ∈ {32,64,128} MB, workers ∈ {1,4,8}.  
- Record: time, RSS, faults, LLC misses. Output CSV + plot.

---

## Roadmap (post‑MVP)

- **Pandas/Polars** adapter for table/column pipelines.
- **PyTorch** adapter for preprocessing.
- **Rust core** (rayon + cache‑friendly tiling), expose PyO3 bindings.
- **Resume/replay**: serialize plan + checkpoints for crash‑safe long streams.
- **GPU pipeline**: pinned host buffers, overlap copies, kernel fusion helpers.

---

## Coding Standards

- Type‑annotate all public functions; run `mypy` in CI.
- Black + Ruff formatting; pre‑commit hooks.
- Docstrings in NumPy style; API docs via mkdocs or pdoc.
- CI: unit tests on Linux/macOS/Windows; optional GPU CI nightly.

---

## Appendix: Minimal Baselines

Provide plain, classic executors for all demos so equivalence and performance comparisons are clear and reproducible.

---

## License

MIT (or BSD‑3‑Clause).

---

## Quickstart Checklist (for the IDE)

1. Create repo scaffold per layout.
2. Implement `SqrtSpacePipeline` with `stage` decorator, basic Planner, classic & sqrt Executors.
3. Implement `cache.py`: LRU (by bytes) + checkpoint ring buffer.
4. Implement `autotune.py`: probe 2–3 plans, guard auto‑fallback.
5. Implement `stages.py`: fft/bandpass/features (NumPy); tiny MLP classifier (NumPy).
6. Implement `cli.py`: `demo audio`, `bench audio`, `plan`.
7. Write tests in `tests/` to prove equivalence and memory reductions.
8. Add `examples/audio_demo.ipynb` and `bench/bench_audio.py`.
9. Ensure `pip install -e .` and `sqrtspace demo audio` work on a fresh env.
10. Document in `README.md` with gif/screenshot of the report.

---

*End of spec.*
