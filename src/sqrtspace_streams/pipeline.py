from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
from .cache import LRUCache, CheckpointRing


class Config(TypedDict, total=False):
    window: int
    hop: int
    checkpoint_every: int
    lru_bytes: int
    max_workers: int
    input_buffer_blocks: int
    device: Literal["cpu", "cuda"]
    backend: Literal["numpy", "cupy", "pandas", "pytorch"]
    mode: Literal["auto", "classic", "sqrt"]


class Kernel:
    def init_state(self, shape, dtype, **kw) -> Dict[str, Any]:  # pragma: no cover (interface)
        raise NotImplementedError

    def step(self, x_block, state, **kw) -> Tuple[np.ndarray, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError


StageFn = Callable[..., np.ndarray]


@dataclass
class _StageDef:
    name: str
    fn: StageFn
    inputs: List[str] = field(default_factory=list)
    pure: bool = True


class SqrtSpacePipeline:
    def __init__(self, **cfg: Config) -> None:
        self.cfg: Config = {
            "window": 4096,
            "hop": 1024,
            "checkpoint_every": 16,
            "lru_bytes": 64 * 1024 * 1024,
            "max_workers": 8,
            "input_buffer_blocks": 64,
            "device": "cpu",
            "backend": "numpy",
            "mode": "auto",
        }
        self.cfg.update(cfg)

        self._stages: Dict[str, _StageDef] = {}
        self._topo_valid: bool = False
        self._input_buffer: List[np.ndarray] = []

    # Stage decorator
    def stage(self, name: str, inputs: Optional[List[str]] = None, pure: bool = True):
        inputs = inputs or []

        def _decorator(fn: StageFn) -> StageFn:
            if name in self._stages:
                raise ValueError(f"Stage '{name}' already defined")
            self._stages[name] = _StageDef(name=name, fn=fn, inputs=list(inputs), pure=pure)
            self._topo_valid = False
            return fn

        return _decorator

    # Graph utilities
    def _toposort(self) -> List[str]:
        if self._topo_valid:
            return self._cached_topo
        indeg: Dict[str, int] = {n: 0 for n in self._stages}
        for n, sd in self._stages.items():
            for u in sd.inputs:
                if u == "input":
                    continue
                if u not in self._stages:
                    raise ValueError(f"Stage '{n}' depends on unknown input '{u}'")
                indeg[n] += 1
        # Kahn
        ready = [n for n, d in indeg.items() if d == 0]
        order: List[str] = []
        seen_edges = 0
        # Build adjacency
        succ: Dict[str, List[str]] = {n: [] for n in self._stages}
        for n, sd in self._stages.items():
            for u in sd.inputs:
                if u == "input":
                    continue
                succ[u].append(n)
        while ready:
            v = ready.pop()
            order.append(v)
            for w in succ[v]:
                indeg[w] -= 1
                seen_edges += 1
                if indeg[w] == 0:
                    ready.append(w)
        if len(order) != len(self._stages):
            raise ValueError("Cycle detected in stage DAG")
        self._cached_topo = order
        self._topo_valid = True
        return order

    # Streaming input
    def push_samples(self, x: np.ndarray) -> bool:
        if x.ndim != 1:
            raise ValueError("push_samples expects 1-D array")
        cap = int(self.cfg["input_buffer_blocks"]) * int(self.cfg["hop"])  # approximate cap in samples
        cur = sum(arr.size for arr in self._input_buffer)
        if cur + x.size > cap:
            return False
        self._input_buffer.append(np.asarray(x))
        return True

    def _form_windows(self) -> np.ndarray:
        if not self._input_buffer:
            return np.empty((0, self.cfg["window"]), dtype=np.float32)
        buf = np.concatenate(self._input_buffer, axis=0)
        W = int(self.cfg["window"]) 
        H = int(self.cfg["hop"]) 
        if buf.size < W:
            return np.empty((0, W), dtype=buf.dtype)
        num = 1 + (buf.size - W) // H
        if num <= 0:
            return np.empty((0, W), dtype=buf.dtype)
        windows = np.lib.stride_tricks.sliding_window_view(buf, W)[::H]
        # keep only the consumed part in buffer
        consumed = (num - 1) * H + W
        remain = buf[consumed:]
        self._input_buffer = [remain] if remain.size else []
        return windows

    # Classic evaluation (pull API)
    def evaluate(self, target: str, t_range: Optional[Tuple[int, int]] = None, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        params = params or {}
        topo = self._toposort()
        if target not in self._stages:
            raise ValueError(f"Unknown target stage '{target}'")
        windows = self._form_windows()
        # Apply t_range as half-open [t0, t1)
        if t_range is not None:
            t0, t1 = t_range
            windows = windows[t0:t1]
        # Choose mode
        mode = self.cfg.get("mode", "auto")
        if mode == "classic" or (mode == "auto" and self._choose_classic(windows)):
            return self._evaluate_classic(windows, target, params)
        return self._evaluate_sqrt(windows, target, params)

    def _evaluate_classic(self, windows: np.ndarray, target: str, params: Dict[str, Any]) -> np.ndarray:
        outputs: Dict[str, np.ndarray] = {}
        for name in self._toposort():
            sd = self._stages[name]
            args: List[np.ndarray] = []
            for u in sd.inputs:
                if u == "input":
                    args.append(windows)
                else:
                    args.append(outputs[u])
            out = sd.fn(*args, **params)
            outputs[name] = out
        return outputs[target]

    def _evaluate_sqrt(self, windows: np.ndarray, target: str, params: Dict[str, Any]) -> np.ndarray:
        # Simple DFS per t_idx with LRU and checkpoint ring
        k = int(self.cfg.get("checkpoint_every", 16))
        lru_bytes = int(self.cfg.get("lru_bytes", 64 * 1024 * 1024))
        lru = LRUCache(capacity_bytes=lru_bytes)
        ring = CheckpointRing(k=k)
        checkpoint_stages = {name for name, sd in self._stages.items() if "input" in sd.inputs}

        def eval_node(stage_id: str, t_idx: int) -> np.ndarray:
            key = (stage_id, t_idx)
            cached = lru.get(key)
            if cached is not None:
                return cached
            ckpt = ring.load(stage_id, t_idx)
            if ckpt is not None:
                return ckpt
            sd = self._stages[stage_id]
            # Prepare input args as single-window batches
            args: List[np.ndarray] = []
            for u in sd.inputs:
                if u == "input":
                    args.append(windows[t_idx : t_idx + 1])
                else:
                    args.append(eval_node(u, t_idx))
            out = sd.fn(*args, **params)
            # Ensure output is a batch of size 1
            if out.ndim == 1:
                out = out[None, ...]
            # maybe cache
            size_bytes = getattr(out, "nbytes", 0)
            lru.put(key, out, size_bytes=size_bytes)
            # maybe checkpoint
            if k > 0 and (t_idx % k == 0) and (stage_id in checkpoint_stages):
                ring.save(stage_id, t_idx, out)
            return out

        num = windows.shape[0]
        rows: List[np.ndarray] = []
        for t_idx in range(num):
            rows.append(eval_node(target, t_idx))
        return np.concatenate(rows, axis=0) if rows else np.empty((0,), dtype=windows.dtype)

    def _choose_classic(self, windows: np.ndarray) -> bool:
        # Heuristic: if windows * per-stage feature sizes likely fits in LRU budget or few windows, use classic
        num = windows.shape[0]
        if num <= 32:
            return True
        per_win_bytes = getattr(windows, "dtype", np.dtype("float32")).itemsize * windows.shape[1]
        stages = max(1, len(self._stages))
        est_live = per_win_bytes * stages
        return est_live < 0.6 * float(self.cfg.get("lru_bytes", 64 * 1024 * 1024))

    # Minimal pull_until using classic mode for now (will switch per mode)
    def pull_until(self, t_target: int, target: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        # For MVP, evaluate full available range up to t_target
        return self.evaluate(target or self._last_target(), t_range=(0, t_target), params=params)

    def _last_target(self) -> str:
        # heuristic: last topo node
        return self._toposort()[-1]


