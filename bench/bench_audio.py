from __future__ import annotations

import time
import numpy as np


def run_once(mode: str = "classic") -> float:
    from sqrtspace_streams.pipeline import SqrtSpacePipeline

    pipe = SqrtSpacePipeline(mode="classic" if mode == "classic" else "sqrt")

    @pipe.stage("fft", inputs=["input"])
    def fft(x: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(x, axis=1))

    @pipe.stage("feat", inputs=["fft"])
    def feat(X: np.ndarray) -> np.ndarray:
        return np.argmax(X, axis=1, keepdims=True).astype(np.int32)

    sr = 48000
    secs = 5
    t = np.arange(sr * secs, dtype=np.float32)
    sig = np.sin(2 * np.pi * 440.0 * t / sr).astype(np.float32)
    pipe.push_samples(sig)
    t0 = time.time()
    pipe.evaluate("feat")
    return time.time() - t0


if __name__ == "__main__":  # pragma: no cover
    dt = run_once("classic")
    print(f"classic: {dt:.3f}s")

