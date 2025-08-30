import numpy as np


def test_classic_pipeline_evaluates_windows():
    from sqrtspace_streams.pipeline import SqrtSpacePipeline

    pipe = SqrtSpacePipeline(window=256, hop=128)

    @pipe.stage("fft", inputs=["input"])
    def fft(x: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(x, axis=1))

    @pipe.stage("feat", inputs=["fft"])
    def feat(X: np.ndarray) -> np.ndarray:
        return np.argmax(X, axis=1, keepdims=True).astype(np.int32)

    sr = 4096
    sig = np.sin(2 * np.pi * 440 * np.arange(sr, dtype=np.float32) / sr).astype(np.float32)
    assert pipe.push_samples(sig)
    out = pipe.evaluate("feat")
    assert out.ndim == 2
    assert out.shape[1] == 1
    # For 256 window, 128 hop, 4096 samples => 1 + (4096-256)//128 = 31 windows
    assert out.shape[0] == 31


