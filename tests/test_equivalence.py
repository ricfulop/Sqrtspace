import numpy as np


def test_classic_equals_sqrt_simple_pipeline():
    from sqrtspace_streams.pipeline import SqrtSpacePipeline

    def build(mode: str):
        pipe = SqrtSpacePipeline(window=256, hop=128, mode=mode)

        @pipe.stage("fft", inputs=["input"])
        def fft(x: np.ndarray) -> np.ndarray:
            return np.abs(np.fft.rfft(x, axis=1))

        @pipe.stage("feat", inputs=["fft"])
        def feat(X: np.ndarray) -> np.ndarray:
            return np.argmax(X, axis=1, keepdims=True).astype(np.int32)

        return pipe

    sr = 4096
    sig = np.sin(2 * np.pi * 440 * np.arange(sr, dtype=np.float32) / sr).astype(np.float32)

    p1 = build("classic")
    assert p1.push_samples(sig)
    out1 = p1.evaluate("feat")

    p2 = build("sqrt")
    assert p2.push_samples(sig)
    out2 = p2.evaluate("feat")

    assert np.array_equal(out1, out2)


