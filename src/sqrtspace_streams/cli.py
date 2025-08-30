from __future__ import annotations

import argparse
from rich.console import Console


def main() -> None:
    parser = argparse.ArgumentParser(prog="sqrtspace", description="SqrtSpace Streams CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a demo")
    demo.add_argument("which", choices=["audio", "logs"], help="Demo to run")
    demo.add_argument("--mode", choices=["auto", "classic", "sqrt"], default="auto")

    bench = sub.add_parser("bench", help="Run a benchmark")
    bench.add_argument("which", choices=["audio", "logs"], help="Benchmark to run")
    bench.add_argument("--iters", type=int, default=1)

    args = parser.parse_args()
    console = Console()

    if args.cmd == "demo" and args.which == "audio":
        from .pipeline import SqrtSpacePipeline
        import numpy as np
        from .metrics import snapshot

        console.print("[bold]Running audio demo (synthetic tone)...[/bold]")

        # Generate 2 seconds of a tone
        sr = 48000
        t = np.arange(sr * 2, dtype=np.float32)
        sig = np.sin(2 * np.pi * 440.0 * t / sr).astype(np.float32)

        # Allocate a sufficient input buffer for the demo (in hop-sized blocks)
        hop = 1024
        input_blocks = max(64, int(sig.size // hop) + 8)
        pipe = SqrtSpacePipeline(input_buffer_blocks=input_blocks, mode=args.mode)

        @pipe.stage("fft", inputs=["input"])
        def fft(x: np.ndarray, *, sr: int = 48000) -> np.ndarray:
            # naive real FFT magnitude per window
            X = np.fft.rfft(x, axis=1)
            return np.abs(X)

        @pipe.stage("features", inputs=["fft"])
        def features(X: np.ndarray) -> np.ndarray:
            # simple feature: max bin per window
            max_bin = np.argmax(X, axis=1, keepdims=True)
            return max_bin.astype(np.int32)

        ok = pipe.push_samples(sig)
        if not ok:
            console.print("[red]Failed to enqueue samples[/red]")
            raise SystemExit(1)

        before = snapshot()
        out = pipe.evaluate("features")
        after = snapshot()
        console.print(f"Output windows: {out.shape[0]}")
        console.print(f"Mode: {args.mode}  RSS: {after.peak_rss_bytes} bytes  IO: R {after.read_mb} MB / W {after.write_mb} MB")
    elif args.cmd == "demo" and args.which == "logs":
        from .pipeline import SqrtSpacePipeline
        import numpy as np
        from .metrics import snapshot

        console.print("[bold]Running logs demo (synthetic latencies)...[/bold]")

        # Synthetic heavy-tail latencies with occasional spikes
        n = 2_000_000
        rng = np.random.default_rng(42)
        base = rng.lognormal(mean=2.0, sigma=0.5, size=n).astype(np.float32)
        spikes_idx = rng.integers(0, n, size=n // 500)
        base[spikes_idx] *= 5.0

        hop = 2048
        input_blocks = max(64, int(n // hop) + 8)
        pipe = SqrtSpacePipeline(window=8192, hop=hop, input_buffer_blocks=input_blocks, mode=args.mode)

        @pipe.stage("quantiles", inputs=["input"])
        def quantiles(x: np.ndarray) -> np.ndarray:
            # Compute p50 and p95 per window
            p50 = np.percentile(x, 50, axis=1, method="nearest")
            p95 = np.percentile(x, 95, axis=1, method="nearest")
            return np.stack([p50, p95], axis=1).astype(np.float32)

        @pipe.stage("alerts", inputs=["quantiles"])
        def alerts(q: np.ndarray) -> np.ndarray:
            # Flag when p95 exceeds a fixed threshold (heuristic)
            thresh = 20.0
            flag = (q[:, 1] > thresh).astype(np.int8)
            return flag[:, None]

        ok = pipe.push_samples(base)
        if not ok:
            console.print("[red]Failed to enqueue samples[/red]")
            raise SystemExit(1)

        before = snapshot()
        out = pipe.evaluate("alerts")
        after = snapshot()
        alerts_count = int(out.sum()) if out.size else 0
        console.print(f"Output windows: {out.shape[0]}  Alerts: {alerts_count}")
        console.print(f"Mode: {args.mode}  RSS: {after.peak_rss_bytes} bytes  IO: R {after.read_mb} MB / W {after.write_mb} MB")
    elif args.cmd == "bench" and args.which == "logs":
        from time import perf_counter
        import numpy as np
        from .pipeline import SqrtSpacePipeline

        # fixed dataset for repeatability
        n = 4_000_000
        rng = np.random.default_rng(123)
        base = rng.lognormal(mean=2.0, sigma=0.6, size=n).astype(np.float32)
        hop = 4096
        win = 16384
        input_blocks = max(64, int(n // hop) + 8)

        def build_pipe(mode: str, k: int, lru: int):
            p = SqrtSpacePipeline(window=win, hop=hop, input_buffer_blocks=input_blocks, mode=mode, checkpoint_every=k, lru_bytes=lru)

            @p.stage("quantiles", inputs=["input"])
            def quantiles(x: np.ndarray) -> np.ndarray:
                p50 = np.percentile(x, 50, axis=1, method="nearest")
                p95 = np.percentile(x, 95, axis=1, method="nearest")
                return np.stack([p50, p95], axis=1).astype(np.float32)

            @p.stage("alerts", inputs=["quantiles"])
            def alerts(q: np.ndarray) -> np.ndarray:
                thresh = 25.0
                flag = (q[:, 1] > thresh).astype(np.int8)
                return flag[:, None]

            return p

        configs = []
        for k in (8, 16, 32):
            for lru_mb in (32, 64, 128):
                configs.append((k, lru_mb << 20))

        console.print("k  LRU(MB)   time_classic(s)   time_sqrt(s)   speedup(classic/sqrt)")
        for (k, lru) in configs:
            # classic
            pc = build_pipe("classic", k, lru)
            pc.push_samples(base)
            t0 = perf_counter()
            pc.evaluate("alerts")
            tc = perf_counter() - t0

            # sqrt
            ps = build_pipe("sqrt", k, lru)
            ps.push_samples(base)
            t0 = perf_counter()
            ps.evaluate("alerts")
            ts = perf_counter() - t0

            sp = (tc / ts) if ts > 0 else float('inf')
            console.print(f"{k:<3d}{(lru>>20):>8d}{tc:>16.3f}{ts:>15.3f}{sp:>20.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()


