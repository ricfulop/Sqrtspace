from __future__ import annotations

import argparse
from rich.console import Console
import json
import time
import os


def main() -> None:
    parser = argparse.ArgumentParser(prog="sqrtspace", description="SqrtSpace Streams CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a demo")
    demo.add_argument("which", choices=["audio", "logs"], help="Demo to run")
    demo.add_argument("--mode", choices=["auto", "classic", "sqrt"], default="auto")

    bench = sub.add_parser("bench", help="Run a benchmark")
    bench.add_argument("which", choices=["audio", "logs"], help="Benchmark to run")
    bench.add_argument("--iters", type=int, default=1)
    bench.add_argument("--n", type=int, default=None, help="Dataset size (logs)")
    bench.add_argument("--win", type=int, default=None, help="Window size")
    bench.add_argument("--hop", type=int, default=None, help="Hop size")
    bench.add_argument("--extreme", action="store_true", help="Measure peak RSS via subprocess")
    bench.add_argument("--csv", type=str, default=None, help="Write CSV results to path")

    # internal helper (undocumented): run one eval and report JSON
    eval_once = sub.add_parser("eval-once", help=argparse.SUPPRESS)
    eval_once.add_argument("which", choices=["logs"], help=argparse.SUPPRESS)
    eval_once.add_argument("--mode", choices=["classic", "sqrt"], required=True)
    eval_once.add_argument("--n", type=int, required=True)
    eval_once.add_argument("--win", type=int, required=True)
    eval_once.add_argument("--hop", type=int, required=True)

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
        console.print("Tip: for an extreme benchmark with peak RSS, run: \n  sqrtspace bench logs --extreme --n 20000000 --win 32768 --hop 4096")
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

        # parameters
        n = args.n or 4_000_000
        win = args.win or 16384
        hop = args.hop or 4096
        rng = np.random.default_rng(123)
        base = rng.lognormal(mean=2.0, sigma=0.6, size=n).astype(np.float32)
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

        # Extreme mode: run each mode in a fresh subprocess and capture peak RSS via resource.ru_maxrss
        if args.extreme:
            import subprocess, sys, shlex
            def run_once(mode: str):
                cmd = f"{shlex.quote(sys.executable)} -m sqrtspace_streams.cli eval-once logs --mode {mode} --n {n} --win {win} --hop {hop}"
                out = subprocess.check_output(cmd, shell=True, env=os.environ)
                return json.loads(out.decode('utf-8'))

            res_c = run_once("classic")
            res_s = run_once("sqrt")
            sp = (res_c["time_s"] / res_s["time_s"]) if res_s["time_s"] > 0 else float('inf')
            console.print("mode     time(s)   peak_rss(MB)   windows   alerts")
            console.print(f"classic  {res_c['time_s']:.3f}      {res_c['peak_rss_mb']:.1f}         {res_c['windows']}      {res_c['alerts']}")
            console.print(f"sqrt     {res_s['time_s']:.3f}      {res_s['peak_rss_mb']:.1f}         {res_s['windows']}      {res_s['alerts']}")
            console.print(f"speedup classic/sqrt: {sp:.2f}×  memory ratio classic/sqrt: {(res_c['peak_rss_mb']/max(1e-6,res_s['peak_rss_mb'])):.2f}×")
            if args.csv:
                with open(args.csv, "w") as f:
                    f.write("mode,time_s,peak_rss_mb,windows,alerts\n")
                    for r, m in ((res_c, "classic"), (res_s, "sqrt")):
                        f.write(f"{m},{r['time_s']:.6f},{r['peak_rss_mb']:.3f},{r['windows']},{r['alerts']}\n")
            return

        # Non-extreme: in-process time-only sweep across configs
        configs = []
        for k in (8, 16, 32):
            for lru_mb in (32, 64, 128):
                configs.append((k, lru_mb << 20))

        console.print("k  LRU(MB)   time_classic(s)   time_sqrt(s)   speedup(classic/sqrt)")
        for (k, lru) in configs:
            pc = build_pipe("classic", k, lru)
            pc.push_samples(base)
            t0 = perf_counter()
            pc.evaluate("alerts")
            tc = perf_counter() - t0

            ps = build_pipe("sqrt", k, lru)
            ps.push_samples(base)
            t0 = perf_counter()
            ps.evaluate("alerts")
            ts = perf_counter() - t0

            sp = (tc / ts) if ts > 0 else float('inf')
            console.print(f"{k:<3d}{(lru>>20):>8d}{tc:>16.3f}{ts:>15.3f}{sp:>20.2f}")
    elif args.cmd == "eval-once" and args.which == "logs":
        # Internal helper: run one logs evaluation and print JSON with time and peak RSS
        import numpy as np
        import resource
        from .pipeline import SqrtSpacePipeline

        n = args.n
        win = args.win
        hop = args.hop
        rng = np.random.default_rng(7)
        base = rng.lognormal(mean=2.0, sigma=0.7, size=n).astype(np.float32)
        input_blocks = max(64, int(n // hop) + 8)
        p = SqrtSpacePipeline(window=win, hop=hop, input_buffer_blocks=input_blocks, mode=args.mode)

        @p.stage("quantiles", inputs=["input"])
        def quantiles(x: np.ndarray) -> np.ndarray:
            p50 = np.percentile(x, 50, axis=1, method="nearest")
            p95 = np.percentile(x, 95, axis=1, method="nearest")
            return np.stack([p50, p95], axis=1).astype(np.float32)

        @p.stage("alerts", inputs=["quantiles"])
        def alerts(q: np.ndarray) -> np.ndarray:
            thresh = 30.0
            flag = (q[:, 1] > thresh).astype(np.int8)
            return flag[:, None]

        p.push_samples(base)
        t0 = time.time()
        out = p.evaluate("alerts")
        dt = time.time() - t0
        ru = resource.getrusage(resource.RUSAGE_SELF)
        peak_kb = getattr(ru, 'ru_maxrss', 0) or 0
        # On macOS ru_maxrss is in bytes; on Linux it's in kilobytes. Normalize to MB best-effort.
        peak_mb = peak_kb / (1024 * 1024) if peak_kb > 1e9 else peak_kb / 1024.0
        res = {
            "time_s": dt,
            "peak_rss_mb": float(peak_mb),
            "windows": int(out.shape[0]) if out.ndim else 0,
            "alerts": int(out.sum()) if out.size else 0,
        }
        print(json.dumps(res))


if __name__ == "__main__":  # pragma: no cover
    main()


