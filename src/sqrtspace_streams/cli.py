from __future__ import annotations

import argparse
from rich.console import Console


def main() -> None:
    parser = argparse.ArgumentParser(prog="sqrtspace", description="SqrtSpace Streams CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a demo")
    demo.add_argument("which", choices=["audio"], help="Demo to run")
    demo.add_argument("--mode", choices=["auto", "classic", "sqrt"], default="auto")

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


if __name__ == "__main__":  # pragma: no cover
    main()


