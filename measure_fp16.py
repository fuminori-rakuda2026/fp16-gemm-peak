#!/usr/bin/env python3
"""measure_fp16.py — FP16 dense GEMM speed on whatever GPU this machine has.

Works on any PyTorch backend: CUDA (NVIDIA, and AMD via ROCm), MPS (Apple), or CPU
for a sanity check. It times a square matrix multiply with the vendor's own library
(cuBLAS / hipBLAS / MPS) and prints one CSV block you can paste into an issue.

    python3 measure_fp16.py                     # auto-detect the device
    python3 measure_fp16.py --sizes 4096,8192   # smaller run
    python3 measure_fp16.py --json              # machine-readable

What it measures: dense FP16 GEMM, FP16 in / FP16 out, no sparsity, GPU time only
(warmup excluded), FLOPs = 2*M*N*K. Times are a mean over --iters, plus the best
single iteration. Both matter: the mean includes thermal throttling.

Not measured: memory bandwidth, quantised formats, end-to-end model speed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time

import torch


def pick_device(want: str) -> torch.device:
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    if dev.type == "mps":
        return "Apple GPU (" + platform.machine() + ")"
    return platform.processor() or "CPU"


def clock_note() -> str:
    """Best-effort clock/thermal context; empty when the platform has no query."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            return out.replace("\n", "; ")
    except Exception:
        pass
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
            return " ".join(out.split())[:120]
    except Exception:
        pass
    return ""


def bench(fn, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return sum(times) / len(times), times[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--sizes", default="4096,8192,16384")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    name = device_name(dev)
    note = clock_note()
    rows = []
    for size in [int(x) for x in args.sizes.split(",")]:
        try:
            a = torch.randn(size, size, dtype=torch.float16, device=dev)
            b = torch.randn(size, size, dtype=torch.float16, device=dev)
            mean_s, best_s = bench(lambda: torch.matmul(a, b), args.iters, args.warmup)
            flops = 2.0 * size**3
            rows.append({
                "device": name,
                "backend": dev.type,
                "M": size, "N": size, "K": size,
                "mean_tflops": round(flops / mean_s / 1e12, 2),
                "best_tflops": round(flops / best_s / 1e12, 2),
            })
            del a, b
        except Exception as exc:  # OOM on a small GPU is a result too
            rows.append({"device": name, "backend": dev.type, "M": size, "N": size, "K": size,
                         "error": f"{type(exc).__name__}: {exc}"[:120]})

    meta = {
        "device": name,
        "backend": dev.type,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "clock_or_thermal": note,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    if args.json:
        print(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1))
        return 0

    print("# " + ", ".join(f"{k}={v}" for k, v in meta.items() if v))
    print("device,backend,M,N,K,mean_tflops,best_tflops")
    for r in rows:
        if "error" in r:
            print(f"# {r['M']}^3 failed: {r['error']}")
        else:
            print(f"{r['device']},{r['backend']},{r['M']},{r['N']},{r['K']},{r['mean_tflops']},{r['best_tflops']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
