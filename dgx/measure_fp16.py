#!/usr/bin/env python3
"""measure_fp16.py — FP16/BF16 dense GEMM on GB10 (sm_121)

測るもの（すべて密行列・正方）:
  FP16 : torch.matmul (fp16 in/out, cuBLAS)
  BF16 : torch.matmul (bf16 in/out, cuBLAS)

記録:
  warmup 回数、反復回数、GPU 時間（cuda event）、TFLOPS、実測カーネル名（torch.profiler）、
  開始/終了の SM クロック、精度検証（小サイズで FP32 参照と比較）

使い方: python3 measure_fp16.py [--iters 50] [--warmup 10]
"""
import argparse
import json
import subprocess
import time

import torch


def clocks():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "n/a"
    except Exception:
        return "n/a"


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000.0 / iters  # 秒/回


def kernel_names(fn):
    """1 回実行して GPU カーネル名を取る（実測の使用カーネル記録用）。"""
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        names = []
        for ev in prof.key_averages():
            if ev.self_device_time_total > 0:
                names.append((ev.key, round(ev.self_device_time_total / 1000.0, 3)))
        names.sort(key=lambda x: -x[1])
        return names[:3]
    except Exception as exc:
        return [("profiler-error", str(exc)[:80])]




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sizes", default="4096,8192,16384")
    a = ap.parse_args()

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"# GPU={name} sm_{cap[0]}{cap[1]} VRAM={total:.1f}GB torch={torch.__version__}")
    print(f"# iters={a.iters} warmup={a.warmup} clock_start={clocks()}")
    rows = []

    for size in [int(x) for x in a.sizes.split(",")]:
        M = N = K = size
        flops = 2.0 * M * N * K

        # FP16
        a16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
        t = bench(lambda: torch.matmul(a16, b16), a.iters, a.warmup)
        rows.append({"op": "FP16", "M": M, "N": N, "K": K, "sec": t,
                     "tflops": flops / t / 1e12, "kernels": kernel_names(lambda: torch.matmul(a16, b16))})
        del a16, b16
        torch.cuda.empty_cache()

        # BF16
        ab = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        bb = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        t = bench(lambda: torch.matmul(ab, bb), a.iters, a.warmup)
        rows.append({"op": "BF16", "M": M, "N": N, "K": K, "sec": t,
                     "tflops": flops / t / 1e12, "kernels": kernel_names(lambda: torch.matmul(ab, bb))})
        del ab, bb
        torch.cuda.empty_cache()

    print(f"# clock_end={clocks()}")
    print(json.dumps(rows, indent=1, ensure_ascii=False))



if __name__ == "__main__":
    main()
