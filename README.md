# FP16 dense peak on GB10 and Apple GPUs

Two standalone harnesses that measure the **FP16 dense** rate of the matrix instruction itself,
on one GPU at a time. Operands stay in registers, so DRAM bandwidth does not decide the number,
and every result is checked against a scalar reference before timing is trusted.

| Path | Target | What it does |
|---|---|---|
| `dgx/gemm_peak.cu` | NVIDIA GB10 (CUDA, WMMA) | 16×16×16 FP16→FP32 tile repeated R times per warp |
| `m5/*.swift` | Apple M-series (Metal 4 tensor ops) | m×n×k sweep with FP16 in, FP16/FP32 out |

## Results

FP16 dense, one GPU, no sparsity. Both are peaks over a parameter sweep, not averages.

| Device | TFLOPS | Best shape | Notes |
|---|---|---|---|
| Apple M5 Ultra, 80-core GPU | **128.5** | 32×32×128 | f16 in, f16 out, Metal tensor ops |
| NVIDIA GB10 (DGX Spark), 48 SM | **105.3** | 16×16×16 tile, repeats | f16 in, f32 accumulate, SM clock 2.19 GHz |

Raw runs: [`results/fp16-peak.csv`](results/fp16-peak.csv).

## Running

CUDA side:

```bash
nvcc -O3 -std=c++17 -arch=sm_121 gemm_peak.cu -o gemm_peak
./gemm_peak <repeats> <blocks> <threads>
```

Apple side:

```bash
swiftc -O main.swift matmul.swift metal_tools.swift tensor.swift -o peak
./peak            # writes mma.csv
```

## What this does and does not measure

- It measures **the rate at which the matrix instruction retires**, with data resident in registers.
  That is the hardware ceiling for dense FP16, independent of the model, the quantisation or the engine.
- It is **not** an LLM benchmark and not a library benchmark. A tuned GEMM (cuBLAS, MPP) reaches
  roughly 77–88% of these peaks; end-to-end tokens/s is a different question again.
- Clocks matter. The GB10 numbers come from a run with the SM clock capped at 2200 MHz
  (measured 2.19 GHz during the run). Apple GPUs expose no user clock control, so the Apple figure
  depends on the thermal state; run it on an idle machine.
- Integer paths are out of scope here. This repository keeps FP16 only.
