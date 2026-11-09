# FP16 dense peak on GB10 and Apple GPUs

Two standalone harnesses that measure the **FP16 dense** rate of the matrix instruction itself,
on one GPU at a time. Operands stay in registers, so DRAM bandwidth does not decide the number,
and every result is checked against a scalar reference before timing is trusted.

**Why these numbers can be compared, what is held constant, and what is not:**
[`docs/FAIRNESS.md`](docs/FAIRNESS.md). The short version is that the *achieved* column uses one
ceiling formula for every device, while absolute TFLOPS still carry the difference in instruction shape.

| Path | Target | What it does |
|---|---|---|
| `dgx/gemm_peak.cu` | NVIDIA GB10 (CUDA, WMMA) | 16×16×16 FP16→FP32 tile repeated R times per warp |
| `m5/*.swift` | Apple M-series (Metal 4 tensor ops) | m×n×k sweep with FP16 in, FP16/FP32 out |
| `dgx/measure_fp16.py` | NVIDIA GB10 (PyTorch / cuBLAS) | full square GEMM through the library, for the "as shipped" number |

## Results

FP16 dense, one GPU, no sparsity. Both are peaks over a parameter sweep, not averages.

| Device | TFLOPS | Best shape | Notes |
|---|---|---|---|
| Apple M5 Ultra, 80-core GPU | **128.5** | 32×32×128 | f16 in, f16 out, Metal tensor ops |
| NVIDIA GB10 (DGX Spark), 48 SM | **105.3** | 16×16×16 tile, repeats | f16 in, f32 accumulate, SM clock 2.19 GHz |

### Ceiling and achieved rate

Absolute TFLOPS hide a difference in instruction shape, so each result is also given against the
ceiling of its own hardware:

```
ceiling = units x FLOP per unit per clock x clock        (dense FP16, no sparsity)
```

| Device | Units | FLOP/unit/clk | Clock | Ceiling | Measured | Achieved |
|---|---|---|---|---|---|---|
| GB10, 48 SM | 48 | 1024 | 2.19 GHz (measured) | 107.6 | 105.3 | 97.9% |
| M5 Ultra, 80-core GPU | 80 | 1024 | 1.57 GHz (estimated) | 128.6 | 128.5 | 99.9% |

The **achieved rate** is the comparable column: same definition on every vendor, and it says how much
of its own paper specification the hardware actually delivers. The absolute TFLOPS are still useful,
but read them knowing that one instruction covers 16x16x16 on GB10 and 32x32x128 on Apple.

The Apple clock is the weak link: Apple publishes neither the matrix rate nor the GPU clock, so the
ceiling above assumes 1.57 GHz from the core count. If the real clock is 1.6 GHz, the ceiling becomes
131.1 and the achieved rate 98.0%. Rows are marked `estimated` in
[`results/ceiling.csv`](results/ceiling.csv) whenever the clock was not measured.

For comparison, a library GEMM on the same GB10 (cuBLAS via PyTorch, 16384³, 50 iterations)
gives **99.4 TFLOPS**, i.e. 94% of the peak above. `dgx/measure_fp16.py` reproduces it.

Raw runs: [`results/fp16-peak.csv`](results/fp16-peak.csv).

## Run it on your GPU

`measure_fp16.py` uses whatever PyTorch backend the machine has, so the same command works on
NVIDIA (cuBLAS), AMD (ROCm/hipBLAS), Apple (MPS) and CPU. That makes it the one number that can be
compared across vendors without arguing about instruction shapes.

```bash
pip install torch          # once
python3 measure_fp16.py --sizes 4096,8192,16384
```

It prints a CSV block with `mean_tflops` (includes thermal throttling) and `best_tflops`.
Post the block in a [GPU result issue](../../issues/new?template=gpu-result.md) — or open a PR that
appends it to `results/community.csv`. Include the `#` line; it carries the torch version and the
clock/thermal reading.

Notes: 16384 needs about 1 GB per matrix in FP16. On a small card use `--sizes 4096,8192`, and say
so in the issue.

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

Library GEMM (CUDA image with PyTorch):

```bash
python3 dgx/measure_fp16.py --iters 50 --warmup 15 --sizes 4096,8192,16384
```

## What this does and does not measure

- It measures **the rate at which the matrix instruction retires**, with data resident in registers.
  That is the hardware ceiling for dense FP16, independent of the model, the quantisation or the engine.
- It is **not** an LLM benchmark and not a library benchmark. A tuned GEMM (cuBLAS, MPP) reaches
  roughly 77–88% of these peaks; end-to-end tokens/s is a different question again.
- Clocks matter. The GB10 numbers come from a run with the SM clock capped at 2200 MHz
  (measured 2.19 GHz during the run). Apple GPUs expose no user clock control, so the Apple figure
  depends on the thermal state; run it on an idle machine.
- Integer and 4-bit paths are out of scope here. FP4 is not one format but at least three
  (NVFP4, MXFP4, unscaled E2M1), each with a different scale layout, so a shared TFLOPS column would
  compare different arithmetic. [`docs/FAIRNESS.md` §8](docs/FAIRNESS.md) spells this out.
