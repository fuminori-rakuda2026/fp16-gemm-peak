# FP16 dense peak on GB10 and Apple GPUs

> **A tap turned fully open, with the hose, the nozzle and the bucket taken off.** What comes out is
> the water source's own maximum — not what a garden hose delivers at the far end. This repository
> does that to a GPU's matrix instruction, and reports both the litres per minute and the percentage
> of each tap's rated capacity.

**Purpose.** Measure how fast a GPU's matrix instruction can retire, on one GPU at a time, so that a
number can be read as *hardware capability* rather than as the quality of a model, an engine or a
quantisation recipe.

**Method, in one paragraph.** Take the same mathematical operation on every device — dense FP16
matrix multiply, FP32 accumulate, no sparsity — keep the operands in registers so DRAM never decides
the result, repeat it on the GPU, check the output against a CPU reference, and only then divide
FLOPs by time. Two devices rarely have the same tap (one instruction covers 16×16×16, another
32×32×128), which is why results are also given as a percentage of each device's own rated ceiling.

**Why this shape.** A library GEMM number mixes the silicon with the library, and a tokens-per-second
number mixes it with the model. Both are useful and both are measured elsewhere; this repository
isolates the layer underneath so that "the hardware cannot go faster" and "the software is leaving
performance on the table" can be told apart.

## Results

| Device | Output / accumulator | TFLOPS | Best shape |
|---|---|---|---|
| NVIDIA GB10 (DGX Spark), 48 SM | FP32 accumulate | **105.3** | 16×16×16 tile, repeats |
| Apple M5 Ultra, 80-core GPU | FP32 accumulate | **125.7** | 16×32×128 |
| Apple M5 Ultra, 80-core GPU | f16 output | **128.5** | 32×32×128 |

Compare like with like: the two FP32-accumulate rows are the comparable pair (125.7 vs 105.3, the
Apple part about 19% higher). The 128.5 row uses an f16 output path, and it is not established that
its accumulation is FP32, so it is listed separately rather than mixed in.

Against each device's own ceiling (`units × FLOP per unit per clock × clock`):

| Device | Output | Ceiling | Measured | Achieved | Clock |
|---|---|---|---|---|---|
| GB10, 48 SM | FP32 acc | 107.6 | 105.3 | 97.8% | 2.19 GHz, measured during the run |
| M5 Ultra, 80-core | FP32 acc | 132.5 | 125.7 | 94.8% | 1.618 GHz, measured under load |
| M5 Ultra, 80-core | f16 out | 132.5 | 128.5 | 97.0% | 1.618 GHz, measured under load |

Both clocks are read off the running device, not from a data sheet: clock reporting through the driver
on the GB10, and `powermetrics --samplers gpu_power` on the Apple part (1617–1619 MHz, stable under
load). The one number still taken from documentation rather than measured is 1024 FLOPs per core per
clock, on both rows — so read the achieved column as "consistent with the published rate", not as a
proof of it. Raw runs: [`results/fp16-peak.csv`](results/fp16-peak.csv),
[`results/ceiling.csv`](results/ceiling.csv).

A normal library call on the same GB10 (cuBLAS via PyTorch, 16384³, 50 iterations) gives **99.4
TFLOPS**, 94% of the peak above — the toolchain leaves a little on the table, which is the kind of
question the peak exists to answer.

## Run it yourself

Run it on your own GPU:

```bash
pip install torch
python3 measure_fp16.py --sizes 4096,8192,16384
```

It picks CUDA, ROCm, MPS or CPU automatically, prints a CSV block with mean and best TFLOPS, and that
block is enough for a [result issue](../../issues/new?template=gpu-result.md). Add a row to
[`results/ceiling.csv`](results/ceiling.csv) when you know your device's units and clock, and mark
`estimated` anything you did not measure.

Peak harnesses in this repository:

| Path | Target |
|---|---|
| `dgx/gemm_peak.cu` | GB10, CUDA WMMA, FP16 in / FP32 accumulate |
| `m5/*.swift` | Apple, Metal 4 tensor ops, FP16 in / FP16 or FP32 out |
| `dgx/measure_fp16.py` | GB10, cuBLAS full GEMM, for the "as shipped" number |

```bash
nvcc -O3 -std=c++17 -arch=sm_121 dgx/gemm_peak.cu -o gemm_peak && ./gemm_peak <repeats> <blocks> <threads>
swiftc -O m5/main.swift m5/matmul.swift m5/metal_tools.swift m5/tensor.swift -o peak && ./peak
python3 dgx/measure_fp16.py --iters 50 --warmup 15 --sizes 4096,8192,16384
```

## What this is not

- Not an LLM benchmark, and not a library benchmark. A tuned GEMM reaches roughly 78–94% of these
  peaks; tokens per second is a different question again.
- No power or energy numbers.
- No 4-bit or integer paths. FP4 alone means at least three different formats (NVFP4, MXFP4, unscaled
  E2M1), each with its own scale layout, so a shared TFLOPS column would compare different arithmetic.
  [`docs/FAIRNESS.md` §8](docs/FAIRNESS.md) spells this out.
- The CUDA peak checks correctness at `repeats=1` (one tile), not after the long repetition loop.

**What is held constant, what is not, and the checklist for adding a GPU:**
[`docs/FAIRNESS.md`](docs/FAIRNESS.md).
