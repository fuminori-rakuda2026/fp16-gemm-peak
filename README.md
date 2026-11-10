# FP16 dense peak on GB10 and Apple GPUs

**Purpose.** Measure the maximum throughput of a GPU's matrix-compute units, one GPU at a time,
independently of model architecture, inference engine and quantisation recipe.

Think of it as measuring a tap's maximum flow with the hose taken off: what is measured is the
source's own limit, not what the plumbing delivers at the far end.

**Method.** All tests use dense FP16 matrix operations without sparsity; accumulation and output
precision are identified separately for each result. The operands are kept in registers so DRAM never
decides the result, the same matrix multiply is repeated on the GPU, the output is checked against a
CPU reference, and only then are FLOPs divided by time. Two devices rarely have the same tap — one
instruction covers 16×16×16, another 32×32×128 — which is why results are also given as a percentage
of each device's own rated ceiling.

**Why measure the peak at all.** A library GEMM number mixes the silicon with the library, and a
tokens-per-second number mixes it with the model. This repository isolates the layer underneath, so
that the hardware's measured compute limit and the software's distance from it can be told apart.

## Results

Measured 2026-11-10.

| Device | Output / accumulator | TFLOPS | Best shape |
|---|---|---|---|
| NVIDIA GB10 (DGX Spark class), 48 SM | FP32 accumulate | **107.7** | 16×16×16 tile, repeats |
| Apple M5 Ultra, 80-core GPU | FP32 accumulate | **125.3** | 16×32×128 |
| Apple M5 Ultra, 80-core GPU | f16 output | **128.2** | 32×32×128 |
| Apple M5 Ultra, 64-core GPU | FP32 accumulate | **100.5** | 16×32×128 |
| Apple M5 Ultra, 64-core GPU | f16 output | **102.8** | 32×32×128 |

These are medians, not the fastest sample. The GB10 ran 25 samples in five separate processes.
Each M5 ran 750, because the Metal harness prints 30 timings per run; the 25-sample rule is a floor,
not a target. The earlier table used the best sample, so the GB10 moved up (105.3 to 107.7) and the
M5 rows moved down slightly.

The comparable pair is the two FP32-accumulate rows: 125.3 against 107.7, so the M5 Ultra is about
16% higher. The f16-output rows stay separate because it is not established that their accumulation
is FP32.

Against each device's own nominal ceiling:

| Device | Output | Ceiling | Measured | Achieved | Clock |
|---|---|---|---|---|---|
| GB10, 48 SM | FP32 acc | 107.6 | 107.7 | 100.1% | 2.190 GHz, measured |
| M5 Ultra, 80-core | FP32 acc | 132.5 | 125.3 | 94.6% | 1.618 GHz, measured |
| M5 Ultra, 80-core | f16 out | 132.5 | 128.2 | 96.8% | 1.618 GHz, measured |
| M5 Ultra, 64-core | FP32 acc | 106.0 | 100.5 | 94.8% (est.) | 1.618 GHz, from the 80-core |
| M5 Ultra, 64-core | f16 out | 106.0 | 102.8 | 97.0% (est.) | 1.618 GHz, from the 80-core |

The GB10 reads 100.1%, a hair above its nominal ceiling. The small excess is consistent with
uncertainty in the ceiling itself: 1024 FLOPs per SM per clock is documentation, and the clock is
sampled every 250 ms rather than per sample. Read it as "at the estimate", not as proof that the
hardware beats its rating. The two 64-core achieved figures are estimates because their clock was
borrowed from the 80-core unit.

Both GPU clocks were measured under load, not taken from a data sheet: clock reporting through the
driver on the GB10, and `powermetrics --samplers gpu_power` on the M5 Ultra (1617–1619 MHz). The one
number still taken from documentation is 1024 FLOPs per core per clock, so read the achieved column
as "consistent with the published rate", not as a proof of it. Raw runs:
[`results/fp16-peak.csv`](results/fp16-peak.csv), [`results/ceiling.csv`](results/ceiling.csv),
[`results/median-20261110.csv`](results/median-20261110.csv),
[`results/samples-20261110.csv`](results/samples-20261110.csv).

### Community results, not independently reproduced

[Issue #1](https://github.com/fuminori-rakuda2026/fp16-gemm-peak/issues/1) reports three NVIDIA GPUs,
measured by the reporter with the unmodified `dgx/gemm_peak.cu` (commit `2fc58ca`, `-arch=sm_120`,
CUDA 13.0).

| GPU | SMs | Median TFLOPS | Best |
|---|---:|---:|---:|
| RTX PRO 5000 72GB Blackwell | 110 | 291.45 | 291.53 |
| RTX PRO 4500 Blackwell | 82 | 238.74 | 238.91 |
| GeForce RTX 5070 Ti | 70 | 107.47 | 107.48 |

Each row is the median of 25 samples from five fresh processes, with the correctness check passing.
These are the reporter's statistics, on the reporter's machines. The maintainer has not run them, so
they stay out of the table above.

The 5070 Ti is the interesting one. NVIDIA rates it at 87.9 TFLOPS for FP16 input with FP32
accumulation at 2.452 GHz. Scaled to the 3.0 GHz seen during the run, that is about 107.5 TFLOPS,
which matches the measurement. The per-SM rate on that card is therefore 512 FLOPs per clock, not
1024; see [`docs/FAIRNESS.md` §4](docs/FAIRNESS.md).

A normal library call on the same GB10 (cuBLAS via PyTorch, 16384³, 50 iterations) gives **99.4
TFLOPS**, approximately 94% of the measured peak — the kind of gap the peak exists to expose.

## Run it yourself

**The portable path — any GPU PyTorch supports.** This is the part that runs on hardware other than
the two machines in the table.

```bash
python3 -m venv .venv && .venv/bin/pip install torch   # once
.venv/bin/python measure_fp16.py --sizes 4096,8192,16384
```

It picks CUDA, ROCm, MPS or CPU automatically, prints a CSV block with mean and best TFLOPS, and that
block is enough for a [result issue](../../issues/new?template=gpu-result.md). Verify it on a machine
we have here (Apple M2 Ultra, MPS): 22.8 TFLOPS at 4096³, 11 seconds for all three sizes.

**The peak harnesses are per-device, not portable.** They are short on purpose, but each one encodes
one architecture's instruction and must be adapted before it will build or mean anything:

| Path | Target | What you change for another device |
|---|---|---|
| `dgx/gemm_peak.cu` | GB10, CUDA WMMA, FP16 in / FP32 accumulate | `-arch=sm_121` → your architecture; tile and repeats are arguments |
| `m5/*.swift` | Apple, Metal 4 tensor ops, FP16 in / FP16 or FP32 out | needs Metal 4 tensor ops (M5-class GPU, recent macOS); other Apple GPUs use simdgroup |
| `dgx/measure_fp16.py` | GB10, cuBLAS full GEMM, for the "as shipped" number | any CUDA GPU; it is the same PyTorch call as the portable script |

```bash
nvcc -O3 -std=c++17 -arch=sm_121 dgx/gemm_peak.cu -o gemm_peak && ./gemm_peak <repeats> <blocks> <threads>
swiftc -O m5/main.swift m5/matmul.swift m5/metal_tools.swift m5/tensor.swift -o peak && ./peak
# For the median protocol, run one shape in five separate processes.
# FP32 output is fastest at 16,32,128; f16 output at 32,32,128.
for i in 1 2 3 4 5; do M5_ONLY=16,32,128 M5_RUNS=5 ./peak >> m5-runs.txt; done
python3 dgx/measure_fp16.py --iters 50 --warmup 15 --sizes 4096,8192,16384
```

If it fails on your machine, that is information: open an issue with the device, the command and the
error. A peak harness that does not build is a gap in this repository, not a mistake on your side.

## What this is not

- Not an LLM benchmark, and not a library benchmark. A tuned GEMM reaches roughly 78–94% of these
  peaks; tokens per second is a different question again.
- No power or energy numbers.
- No 4-bit or integer paths. FP4 alone means at least three different formats (NVFP4, MXFP4, unscaled
  E2M1), each with its own scale layout, so a shared TFLOPS column would compare different arithmetic.
  [`docs/FAIRNESS.md` §8](docs/FAIRNESS.md) spells this out.
- The CUDA peak checks correctness at `repeats=1` (one tile), not after the long repetition loop.
- `measure_fp16.py` reports the library's own GEMM, so it is not directly comparable with the peak
  rows; it is the "what an application actually gets" number.

**What is held constant, what is not, and the checklist for adding a GPU:**
[`docs/FAIRNESS.md`](docs/FAIRNESS.md).

## Scope and upkeep

This is a personal measurement that got cleaned up enough to publish, not a product. It is published so
that the numbers can be checked and repeated, not because there is a roadmap behind it.

- Issues are read as data: a result on another GPU, or a build failure with the device and the command.
- There is no promise of a fix, a reply time, or a new feature. A pull request is welcome; silence is the
  default and means only that.
- The numbers will age. Toolchains, drivers and clocks move; a row without a date is a row to trust less.
