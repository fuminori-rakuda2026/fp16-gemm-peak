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

| Device | Output / accumulator | TFLOPS | Best shape |
|---|---|---|---|
| NVIDIA GB10 (DGX Spark class), 48 SM | FP32 accumulate | **107.7** | 16×16×16 tile, repeats |
| Apple M5 Ultra, 80-core GPU | FP32 accumulate | **125.3** | 16×32×128 |
| Apple M5 Ultra, 80-core GPU | f16 output | **128.2** | 32×32×128 |
| Apple M5 Ultra, 64-core GPU | FP32 accumulate | **100.5** | 16×32×128 |

All four rows are medians measured on 2026-11-10: 25 samples on the GB10 (5 fresh processes ×
5 samples) and 750 samples on each M5 Ultra (5 fresh processes × 5 repeats × 30 timing samples; the
Metal harness prints one number per timing sample, so 25 is a floor rather than a target). The
earlier table listed the best sample; the medians are 2.4 TFLOPS higher on the GB10 and 0.3–0.4 lower
on the M5 Ultra, which is why the numbers moved.

Compare like with like: the FP32-accumulate rows are the comparable pair (125.3 vs 107.7, so the
M5 Ultra is about 16% higher). The f16-output rows use an f16 output path, and it is not established
that their accumulation is FP32, so they are listed separately rather than mixed in.

Against each device's own ceiling (`units × FLOP per unit per clock × clock`):

| Device | Output | Ceiling | Measured | Achieved | Clock |
|---|---|---|---|---|---|
| GB10, 48 SM | FP32 acc | 107.6 | 107.7 | 100.1% | 2.190 GHz, measured during the run |
| M5 Ultra, 80-core | FP32 acc | 132.5 | 125.3 | 94.6% | 1.618 GHz, measured under load |
| M5 Ultra, 80-core | f16 out | 132.5 | 128.2 | 96.8% | 1.618 GHz, measured under load |
| M5 Ultra, 64-core | FP32 acc | 106.0 | 100.5 | 94.8% | 1.618 GHz, measured under load |
| M5 Ultra, 64-core | f16 out | 106.0 | 102.8 | 97.0% | 1.618 GHz, measured under load |

The GB10 row sits 0.09% above its nominal ceiling. This is not display rounding: the excess reflects
uncertainty in the ceiling itself (a 1024 FLOPs/SM/clock figure taken from documentation, and clock
sampled at 250 ms rather than per sample), so read it as "at the nominal estimate", not as proof that
the hardware exceeds its rating.

Both GPU clocks were measured under load, not taken from a data sheet: clock reporting through the
driver on the GB10, and `powermetrics --samplers gpu_power` on the M5 Ultra (1617–1619 MHz). The one number still taken from documentation rather than measured is 1024 FLOPs per core per
clock, on the GB10 and M5 rows — so read the achieved column as "consistent with the published rate",
not as a proof of it. Raw runs: [`results/fp16-peak.csv`](results/fp16-peak.csv),
[`results/ceiling.csv`](results/ceiling.csv).

### Community-reported results — not independently reproduced by the maintainer

These come from [Issue #1](https://github.com/fuminori-rakuda2026/fp16-gemm-peak/issues/1), measured
by the reporter on their own machines with the unmodified `dgx/gemm_peak.cu` (commit `2fc58ca`,
`-arch=sm_120`, CUDA 13.0). They are medians of 25 samples from five fresh processes, with the
correctness check passing. They are listed apart from the table above because the maintainer has not
run them, and because the statistical protocol is the reporter's own: treat them as a careful report,
not as an independent reproduction or as a like-for-like run of the same lab procedure.

| GPU | SMs | Median TFLOPS | Mean | Best | Min | Std | Clock context |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 5000 72GB Blackwell | 110 | **291.45** | 291.45 | 291.53 | 291.41 | 0.030 | 2587 MHz median |
| RTX PRO 4500 Blackwell | 82 | **238.74** | 238.69 | 238.91 | 238.00 | 0.180 | 2827 MHz median |
| GeForce RTX 5070 Ti | 70 | **107.47** | 107.47 | 107.48 | 107.46 | 0.006 | 3000 MHz median |

The clock figures are context sampled during the runs, not per-sample clocks. Raw rows:
[`results/community-peak.csv`](results/community-peak.csv).

The 5070 Ti is the interesting row: 107.5 TFLOPS is about 100% of NVIDIA's published 87.9 TFLOPS
FP32-accumulate figure once its 2.452 GHz rating is scaled to the 3.0 GHz observed here. That says the
FP32-accumulate rate per SM per clock is **not** 1024 on that part; see
[`docs/FAIRNESS.md` §4](docs/FAIRNESS.md).

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
# one shape, five repeats, for the 25-sample protocol:
M5_ONLY=32,32,128 M5_RUNS=5 ./peak
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
