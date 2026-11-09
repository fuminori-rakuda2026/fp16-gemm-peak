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
| NVIDIA GB10 (DGX Spark), 48 SM | FP32 accumulate | **105.3** | 16×16×16 tile, repeats |
| Apple M5 Ultra, 80-core GPU | FP32 accumulate | **125.7** | 16×32×128 |
| Apple M5 Ultra, 80-core GPU | f16 output | **128.5** | 32×32×128 |

Compare like with like: the two FP32-accumulate rows are the comparable pair (125.7 vs 105.3, so the
M5 Ultra is about 19% higher). The 128.5 row uses an f16 output path, and it is not established that
its accumulation is FP32, so it is listed separately rather than mixed in.

Against each device's own ceiling (`units × FLOP per unit per clock × clock`):

| Device | Output | Ceiling | Measured | Achieved | Clock |
|---|---|---|---|---|---|
| GB10, 48 SM | FP32 acc | 107.6 | 105.3 | 97.8% | 2.19 GHz, measured during the run |
| M5 Ultra, 80-core | FP32 acc | 132.5 | 125.7 | 94.8% | 1.618 GHz, measured under load |
| M5 Ultra, 80-core | f16 out | 132.5 | 128.5 | 97.0% | 1.618 GHz, measured under load |

Both GPU clocks were measured under load, not taken from a data sheet: clock reporting through the
driver on the GB10, and `powermetrics --samplers gpu_power` on the M5 Ultra (1617–1619 MHz). The one number still taken from documentation rather than measured is 1024 FLOPs per core per
clock, on both rows — so read the achieved column as "consistent with the published rate", not as a
proof of it. Raw runs: [`results/fp16-peak.csv`](results/fp16-peak.csv),
[`results/ceiling.csv`](results/ceiling.csv).

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
