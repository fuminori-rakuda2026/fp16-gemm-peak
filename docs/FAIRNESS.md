# Why these numbers can be compared

This repository measures the tap, not the plumbing. The definitions and the measurement conditions
follow; the analogy itself is in the [README](../README.md).

| Faucet | Here |
|---|---|
| Tap fully open | The matrix instruction issued back to back |
| Hose removed | Operands in registers, no DRAM traffic, no library overhead |
| Litres per minute | TFLOPS, FLOPs = 2·M·N·K |
| The tap's rated capacity | units × FLOP per unit per clock × clock |
| Percentage of rated capacity | The achieved column |

A GPU comparison is only worth reading if you know what was held constant and what was not.
This page states both, then gives the rule for adding another GPU.

## 1. What is being compared

Two different quantities, never mixed in one column:

| Column | Question it answers | How it is measured |
|---|---|---|
| **Peak** | How much throughput can this GPU's matrix units reach? | Small tile, operands in registers, repeated, timed on the GPU |
| **Achieved** | How much of its own specification does the hardware actually deliver? | Measured peak ÷ ceiling, both from the same definition |
| **Library GEMM** | What do you get from the normal toolchain? | `measure_fp16.py` (cuBLAS / hipBLAS / MPS), same script everywhere |

A peak and a library number answer different questions. Putting them side by side without labels is
the most common way a comparison goes wrong, so the tables here keep them apart.

## 2. What is held constant on every device

1. **The operation**: dense matrix multiply. Square shapes for the library path, one tile repeated
   for the peak path. No convolutions, no attention, no model.
2. **The precision**: FP16 inputs, and the accumulator stated for every row. FP32 accumulate and f16
   output are different measurements and are never averaged into one number. FP8, FP4 and INT8 are out of scope here
   precisely because their scale formats are vendor-specific and are not the same thing.
3. **No sparsity.** Vendor headline numbers are often 2:4 sparse; those are double the dense rate.
4. **No DRAM in the loop** for the peak path: operands stay in registers, so bandwidth does not
   decide the result.
5. **Correctness before speed**: every peak run checks its output against a scalar or CPU reference
   and reports the error. A number is not reported when the check fails.
6. **Same arithmetic for the score**: FLOPs = 2·M·N·K, no per-vendor adjustments.
7. **Warmup excluded, same statistics**: warmup iterations are dropped, then mean and best are both
   reported. The mean includes thermal throttling; the best is the burst rate.

## 3. What necessarily differs, and why that is acceptable

| Difference | Why it is unavoidable | Which direction it pushes |
|---|---|---|
| Instruction shape (16×16×16 on GB10 vs 32×32×128 on Apple) | Each vendor defines its own matrix instruction; there is no common ISA | A larger tile amortises issue overhead better, so it can flatter the larger-tile device |
| API (CUDA WMMA vs Metal tensor ops) | Vendor-specific by definition | Newer APIs are less battle-tested |
| Clock control | Apple exposes none; NVIDIA and AMD do | A device whose clock can be pinned looks more stable |
| Library maturity (cuBLAS vs MPS vs hipBLAS) | Out of the hardware's control | Shown separately in the library column |

None of these change the arithmetic of the **achieved** column, because that column divides by a
ceiling computed with the same formula for every device. That is the point of normalising.

## 4. The ceiling, and why it makes the comparison fair

```
ceiling_tflops  = units × FLOP per unit per clock × clock_ghz / 1000
achieved_pct    = measured_peak_tflops ÷ ceiling_tflops × 100
```

The formula is identical for every vendor; only the three inputs change. Two consequences:

- **Absolute TFLOPS** carry the instruction-shape difference and should be read as "what this
  hardware does", not "which hardware is better".
- **Achieved %** uses one normalisation formula on every vendor: it says how close the silicon comes to
  its own paper
  number. 98% on one device and 74% on another is a meaningful statement about each device.

**Same normalization formula.** Both GPU clocks were measured under load, so the two achieved rates
rest on measurements of the same kind. What is still documentation rather than measurement is the
FLOPs-per-core-per-clock figure, and it is the same number on both rows; if it is wrong, it is wrong
for both.

Every row in [`results/ceiling.csv`](../results/ceiling.csv) carries the clock value **and where it
came from**. A row whose clock was not measured is marked `estimated`, and its achieved rate is
correspondingly weaker evidence. The Apple row is currently of that kind: Apple publishes neither
the matrix rate nor the GPU clock.

## 5. What would make a comparison unfair

Use this as a checklist before adding a row:

- [ ] Sparse and dense results in the same column
- [ ] FP4/FP8/INT8 mixed with FP16 (different scale formats, different arithmetic)
- [ ] A peak measured at 4096³ compared with a library number at 128³
- [ ] Best-of-N on one device and mean on another
- [ ] A ceiling taken from a vendor AI-TOPS figure that assumes sparsity
- [ ] A clock that was neither measured nor labelled estimated
- [ ] Thermal throttling hidden by reporting only the burst value
- [ ] A result reported before its correctness check passed

## 6. Adding your own GPU

1. `python3 measure_fp16.py --sizes 4096,8192,16384` — one command, any backend.
2. Add a row to [`results/ceiling.csv`](../results/ceiling.csv): units, FLOP per unit per clock,
   clock and its source, then your measured peak and the achieved rate.
3. If you also have a peak harness for your vendor, say which instruction shape it used.
4. Open an issue with the block, or a PR that appends the row. Mark `estimated` anything you did not
   measure directly.

## 7. Known weaknesses of the current data

- The Apple clock is measured, not assumed: `powermetrics --samplers gpu_power` reports 1617–1619 MHz
  while the harness runs, so the ceiling is 132.5 and the achieved rates are 94.8% (FP32 accumulate)
  and 97.0% (f16 output). An earlier revision of this page used an assumed 1.57 GHz; the measurement
  moved that device's ceiling up by 3% and its achieved rate down.
- The per-clock rate of 1024 FLOPs per core is still documentation, not measurement. It is the same
  assumption on both devices, but it is an assumption.
- It is **not established** that the Apple f16-output path accumulates in FP32. That is why the
  comparable pair is the two FP32-accumulate rows (125.7 vs 105.3), and the f16-output row is reported
  on its own.
- The CUDA peak checks correctness at `repeats=1` (one tile), not after the long repetition loop.
- Only two devices have a measured peak. The other rows in the README table are derived values,
  labelled as such.
- No power or energy numbers. A rate per watt would change the ranking, and it is not measured here.
- The library column shows the toolchain, not the silicon: MPS and cuBLAS are not equally mature.

## 8. Formats this repository deliberately leaves out

The word FP4 covers at least three different numerical systems, and they are not interchangeable:

| Format | Element | Scale | Where it exists |
|---|---|---|---|
| NVFP4 | E2M1 | per-16 elements, E4M3 (plus a global scale in practice) | NVIDIA Blackwell |
| MXFP4 | E2M1 | per-32 elements, E8M0 (power of two) | OCP standard, several vendors |
| "FP4" unscaled | E2M1 | none | some Apple MPP paths |

A TFLOPS figure means different arithmetic depending on which row of that table produced it, so putting
them in one column would be the exact mistake this document is meant to prevent.

Two further problems make a cross-vendor FP4 table dishonest rather than merely imprecise:

1. **Scale handling dominates the result.** A block scale is a lookup and a multiply per block; whether
   the kernel reads it as E4M3 or E8M0, and how often, changes the rate. Our own measurement of an
   unscaled E2M1 path came out *below* FP16 on the same device, while a block-scaled path on the other
   device came out several times above it. That contrast is a statement about two different paths, not
   about two pieces of silicon.
2. **Many GPUs have no FP4 instruction at all.** Ada and Ampere generations, and the Intel Arc Pro
   series in this table, expose no FP4 path; their 4-bit work is done by dequantising in software.
   A blank cell there is information, but it is not a rate.

If you want a 4-bit comparison, state all of: element format, scale format and block size, whether a
global scale was applied, and the kernel that ran. Without those four, the number does not mean
anything comparable. This is why the repository reports FP16 only, and why the FP16 column carries a
measured ceiling for each device.
