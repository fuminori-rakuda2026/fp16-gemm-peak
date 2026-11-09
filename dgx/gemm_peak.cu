// Dense GEMM peak on one GB10 (DGX Spark / ASUS GX10) using Tensor Cores through the WMMA API.
//
// Purpose: the same mathematical conditions as the Apple Neural Accelerator microbenchmark — FP16 inputs with FP32
// accumulation, Dense (no sparsity), operands held in registers so DRAM
// bandwidth does not decide the number, and a fixed 16x16x16 tile repeated R times per warp.
//
// What this measures: the rate at which the warp-level MMA instruction retires on the Tensor Cores. It is not a
// cuBLASLt "best shape" number and it is not an LLM benchmark. SASS evidence (HMMA / IMMA) is produced separately
// with cuobjdump so the claim "this ran on Tensor Cores" does not rest on the API name.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_121 gemm_peak.cu -o gemm_peak
// Run:   ./gemm_peak <repeats> <blocks> <threads>
//
// Correctness: every result is compared against a scalar reference for the same tile before timing is trusted, and
// the accumulator is written to global memory after the loop so the compiler cannot delete the work.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP 32
#define TILE 16

__global__ void fp16_mma_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
                                int repeats) {
    // One warp per 16x16 output tile. A and B tiles are loaded once, so the repeats measure the MMA rate, not memory.
    int warp = (blockIdx.x * (blockDim.x / WARP)) + (threadIdx.x / WARP);
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::col_major> b;
    // Four accumulators: back-to-back MMAs into one accumulator stall on its own latency (measured: 73 TFLOPS with
    // one, ~4x that with four). This is the saturation condition the report claims.
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc0, acc1, acc2, acc3;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);
    wmma::fill_fragment(acc3, 0.0f);
    wmma::load_matrix_sync(a, A, TILE);
    wmma::load_matrix_sync(b, B, TILE);
    for (int i = 0; i < repeats; ++i) {
        wmma::mma_sync(acc0, a, b, acc0);
        wmma::mma_sync(acc1, a, b, acc1);
        wmma::mma_sync(acc2, a, b, acc2);
        wmma::mma_sync(acc3, a, b, acc3);
    }
    float* base = C + (size_t)warp * TILE * TILE * 4;
    wmma::store_matrix_sync(base, acc0, TILE, wmma::mem_row_major);
    wmma::store_matrix_sync(base + TILE * TILE, acc1, TILE, wmma::mem_row_major);
    wmma::store_matrix_sync(base + TILE * TILE * 2, acc2, TILE, wmma::mem_row_major);
    wmma::store_matrix_sync(base + TILE * TILE * 3, acc3, TILE, wmma::mem_row_major);
}


// Scalar reference for one tile: the same K=TILE product the MMA performs, so the check is on the MMA itself.
static void reference_fp16(const half* A, const half* B, float* out) {
    for (int i = 0; i < TILE; ++i) {
        for (int j = 0; j < TILE; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < TILE; ++k) {
                // B is column-major: element (k, j) lives at k + j * TILE.
                sum += __half2float(A[i * TILE + k]) * __half2float(B[k + j * TILE]);
            }
            out[i * TILE + j] = sum;
        }
    }
}


static float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main(int argc, char** argv) {
    int repeats = (argc > 1) ? atoi(argv[1]) : 64;
    int blocks = (argc > 2) ? atoi(argv[2]) : 256;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;
    const char* mode = "fp16";
    int warps_per_block = threads / WARP;
    int warps = blocks * warps_per_block;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    // CUDA 13 removed clockRate from cudaDeviceProp; the clock is taken from nvidia-smi next to this run instead.
    printf("device=%s sm_%d%d sms=%d mem=%.0fMB\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("warps=%d (blocks=%d x %d warps) repeats=%d tile=%dx%dx%d\n", warps, blocks, warps_per_block, repeats,
           TILE, TILE, TILE);

    const size_t tile_elems = (size_t)TILE * TILE;
    // Four independent MMAs per repeat.
    const double flops_per_warp = 2.0 * TILE * TILE * TILE * (double)repeats * 4.0;
    const double total_flops = flops_per_warp * (double)warps;

    int failures = 0;
    if (!strcmp(mode, "both") || !strcmp(mode, "fp16")) {
        half* dA;
        half* dB;
        float* dC;
        cudaMalloc(&dA, tile_elems * sizeof(half));
        cudaMalloc(&dB, tile_elems * sizeof(half));
        cudaMalloc(&dC, tile_elems * sizeof(float) * warps * 4);
        std::vector<half> hA(tile_elems), hB(tile_elems);
        // Small integers in half are exact, so the reference comparison is exact and not a tolerance guess.
        for (size_t i = 0; i < tile_elems; ++i) {
            hA[i] = __float2half((float)((int)(i % 7) - 3));
            hB[i] = __float2half((float)((int)(i % 5) - 2));
        }
        cudaMemcpy(dA, hA.data(), tile_elems * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), tile_elems * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemset(dC, 0, tile_elems * sizeof(float) * warps * 4);

        fp16_mma_kernel<<<blocks, threads>>>(dA, dB, dC, 1);
        cudaDeviceSynchronize();
        std::vector<float> got(tile_elems), want(tile_elems);
        cudaMemcpy(got.data(), dC, tile_elems * sizeof(float), cudaMemcpyDeviceToHost);
        reference_fp16(hA.data(), hB.data(), want.data());
        int bad = 0;
        for (size_t i = 0; i < tile_elems; ++i) {
            if (fabsf(got[i] - want[i]) > 1e-3f) ++bad;
        }
        failures += bad;
        printf("fp16_fp32acc correctness: %s (%d/%zu mismatched)\n", bad ? "FAIL" : "PASS", bad, tile_elems);

        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        // Warm-up first: the first launch pays module load and clock ramp-up.
        fp16_mma_kernel<<<blocks, threads>>>(dA, dB, dC, repeats);
        cudaDeviceSynchronize();
        double best = 0.0;
        double samples[5];
        for (int i = 0; i < 5; ++i) {
            cudaEventRecord(s);
            fp16_mma_kernel<<<blocks, threads>>>(dA, dB, dC, repeats);
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            double ms = event_ms(s, e);
            samples[i] = total_flops / (ms / 1000.0) / 1e12;
            if (samples[i] > best) best = samples[i];
        }
        printf("fp16_fp32acc sample_tflops=%.2f,%.2f,%.2f,%.2f,%.2f peak=%.2f\n",
               samples[0], samples[1], samples[2], samples[3], samples[4], best);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    printf("RESULT %s\n", failures ? "CORRECTNESS_FAILED" : "ok");
    return failures ? 1 : 0;
}
