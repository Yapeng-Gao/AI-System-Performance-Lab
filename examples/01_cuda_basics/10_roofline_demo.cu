/**
 * [Module A] 10. Roofline probes: HBM copy bandwidth + FP32 FMA compute
 *
 * A: float4 copy (read+write) on a large buffer → measured GB/s
 * B: register FMA probe → measured FP32 TFLOPS
 * C: print arithmetic intensity and position vs ridge from *measured* roofs
 *
 * Metric: CUDA event median (warmup + runs). Not Host chrono.
 * NCU Speed-of-Light / Roofline chart: optional (10_profile_roofline.sh).
 * Prescriptions: B-01 / B-10. Sanitizer: A-09.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace {

constexpr int kWarmup = 2;
constexpr int kRuns = 7;
constexpr int kBlock = 256;
constexpr int kComputeIters = 1000;

float median_f(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n == 0) {
        return 0.0f;
    }
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

__global__ void bandwidth_kernel(const float4* __restrict__ in,
                                 float4* __restrict__ out,
                                 size_t n_vectors) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = tid; i < n_vectors; i += stride) {
        out[i] = in[i];
    }
}

__global__ void compute_fma_kernel(float* data, int n, int iters) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }

    float r0 = data[idx];
    float r1 = r0 * 0.5f;
    float r2 = r0 * 0.2f;
    float r3 = r0 * 0.8f;
    const float alpha = 1.00001f;
    const float beta = 0.00001f;

#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        r0 = __fmaf_rn(r0, alpha, beta);
        r1 = __fmaf_rn(r1, alpha, beta);
        r2 = __fmaf_rn(r2, alpha, beta);
        r3 = __fmaf_rn(r3, alpha, beta);
    }
    data[idx] = r0 + r1 + r2 + r3;
}

template <typename Fn>
float time_median_ms(Fn&& once, int warmup, int runs) {
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        once();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        once();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return median_f(samples);
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("sm_%d%d  SMs=%d  memoryBusWidth=%d-bit\n", prop.major, prop.minor,
           prop.multiProcessorCount, prop.memoryBusWidth);
    printf("Metric: CUDA event median (warmup=%d, runs=%d)\n", kWarmup, kRuns);
    printf("Note: CUDA 12+ has no prop clockRate; theory peaks below are NOT used.\n");
    printf("      Ridge uses *measured* BW and TFLOPS roofs.\n\n");

    // Large enough to miss L2 residency for HBM-oriented copy (see B-04).
    constexpr size_t N = 16ull * 1024ull * 1024ull;  // float elements
    constexpr size_t bytes = N * sizeof(float);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 0, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    const size_t n_vec = N / 4;
    const int bw_grid =
        static_cast<int>((n_vec + static_cast<size_t>(kBlock) - 1) / kBlock);

    const float bw_ms = time_median_ms(
        [&]() {
            bandwidth_kernel<<<bw_grid, kBlock>>>(
                reinterpret_cast<const float4*>(d_in),
                reinterpret_cast<float4*>(d_out), n_vec);
        },
        kWarmup, kRuns);

    // Read + write payload.
    const double bw_gbs = (static_cast<double>(bytes) * 2.0) /
                          (static_cast<double>(bw_ms) * 1e-3) / 1e9;

    printf("[A] Bandwidth probe (float4 copy, read+write)\n");
    printf("    payload=%.2f MiB  median=%.3f ms\n",
           bytes / (1024.0 * 1024.0), bw_ms);
    printf("    achieved_bw=%.2f GB/s\n", bw_gbs);
    printf("    AI_copy≈0 FLOP/byte (pure memcpy)\n\n");

    const int compute_grid = prop.multiProcessorCount * 4;
    const int active_threads = compute_grid * kBlock;

    const float compute_ms = time_median_ms(
        [&]() {
            compute_fma_kernel<<<compute_grid, kBlock>>>(d_in, active_threads,
                                                         kComputeIters);
        },
        kWarmup, kRuns);

    // 4 independent FMAs per iter; each FMA = 2 FLOPs.
    const double total_flops =
        static_cast<double>(active_threads) * kComputeIters * 4.0 * 2.0;
    const double tflops =
        total_flops / (static_cast<double>(compute_ms) * 1e-3) / 1e12;

    // Per thread: read 4 B + write 4 B; FLOPs = iters * 4 * 2.
    const double ai_compute =
        (static_cast<double>(kComputeIters) * 4.0 * 2.0) / 8.0;

    printf("[B] Compute probe (register FMA, iters=%d)\n", kComputeIters);
    printf("    threads=%d  median=%.3f ms\n", active_threads, compute_ms);
    printf("    achieved_fp32=%.2f TFLOPS\n", tflops);
    printf("    AI_compute≈%.1f FLOP/byte  (2*4*iters / 8B)\n\n", ai_compute);

    // C: ridge from measured roofs (FLOP/byte).
    const double ridge_ai =
        (bw_gbs > 0.0) ? (tflops * 1e3) / bw_gbs : 0.0;

    printf("[C] Roofline read (measured roofs)\n");
    printf("    ridge_AI ≈ peak_FLOP/s / peak_BW ≈ %.2f FLOP/byte\n", ridge_ai);
    printf("    copy:    AI≈0     << ridge → memory-bound side\n");
    printf("    compute: AI≈%.1f %s ridge → compute-bound side\n", ai_compute,
           (ai_compute > ridge_ai) ? ">>" : "vs");
    printf("    Next: memory side → Module B; compute side → ILP/algorithm/Tensor.\n");
    printf("    SOL triage checklist → B-10. Optional NCU: 10_profile_roofline.sh\n");

    // Touch outputs so stores are not trivially DCE'd across the process.
    float sink = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sink, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    volatile float keep = sink;
    (void)keep;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
