/**
 * [Module A] 04. SIMT / Divergence / Replay
 * lane0 clock64 median after warmup. Not CUDA event kernel time.
 *
 * - Math: uniform FMA vs odd/even if (serialized + mask)
 * - SMEM: stride 1 vs stride 32 (bank replay demo; padding is B-02)
 * Results written to sink[] to prevent DCE (old build printed 1 cycle).
 */

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int kIters = 4096;
constexpr int kWarmup = 5;
constexpr int kRuns = 21;

__global__ void math_divergence_kernel(int mode, long long* duration, float* sink, int iters) {
    const int tid = threadIdx.x;
    float val = float(tid) + 1.0f;

    long long start = clock64();
    if (mode == 0) {
        for (int i = 0; i < iters; ++i) {
            val = val * 1.0001f + 1.0f;
        }
    } else if (tid & 1) {
        for (int i = 0; i < iters; ++i) {
            val = val * 1.0001f + 1.0f;
        }
    } else {
        for (int i = 0; i < iters; ++i) {
            val = val * 1.0001f + 2.0f;
        }
    }
    long long end = clock64();

    sink[tid] = val;
    if (tid == 0) {
        *duration = end - start;
    }
}

__global__ void bank_conflict_kernel(int stride, long long* duration, int* sink, int iters) {
    __shared__ volatile int s_data[32 * 33];
    const int tid = threadIdx.x;
    for (int i = tid; i < 32 * 33; i += blockDim.x) {
        s_data[i] = i;
    }
    __syncthreads();

    const int idx = tid * stride;
    int acc = 0;
    long long start = clock64();
    for (int i = 0; i < iters; ++i) {
        acc += s_data[idx];
        s_data[idx] = acc;
    }
    long long end = clock64();

    sink[tid] = acc;
    if (tid == 0) {
        *duration = end - start;
    }
}

static long long median_ll(std::vector<long long>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static long long run_math(int mode, long long* d_duration, float* d_sink) {
    std::vector<long long> samples;
    samples.reserve(kRuns);
    for (int w = 0; w < kWarmup; ++w) {
        math_divergence_kernel<<<1, 32>>>(mode, d_duration, d_sink, kIters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < kRuns; ++r) {
        math_divergence_kernel<<<1, 32>>>(mode, d_duration, d_sink, kIters);
        CUDA_CHECK(cudaDeviceSynchronize());
        long long h = 0;
        CUDA_CHECK(cudaMemcpy(&h, d_duration, sizeof(long long), cudaMemcpyDeviceToHost));
        samples.push_back(h);
    }
    return median_ll(samples);
}

static long long run_bank(int stride, long long* d_duration, int* d_sink) {
    std::vector<long long> samples;
    samples.reserve(kRuns);
    for (int w = 0; w < kWarmup; ++w) {
        bank_conflict_kernel<<<1, 32>>>(stride, d_duration, d_sink, kIters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < kRuns; ++r) {
        bank_conflict_kernel<<<1, 32>>>(stride, d_duration, d_sink, kIters);
        CUDA_CHECK(cudaDeviceSynchronize());
        long long h = 0;
        CUDA_CHECK(cudaMemcpy(&h, d_duration, sizeof(long long), cudaMemcpyDeviceToHost));
        samples.push_back(h);
    }
    return median_ll(samples);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[Host] GPU: %s\n", prop.name);
    printf("[Host] Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("[Host] Metric: lane0 clock64 median (warmup=%d, runs=%d, iters=%d)\n",
           kWarmup, kRuns, kIters);
    printf("[Host] Not CUDA event kernel time (see A-08).\n\n");

    long long* d_duration = nullptr;
    float* d_sink_f = nullptr;
    int* d_sink_i = nullptr;
    CUDA_CHECK(cudaMalloc(&d_duration, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink_f, 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sink_i, 32 * sizeof(int)));

    const long long base = run_math(0, d_duration, d_sink_f);
    const long long divg = run_math(1, d_duration, d_sink_f);
    printf("[Divergence] uniform cycles : %lld\n", base);
    printf("[Divergence] odd/even if    : %lld\n", divg);
    if (base > 0) {
        printf("[Divergence] ratio          : %.2fx  (textbook isolation ~2x; not a bill)\n\n",
               (double)divg / (double)base);
    }

    const long long no_cf = run_bank(1, d_duration, d_sink_i);
    const long long cf32 = run_bank(32, d_duration, d_sink_i);
    printf("[Replay]     stride-1 cycles : %lld\n", no_cf);
    printf("[Replay]     stride-32       : %lld\n", cf32);
    if (no_cf > 0) {
        printf("[Replay]     ratio           : %.2fx  (textbook 32-way ~32x; padding -> B-02)\n",
               (double)cf32 / (double)no_cf);
    }

    CUDA_CHECK(cudaFree(d_duration));
    CUDA_CHECK(cudaFree(d_sink_f));
    CUDA_CHECK(cudaFree(d_sink_i));
    return 0;
}
