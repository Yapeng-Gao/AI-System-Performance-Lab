/**
 * [Module A] 08. Streams / Events / H2D→Compute→D2H pipeline
 *
 * A: serial — pageable host + default stream 0
 * B: depth-first — pinned + NonBlocking streams (H2D→K→D2H per chunk)
 * C: breadth-first — same pinned/streams, but all H2D then all K then all D2H
 *
 * Metric: CUDA event median over the whole device workload (warmup + runs).
 * clock64 busy-wait lengthens the kernel so overlap is visible — not a real AI.
 * Pinned GB/s / CE saturation: B-06. Device async / TMA: B-07/B-08. Graph: C-06.
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

constexpr int kTotalElements = 8 * 1024 * 1024;  // 32 MiB float
constexpr int kChunkSize = 256 * 1024;           // 1 MiB float
constexpr long long kKernelLoad = 100000;        // clock64 busy-wait (artificial)
constexpr int kNumStreams = 4;
constexpr int kWarmup = 2;
constexpr int kRuns = 7;

__global__ void heavy_compute_kernel(float* data, int n, long long delay_clocks) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    float val = data[idx];
    const long long start = clock64();
    while (clock64() - start < delay_clocks) {
        val = sinf(val) * cosf(val);
    }
    data[idx] = val + 1.0f;
}

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

enum class Schedule { Serial, DepthFirst, BreadthFirst };

const char* schedule_name(Schedule s) {
    switch (s) {
        case Schedule::Serial:
            return "A_serial_pageable_default";
        case Schedule::DepthFirst:
            return "B_depth_first_pinned";
        case Schedule::BreadthFirst:
            return "C_breadth_first_pinned";
    }
    return "?";
}

void launch_chunk_kernel(float* d_base, int offset, int chunk, long long load,
                         cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (chunk + threads - 1) / threads;
    heavy_compute_kernel<<<blocks, threads, 0, stream>>>(d_base + offset, chunk, load);
}

void run_once(Schedule sched, float* h_data, float* d_data, int total, int chunk,
              long long load, const std::vector<cudaStream_t>& streams) {
    const int num_chunks = total / chunk;
    const size_t chunk_bytes = static_cast<size_t>(chunk) * sizeof(float);

    if (sched == Schedule::Serial) {
        for (int i = 0; i < num_chunks; ++i) {
            const int offset = i * chunk;
            CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes,
                                       cudaMemcpyHostToDevice, 0));
            launch_chunk_kernel(d_data, offset, chunk, load, 0);
            CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes,
                                       cudaMemcpyDeviceToHost, 0));
        }
        return;
    }

    const int n_streams = static_cast<int>(streams.size());
    if (sched == Schedule::DepthFirst) {
        for (int i = 0; i < num_chunks; ++i) {
            const int s = i % n_streams;
            const int offset = i * chunk;
            CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes,
                                       cudaMemcpyHostToDevice, streams[s]));
            launch_chunk_kernel(d_data, offset, chunk, load, streams[s]);
            CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes,
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        return;
    }

    // Breadth-first: all H2D, then all kernels, then all D2H (same streams).
    for (int i = 0; i < num_chunks; ++i) {
        const int s = i % n_streams;
        const int offset = i * chunk;
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes,
                                   cudaMemcpyHostToDevice, streams[s]));
    }
    for (int i = 0; i < num_chunks; ++i) {
        const int s = i % n_streams;
        const int offset = i * chunk;
        launch_chunk_kernel(d_data, offset, chunk, load, streams[s]);
    }
    for (int i = 0; i < num_chunks; ++i) {
        const int s = i % n_streams;
        const int offset = i * chunk;
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes,
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
}

float time_once_ms(Schedule sched, float* h_data, float* d_data, int total, int chunk,
                   long long load, const std::vector<cudaStream_t>& streams,
                   cudaEvent_t start, cudaEvent_t stop) {
    // NonBlocking streams do not join the legacy default stream; must device-sync
    // before recording stop, or ElapsedTime can finish early.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    run_once(sched, h_data, d_data, total, chunk, load, streams);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

float bench(Schedule sched, float* h_data, float* d_data, int total, int chunk,
            long long load, const std::vector<cudaStream_t>& streams, int warmup,
            int runs) {
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        (void)time_once_ms(sched, h_data, d_data, total, chunk, load, streams, start,
                           stop);
    }

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        samples.push_back(time_once_ms(sched, h_data, d_data, total, chunk, load,
                                       streams, start, stop));
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
    printf("sm_%d%d  asyncEngineCount=%d\n", prop.major, prop.minor,
           prop.asyncEngineCount);
    printf("Metric: CUDA event median (warmup=%d, runs=%d); not Host chrono\n",
           kWarmup, kRuns);
    printf("Load: clock64 busy-wait=%lld (artificial, so overlap is visible)\n",
           kKernelLoad);
    printf("Data: %.2f MiB total, %.2f MiB/chunk, streams=%d, chunks=%d\n\n",
           kTotalElements * sizeof(float) / (1024.0 * 1024.0),
           kChunkSize * sizeof(float) / (1024.0 * 1024.0), kNumStreams,
           kTotalElements / kChunkSize);

    if (prop.asyncEngineCount < 1) {
        printf("[Warn] asyncEngineCount < 1: do not expect copy||compute overlap.\n");
    }

    const size_t total_bytes =
        static_cast<size_t>(kTotalElements) * sizeof(float);

    float* h_pageable = static_cast<float*>(malloc(total_bytes));
    if (!h_pageable) {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }
    float* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, total_bytes));
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));

    for (int i = 0; i < kTotalElements; ++i) {
        h_pageable[i] = 1.0f;
        h_pinned[i] = 1.0f;
    }

    std::vector<cudaStream_t> streams(kNumStreams);
    for (int i = 0; i < kNumStreams; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    std::vector<cudaStream_t> no_streams;  // serial uses stream 0 only

    const float ms_a =
        bench(Schedule::Serial, h_pageable, d_data, kTotalElements, kChunkSize,
              kKernelLoad, no_streams, kWarmup, kRuns);
    const float ms_b =
        bench(Schedule::DepthFirst, h_pinned, d_data, kTotalElements, kChunkSize,
              kKernelLoad, streams, kWarmup, kRuns);
    const float ms_c =
        bench(Schedule::BreadthFirst, h_pinned, d_data, kTotalElements, kChunkSize,
              kKernelLoad, streams, kWarmup, kRuns);

    // Touch a few results so the writes are not trivially DCE'd across runs.
    volatile float sink = h_pinned[0] + h_pageable[0];
    (void)sink;

    printf("mode,median_ms\n");
    printf("%s,%.3f\n", schedule_name(Schedule::Serial), ms_a);
    printf("%s,%.3f\n", schedule_name(Schedule::DepthFirst), ms_b);
    printf("%s,%.3f\n", schedule_name(Schedule::BreadthFirst), ms_c);
    printf("\n");
    printf("ratio A/B (serial / depth-first): %.2fx\n", ms_a / ms_b);
    printf("ratio C/B (breadth / depth-first): %.2fx\n", ms_c / ms_b);
    printf("[Host] Event WaitEvent demo: prose only in A-08 (optional; Graph→C-06).\n");

    for (int i = 0; i < kNumStreams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
    free(h_pageable);
    return 0;
}
