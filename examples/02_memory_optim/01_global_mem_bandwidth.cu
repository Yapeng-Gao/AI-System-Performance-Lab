/**
 * [Module B] 11. Global Memory 极致优化
 * Bandwidth Micro-Benchmark: 覆盖物理层、指令层与缓存层的所有优化手段
 *
 * 测试项：
 * 1. [Physical] Misaligned: 错位访问 (带宽杀手)
 * 2. [Physical] Aligned: 标准对齐
 * 3. [Instruction] Vectorized: float4 (减少指令数)
 * 4. [Instruction] Async Copy: Pipeline Copy (cp.async / TMA 预演)
 * 5. [Cache] LDG.NT: Non-Temporal Load (流式读取，不污染 L2)
 * 6. [Cache] L2 Persistence: 锁住 L2 Cache (模拟权重复用)
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
// 引入现代 CUDA 异步库
#include <cuda/pipeline>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

const size_t N = 16 * 1024 * 1024;
const size_t BYTES = N * sizeof(float);

// =========================================================
// Kernel 1: Misaligned (物理层反例)
// =========================================================
__global__ void misaligned_copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - offset) {
        out[idx] = in[idx + offset]; // 错位读取，破坏 Transaction
    }
}

// =========================================================
// Kernel 2: Vectorized float4 (指令层基础)
// =========================================================
__global__ void vectorized_copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n_vec; i += stride) {
        out[i] = in[i];
    }
}

// =========================================================
// Kernel 3: LDG.NT (缓存层优化 - 流式数据)
// =========================================================
// 使用内建函数 __ldcs (Load Cache Streaming) 生成 LDG.NT 指令
// 告诉硬件：这条数据读完就扔，别占 L2 位置
__global__ void non_temporal_copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n_vec; i += stride) {
        // __ldcs: Load with Cache Streaming hint
        // 对应 SASS: LDG.E.128.STREAM
        float4 val = __ldcs(&in[i]);
        out[i] = val;
    }
}

// =========================================================
// Kernel 4: Async Copy Pipeline (指令层进阶 - TMA/cp.async)
// =========================================================
// 使用 CUDA 12 推荐的 cuda::pipeline 原语
// 在 Ampere 上映射为 cp.async，在 Hopper 上为 TMA 铺路
__global__ void async_copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n_vec) {
    // 这是一个简单的 Pipeline 演示：Global -> Shared (Async) -> Register -> Global
    // 实际 GEMM 中是 Global -> Shared -> Register -> Compute
    extern __shared__ float4 smem_buffer[]; // 动态共享内存

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 创建 Pipeline 对象
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 假设每个 Block 处理一块连续数据
    if (idx < n_vec) {
        // 1. 异步搬运: Global -> Shared (Bypass Register)
        // 这里的 memcpy_async 会生成 cp.async 指令
        pipe.producer_acquire();
        cuda::memcpy_async(&smem_buffer[tid], &in[idx], sizeof(float4), pipe);
        pipe.producer_commit();

        // 2. 等待搬运完成
        pipe.consumer_wait();

        // 3. 消费数据 (这里简单写回，实际中是做计算)
        out[idx] = smem_buffer[tid];

        // 4. 释放资源
        pipe.consumer_release();
    }
}

// =========================================================
// Kernel 5: L2 Simulation (缓存层优化 - 权重驻留)
// =========================================================
__global__ void repeated_access_kernel(const float* __restrict__ in, float* __restrict__ out, int n, int repeats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 0.0f;
        for (int r = 0; r < repeats; ++r) {
            val += in[idx]; // 如果 L2 驻留生效，这里将飞快
        }
        out[idx] = val;
    }
}

// --- 计时辅助 ---
template <typename Func>
float measure_bw(Func kernel_launcher, const char* label, size_t bytes) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    kernel_launcher(); // Warmup
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; ++i) kernel_launcher();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 10.0f;
    float gbps = (bytes / 1e9) / (avg_ms / 1000.0f);

    printf("[%-20s] %7.2f GB/s\n", label, gbps);
    return gbps;
}

int main() {
    int dev_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);
    printf("GPU: %s | L2: %.2f MB\n", prop.name, prop.l2CacheSize / 1024.0 / 1024.0);

    // 计算理论带宽
#if defined(CUDA_VERSION) && CUDA_VERSION < 12000
    // CUDA 11.x 及更早版本
    double theory_bw = (double)prop.memoryClockRate * prop.memoryBusWidth * 2 / 8 / 1e6;
    printf("Theoretical Bandwidth: %.2f GB/s\n\n", theory_bw);
#else
    // CUDA 12+ 版本：memoryClockRate 字段已移除
    // 使用保守估算，实际应用中应使用 NVML API 获取准确值
    double mem_clock_ghz = 1.0; // 保守估算，实际值需要通过 NVML 获取
    double theory_bw = mem_clock_ghz * (prop.memoryBusWidth / 8.0) * 2.0;
    printf("Theoretical Bandwidth: %.2f GB/s (Estimated, use NVML API for accurate value)\n\n", theory_bw);
#endif

    float *d_in, *d_out;
    cudaMalloc(&d_in, BYTES);
    cudaMalloc(&d_out, BYTES);

    int block = 256;
    int grid = (N + block - 1) / block;
    int vec_grid = (N/4 + block - 1) / block;

    // 1. 物理层：错位 vs 对齐
    measure_bw([&](){ misaligned_copy_kernel<<<grid, block>>>(d_in, d_out, N, 1); },
               "1. Misaligned", BYTES*2);

    // 2. 指令层：向量化 float4
    measure_bw([&](){ vectorized_copy_kernel<<<vec_grid, block>>>((float4*)d_in, (float4*)d_out, N/4); },
               "2. Vectorized float4", BYTES*2);

    // 3. 缓存层：LDG.NT (Non-Temporal)
    measure_bw([&](){ non_temporal_copy_kernel<<<vec_grid, block>>>((float4*)d_in, (float4*)d_out, N/4); },
               "3. LDG.NT (Stream)", BYTES*2);

    // 4. 架构层：Async Copy (Pipeline)
    // 动态 Shared Mem 大小 = block * sizeof(float4)
    measure_bw([&](){ async_copy_kernel<<<vec_grid, block, block*sizeof(float4)>>>((float4*)d_in, (float4*)d_out, N/4); },
               "4. Async Copy", BYTES*2);

    // 5. 缓存层：L2 Persistence
    size_t l2_data_size = 20 * 1024 * 1024; // 20MB < L2 Size
    int repeats = 50;
    cudaStream_t s; cudaStreamCreate(&s);

    // 5.1 默认 LRU
    printf("\n=== L2 Persistence Test (20MB Data, 50 Repeats) ===\n");
    float bw_lru = measure_bw([&](){
        repeated_access_kernel<<<l2_data_size/4/256, 256, 0, s>>>(d_in, d_out, l2_data_size/4, repeats);
    }, "L2 Default (LRU)", l2_data_size * repeats);

    // 5.2 开启驻留
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr = d_in;
    attr.accessPolicyWindow.num_bytes = l2_data_size;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);

    float bw_persist = measure_bw([&](){
        repeated_access_kernel<<<l2_data_size/4/256, 256, 0, s>>>(d_in, d_out, l2_data_size/4, repeats);
    }, "L2 Persisting", l2_data_size * repeats);

    printf(">> Improvement: %.2f%%\n", (bw_persist - bw_lru)/bw_lru * 100.0f);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}