/**
 * [Module B] 11. Global Memory 极致优化
 * Bandwidth Micro-Benchmark: 验证 HBM 对齐、向量化 (float4) 与 L2 驻留控制
 *
 * 实验组：
 * 1. Misaligned Copy: 故意制造错位访问，破坏 Coalescing。
 * 2. Scalar Copy: 标准 float 读取。
 * 3. Vectorized Copy: 使用 float4 读取，验证指令发射效率。
 * 4. L2 Persistence: 使用 CUDA 12 API 锁定 L2 Cache。
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 数据规模：64MB (足够大以触达 HBM 带宽墙)
const size_t N = 16 * 1024 * 1024;
const size_t BYTES = N * sizeof(float);

// =========================================================
// Kernel 1: Misaligned / Strided Access (带宽杀手)
// =========================================================
// 如果 offset 不是 32 的倍数（例如 1），或者 stride > 1，
// 一个 Warp 的请求将分裂成多个 Memory Transactions。
__global__ void misaligned_copy_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - offset) {
        // 读取地址 = base + idx + offset
        // 如果 offset=1, 线程0读地址4, 线程1读地址8...
        // 导致整个 Warp 的访问无法与 32-byte Sector 对齐
        out[idx] = in[idx + offset];
    }
}

// =========================================================
// Kernel 2: Aligned Scalar Copy (基准线)
// =========================================================
// 标准的合并访问，但指令数较多 (LDG.E.32)
__global__ void aligned_copy_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = in[i];
    }
}

// =========================================================
// Kernel 3: Vectorized Copy (指令层优化)
// =========================================================
// 强制生成 LDG.E.128，减少 75% 的指令发射压力
__global__ void vectorized_copy_kernel(const float4* __restrict__ in,
                                       float4* __restrict__ out,
                                       int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n_vec; i += stride) {
        out[i] = in[i];
    }
}

// =========================================================
// Kernel 4: L2 Simulation (模拟权重重用)
// =========================================================
// 反复读取同一块小内存，模拟深度学习中的 Weight Reuse
__global__ void repeated_access_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int n, int repeats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 0.0f;
        // 模拟多次读取同一地址
        // 如果 L2 驻留生效，后续读取将极快
        for (int r = 0; r < repeats; ++r) {
            val += in[idx];
        }
        out[idx] = val;
    }
}

// --- 辅助函数：运行并计时 ---
template <typename Func>
float measure_performance(Func kernel_launcher, const char* label, size_t bytes_transferred) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    kernel_launcher();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        kernel_launcher();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / 10.0f;

    // GB/s
    float gbps = (bytes_transferred / 1e9) / (avg_ms / 1000.0f);
    printf("[%-20s] Time: %6.3f ms | Bandwidth: %7.2f GB/s\n", label, avg_ms, gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return gbps;
}

int main() {
    int dev_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    printf("Target GPU: %s (L2 Cache: %.2f MB)\n", prop.name, prop.l2CacheSize / 1024.0 / 1024.0);
    printf("Total Data: %.2f MB\n\n", BYTES / 1024.0 / 1024.0);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, BYTES));
    CUDA_CHECK(cudaMalloc(&d_out, BYTES));

    // Block/Grid 配置
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    // --- Experiment 1: Misaligned Access ---
    // Offset = 1 (4 bytes), 破坏了 32-byte 甚至 128-byte 对齐
    measure_performance([&]() {
        misaligned_copy_kernel<<<grid_size, block_size>>>(d_in, d_out, N, 1);
    }, "Misaligned (Off=1)", BYTES * 2); // Read + Write

    // --- Experiment 2: Aligned Scalar ---
    measure_performance([&]() {
        aligned_copy_kernel<<<grid_size, block_size>>>(d_in, d_out, N);
    }, "Aligned (Float)", BYTES * 2);

    // --- Experiment 3: Vectorized (float4) ---
    measure_performance([&]() {
        vectorized_copy_kernel<<<grid_size/4, block_size>>>((float4*)d_in, (float4*)d_out, N/4);
    }, "Vectorized (Float4)", BYTES * 2);

    // --- Experiment 4: L2 Persistence (CUDA 11/12 Feature) ---
    printf("\n=== L2 Persistence Control (Weight Reuse Simulation) ===\n");

    // 我们只使用 20MB 数据，确保它能装进 L2 (假设 L2 > 20MB，如 A100/4090)
    size_t l2_subset_bytes = 20 * 1024 * 1024;
    int subset_n = l2_subset_bytes / sizeof(float);
    int repeats = 10; // 重复读取 10 次

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 4.1 默认行为 (LRU)
    // 第一次由 LRU 管理
    float bw_default = measure_performance([&]() {
        repeated_access_kernel<<<subset_n/256, 256, 0, stream>>>(d_in, d_out, subset_n, repeats);
    }, "L2 Default (LRU)", l2_subset_bytes * repeats); // 只算 Read 量

    // 4.2 显式 L2 驻留 (Persisting)
    // 定义访问策略窗口
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_in);
    stream_attr.accessPolicyWindow.num_bytes = l2_subset_bytes;
    stream_attr.accessPolicyWindow.hitRatio = 1.0; // 100% 驻留
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // 命中设为驻留
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; // 未命中设为流式

    CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr));

    float bw_persisting = measure_performance([&]() {
        repeated_access_kernel<<<subset_n/256, 256, 0, stream>>>(d_in, d_out, subset_n, repeats);
    }, "L2 Persisting", l2_subset_bytes * repeats);

    printf(">> L2 Control Improvement: %.2f%%\n", (bw_persisting - bw_default) / bw_default * 100.0f);

    // 清理
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}