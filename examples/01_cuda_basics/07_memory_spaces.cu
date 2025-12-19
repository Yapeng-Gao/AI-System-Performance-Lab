/**
 * [Module A] 07. 内存模型全景
 * 验证目标：
 * 1. 地址空间探测 (Address Space Probing)
 * 2. UVA Zero-Copy 实战 (Host Pinned Memory direct access)
 * 3. 制造 Local Memory Spilling (缓存污染源)
 * 4. __restrict__ 对编译生成代码的影响
 */

#include <iostream>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 声明一个 Device 全局变量用于地址测试
__device__ int g_device_var = 42;

// ==========================================
// Part 1: 地址空间探测 Kernel
// ==========================================
__global__ void address_space_probe(int* d_ptr, int* h_pinned_ptr) {
    // 声明 Shared Memory
    __shared__ int s_var;
    // 声明 Local Variable (通常在寄存器，但也可能在 Local Memory)
    int l_var = 10;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n[Device] === Memory Address Map ===\n");
        printf("  Global Memory (HBM) Ptr:    %p\n", d_ptr);
        printf("  Global Variable (Static):   %p\n", &g_device_var);
        printf("  Shared Memory (SRAM):       %p (Small offset usually)\n", &s_var);
        printf("  Local Variable (Stack):     %p (If address taken -> Local Mem)\n", &l_var);
        printf("  Host Pinned Ptr (UVA/PCIe): %p\n", h_pinned_ptr);
        printf("================================\n\n");

        // 测试 UVA Zero-Copy: 直接在 GPU 读取 CPU 内存
        // 如果没有 UVA，这里会 Segfault
        int val = *h_pinned_ptr;
        printf("[Device] Read from Host Pinned Memory: %d (Success! UVA works)\n", val);
    }
}

// ==========================================
// Part 2: 制造 Register Spilling (Local Memory)
// ==========================================
// 我们定义一个巨大的数组，且动态索引，强迫编译器将其放入 Local Memory (HBM)
// 并在 SASS 中生成 LDL/STL 指令
__global__ void force_local_memory_spill(float* out, int n) {
    int tid = threadIdx.x;

    // 巨大的局部数组，寄存器放不下 -> 溢出到 Local Memory
    float local_buffer[100];

    // 初始化 (防止被优化掉)
    #pragma unroll
    for (int i = 0; i < 100; ++i) {
        local_buffer[i] = tid * 0.01f + i;
    }

    // 动态索引访问 (编译器无法优化为寄存器别名)
    // 这里的读写将极其缓慢，并污染 L1 Cache
    for (int i = 0; i < 100; ++i) {
        int idx = (tid + i) % 100;
        local_buffer[idx] += 1.0f;
    }

    out[tid] = local_buffer[0];
}

// ==========================================
// Part 3: __restrict__ 优化测试
// ==========================================

// Case A: 无 restrict (编译器必须保守，假设 a, b, c 可能重叠)
__global__ void add_no_restrict(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // 编译器很难生成 LDG.128 (vectorized load)，因为怕 a[i] 修改了 b[i+1]
        c[idx] = a[idx] + b[idx];
    }
}

// Case B: 有 restrict (编译器大胆优化)
// 可能生成 LDG.NC (Texture Cache) 或 LDG.128 (Vectorized)
__global__ void add_with_restrict(float* __restrict__ a,
                                  float* __restrict__ b,
                                  float* __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("[Host] Starting Memory Hierarchy Analysis...\n");

    // --- 1. 准备 UVA 环境 ---
    int* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));

    int* h_pinned_ptr;
    // cudaHostAllocMapped: 允许 Device 访问此 Host 内存
    CUDA_CHECK(cudaHostAlloc(&h_pinned_ptr, sizeof(int), cudaHostAllocMapped));
    *h_pinned_ptr = 999; // CPU 写入

    // --- 2. 启动地址探测 Kernel ---
    printf("[Host] Launching Address Probe...\n");
    address_space_probe<<<1, 1>>>(d_ptr, h_pinned_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 3. 验证 Zero-Copy 性能差异 (简单演示) ---
    // Zero-Copy 走 PCIe (64GB/s)，Global Memory 走 HBM (2000GB/s)
    // 在大量随机访问下，Pinned Memory 会显著慢于 Device Memory
    printf("\n[Host] To see Local Memory Spilling instructions (LDL/STL),\n"
           "       please run the accompanying '07_inspect_sass.sh' script.\n");

    // 触发 Spilling Kernel 编译
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));
    force_local_memory_spill<<<1, 256>>>(d_out, 256);

    // 触发 Restrict Kernel 编译
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, 1024*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, 1024*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, 1024*sizeof(float)));
    add_no_restrict<<<1, 256>>>(da, db, dc, 1024);
    add_with_restrict<<<1, 256>>>(da, db, dc, 1024);

    CUDA_CHECK(cudaDeviceSynchronize());

    // 清理
    CUDA_CHECK(cudaFree(d_ptr));
    CUDA_CHECK(cudaFreeHost(h_pinned_ptr)); // 注意用 FreeHost
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    return 0;
}