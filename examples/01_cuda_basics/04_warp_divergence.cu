/**
 * [Module A] 04. 线程调度：SIMT, Divergence 与 Replay
 * Micro-benchmark: 量化分支发散与 Shared Memory Bank Conflict 的物理代价
 *
 * 实验设计：
 * 1. Math Latency: 对比无分支代码 vs 奇偶分支代码 (ALU 利用率减半验证)
 * 2. Replay Latency: 对比无冲突访问 vs 32-way Bank Conflict (指令重播验证)
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>

// --- 宏定义：生产级错误检查 ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 循环次数，足够大以摊薄 clock() 开销，但不要大到触发 TDR (超时)
constexpr int PER_KERNEL_ITERS = 1000;

// --- Device 辅助函数：防止编译器优化 ---
__device__ __forceinline__ void d_print_value(int val) {
    // 使用 volatile 变量防止编译器优化掉这个调用
    volatile int dummy = val;
    (void)dummy; // 抑制未使用变量警告
}

// --- Kernel 1: Math Divergence 测试 ---
// 模式 0: Baseline (无分支)
// 模式 1: Divergent (奇偶线程不同分支)
__global__ void math_divergence_kernel(int mode, long long* duration, int iters) {
    int tid = threadIdx.x;
    float val = float(tid);

    // 预热指令 Cache
    if (val > 0.0f) val += 1.0f;

    long long start = clock64();

    if (mode == 0) {
        // [Baseline] 所有线程走同一条路径 -> SIMT 满载
        #pragma unroll 4
        for (int i = 0; i < iters; ++i) {
            val += 2.0f;
            val *= 1.01f;
        }
    } else {
        // [Divergent] 奇偶线程走不同路径 -> SIMT 串行化
        // 硬件行为：先 Mask 偶数线程执行 A，再 Mask 奇数线程执行 B
        // 理论吞吐量减半
        #pragma unroll 4
        for (int i = 0; i < iters; ++i) {
            if (tid % 2 == 0) {
                val += 2.0f; // 分支 A
            } else {
                val *= 1.01f; // 分支 B
            }
        }
    }

    long long end = clock64();

    // 防止编译器优化掉计算逻辑
    if (val == 999999.0f) d_print_value((int)val);

    // 仅记录 Warp 0 Lane 0 的时间作为代表
    if (tid == 0 && blockIdx.x == 0) {
        *duration = (end - start);
    }
}

// --- Kernel 2: Shared Memory Replay 测试 (Bank Conflict) ---
// 模式 0: No Conflict (Stride = 1)
// 模式 1: 32-way Conflict (Stride = 32)
__global__ void bank_conflict_kernel(int stride, long long* duration, int iters) {
    // 声明 Shared Memory (Volatile 防止编译器优化掉内存访问)
    // 大小足够容纳最大 stride 的访问
    __shared__ volatile int s_data[32 * 33];

    int tid = threadIdx.x;

    // 初始化 Shared Memory 以避免非法访问 (虽不严格必要，但为了稳健)
    for (int i = tid; i < 32 * 33; i += blockDim.x) {
        s_data[i] = i;
    }
    __syncthreads();

    // 索引计算：
    // Case 0 (Stride=1): 线程 i 访问地址 i。
    //    Address: 0, 4, 8, ...
    //    Bank: 0, 1, 2, ... (无冲突，1个周期完成)
    //
    // Case 1 (Stride=32): 线程 i 访问地址 i * 32。
    //    Address: 0, 128, 256...
    //    Bank: (Addr / 4) % 32 = (i * 32) % 32 = 0
    //    所有 32 个线程都访问 Bank 0！(32-way 冲突，指令重播 32 次)
    int idx = tid * stride;

    long long start = clock64();

    #pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        // 读后写，制造依赖链，迫使指令串行执行
        s_data[idx] += 1;
    }

    long long end = clock64();

    if (tid == 0 && blockIdx.x == 0) {
        *duration = (end - start);
    }
}

int main() {
    printf("=================================================================\n");
    printf("   AI System Performance Lab - SIMT & Replay Analyzer   \n");
    printf("=================================================================\n");

    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("Running on GPU: %s (Arch sm_%d%d)\n\n",
           prop.name, prop.major, prop.minor);

    long long *d_duration, h_duration;
    CUDA_CHECK(cudaMalloc(&d_duration, sizeof(long long)));

    // --- 实验 1: Math Divergence ---
    printf("[Experiment 1] Measuring Warp Divergence Cost (ALU)\n");

    // 1.1 Baseline
    math_divergence_kernel<<<1, 32>>>(0, d_duration, PER_KERNEL_ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_duration, d_duration, sizeof(long long), cudaMemcpyDeviceToHost));
    double time_baseline = (double)h_duration;
    printf("  Baseline (No Branch) Cycles : %.0f\n", time_baseline);

    // 1.2 Divergent
    math_divergence_kernel<<<1, 32>>>(1, d_duration, PER_KERNEL_ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_duration, d_duration, sizeof(long long), cudaMemcpyDeviceToHost));
    double time_divergent = (double)h_duration;
    printf("  Divergent (If-Else) Cycles  : %.0f\n", time_divergent);

    printf("  >> Performance Penalty      : %.2fx Slower (Ideal: 2.0x)\n\n",
           time_divergent / time_baseline);


    // --- 实验 2: Bank Conflict Replay ---
    printf("[Experiment 2] Measuring Instruction Replay Cost (Shared Mem)\n");

    // 2.1 No Conflict (Stride 1)
    bank_conflict_kernel<<<1, 32>>>(1, d_duration, PER_KERNEL_ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_duration, d_duration, sizeof(long long), cudaMemcpyDeviceToHost));
    double time_no_conflict = (double)h_duration;
    printf("  Linear Access (No Conflict) : %.0f cycles\n", time_no_conflict);

    // 2.2 32-way Conflict (Stride 32)
    bank_conflict_kernel<<<1, 32>>>(32, d_duration, PER_KERNEL_ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_duration, d_duration, sizeof(long long), cudaMemcpyDeviceToHost));
    double time_conflict = (double)h_duration;
    printf("  Stride-32 (32-way Conflict) : %.0f cycles\n", time_conflict);

    printf("  >> Replay Penalty           : %.2fx Slower (Ideal: 32.0x)\n",
           time_conflict / time_no_conflict);

    printf("\nNote: 'Ideal' assumes pure isolation. Real hardware pipelines may hide some latency.\n");

    CUDA_CHECK(cudaFree(d_duration));
    return 0;
}