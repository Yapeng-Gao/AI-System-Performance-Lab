/**
 * [Module B] 12. Shared Memory 深度优化
 * Bank Conflict Analyzer: 验证 Padding 与 XOR Swizzling 的效果
 *
 * 场景模拟：
 * 我们模拟一个 32x32 的矩阵块操作。
 * Warp 中的 32 个线程试图按"列"访问这个矩阵（这是矩阵转置或 GEMM 中的常见模式）。
 *
 * 测试项：
 * 1. Naive: 发生 32-way Bank Conflict (最慢)
 * 2. Padding: 使用 [32][33] 布局消除冲突 (快，但浪费空间)
 * 3. Swizzling: 使用 XOR 映射消除冲突 (快，且零空间浪费)
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 矩阵块大小：32x32 float
constexpr int TILE_DIM = 32;
// 循环次数，放大 Latency 差异
constexpr int REPEAT_ITERS = 1000;

// =========================================================
// Kernel 1: Naive (32-way Bank Conflict)
// =========================================================
// 布局: smem[32][32]
// 访问: 线程 tid 读取 smem[tid][0], smem[tid][1]...
// 实际上，这模拟的是列访问模式的冲突本质。
// 如果 stride = 32 (一行 32 float)，那么 smem[0][0] 和 smem[1][0] 都在 Bank 0
__global__ void naive_conflict_kernel(float* out, long long* duration) {
    // 声明标准布局
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int tid = threadIdx.x;

    // 初始化 (避免编译器优化)
    for (int i = 0; i < TILE_DIM; ++i) tile[tid][i] = tid + i;
    __syncthreads();

    long long start = clock64();

    float val = 0.0f;
    #pragma unroll
    for (int k = 0; k < REPEAT_ITERS; ++k) {
        // 模拟列访问冲突：
        // 假设我们要读取第 k 列。
        // 线程 0 读取 tile[0][k] -> Bank k
        // 线程 1 读取 tile[1][k] -> Bank k
        // ...
        // 线程 31 读取 tile[31][k] -> Bank k
        // 所有 32 个线程访问同一个 Bank k -> 32-way Conflict!

        // 为了演示方便，这里我们让每个线程访问不同行的同一列
        // 简单的写法：访问 tile[tid][fixed_col]
        // 但为了让编译器生成指令，我们让列动起来
        int col = k % TILE_DIM;
        val += tile[tid][col];
    }

    long long end = clock64();

    if (tid == 0) {
        *duration = (end - start);
        out[0] = val; // 防止被优化
    }
}

// =========================================================
// Kernel 2: Padding (空间换时间)
// =========================================================
// 布局: smem[32][33] -> 多了一列 Padding
__global__ void padding_kernel(float* out, long long* duration) {
    // 关键点：[33]
    // Row 0 占用 33 floats. Row 1 的起始地址比 Row 0 偏移 33 words.
    // 33 % 32 = 1. 所以每一行起始 Bank 偏移 1。
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tid = threadIdx.x;

    for (int i = 0; i < TILE_DIM; ++i) tile[tid][i] = tid + i;
    __syncthreads();

    long long start = clock64();

    float val = 0.0f;
    #pragma unroll
    for (int k = 0; k < REPEAT_ITERS; ++k) {
        int col = k % TILE_DIM;
        // 访问模式与 Naive 完全相同
        // 但由于物理内存布局变了，tile[0][col] 和 tile[1][col] 在不同 Bank
        val += tile[tid][col];
    }

    long long end = clock64();

    if (tid == 0) {
        *duration = (end - start);
        out[0] = val;
    }
}

// =========================================================
// Kernel 3: XOR Swizzling (现代 Tensor Core 标准)
// =========================================================
// 布局: smem[32][32] -> 紧凑布局
// 映射: 逻辑列 -> 物理 Bank (通过 XOR)
__global__ void swizzle_kernel(float* out, long long* duration) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int tid = threadIdx.x;

    // 为了避免访问未初始化的 Shared Memory，这里做一个简单的初始化。
    // 对于 Bank Conflict 分析，我们只关心访存模式，对具体数值不敏感。
    for (int i = 0; i < TILE_DIM; ++i) {
        tile[tid][i] = tid + i;
    }
    __syncthreads();

    long long start = clock64();

    float val = 0.0f;
    #pragma unroll
    for (int k = 0; k < REPEAT_ITERS; ++k) {
        int logical_col = k % TILE_DIM;
        int logical_row = tid;

        // --- Swizzling Logic ---
        // 教学版: col ^ row
        // 物理列 = 逻辑列 XOR 逻辑行
        // 当我们访问同一逻辑列(logical_col固定)时，logical_row(tid) 变化
        // 导致 physical_col = const ^ tid，这会生成 0..31 的排列
        // 从而完美错开 Bank

        // 进阶版 (CUTLASS 风格):
        // int physical_col = logical_col ^ (logical_row >> 3);
        // 这里演示最基础的 XOR，效果最明显
        int physical_col = logical_col ^ logical_row;

        val += tile[logical_row][physical_col];
    }

    long long end = clock64();

    if (tid == 0) {
        *duration = (end - start);
        out[0] = val;
    }
}

int main() {
    printf("==============================================================\n");
    printf("   Shared Memory Bank Conflict Analysis (Block Size: 32x32)   \n");
    printf("==============================================================\n");

    float *d_out;
    long long *d_dur, h_dur_naive, h_dur_pad, h_dur_swizzle;

    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dur, sizeof(long long)));

    // 1. Naive
    naive_conflict_kernel<<<1, 32>>>(d_out, d_dur);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_dur_naive, d_dur, sizeof(long long), cudaMemcpyDeviceToHost));
    printf("[Naive]   32-way Conflict Cycles: %lld\n", h_dur_naive);

    // 2. Padding
    padding_kernel<<<1, 32>>>(d_out, d_dur);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_dur_pad, d_dur, sizeof(long long), cudaMemcpyDeviceToHost));
    printf("[Padding] Conflict-Free Cycles  : %lld\n", h_dur_pad);

    // 3. Swizzling
    swizzle_kernel<<<1, 32>>>(d_out, d_dur);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_dur_swizzle, d_dur, sizeof(long long), cudaMemcpyDeviceToHost));
    printf("[Swizzle] XOR Pattern Cycles    : %lld\n", h_dur_swizzle);

    // Analysis
    printf("\n=== Performance Gain ===\n");
    printf("Padding Speedup : %.2fx\n", (double)h_dur_naive / h_dur_pad);
    printf("Swizzle Speedup : %.2fx\n", (double)h_dur_naive / h_dur_swizzle);

    // Check theoretical limits
    // 理想情况下，32-way conflict 会慢 32 倍。但由于流水线掩盖，实测通常在 20x-30x 之间。
    if (h_dur_naive / h_dur_pad > 10.0) {
        printf(">> Result: SUCCESS. Heavy conflicts detected and resolved.\n");
    } else {
        printf(">> Result: WARNING. Speedup lower than expected (check Warp size?).\n");
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_dur));
    return 0;
}