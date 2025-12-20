/**
 * [Module A] 09. 调试与错误诊断
 * Bug Generator: 故意制造三种典型 GPU 错误，用于演示 Sanitizer 的威力。
 *
 * 用法: ./09_debug_and_sanitizer <mode>
 * mode 0: Out-of-Bounds (越界访问)
 * mode 1: Race Condition (数据竞争)
 * mode 2: Illegal Sync (非法同步/死锁)
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

// --- Bug 1: 内存越界 (Out of Bounds) ---
// 申请了 N 个，但试图访问 N+1
__global__ void oob_kernel(int* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 错误：当 idx == n 时，发生了越界写入
    if (idx <= n) {
        data[idx] = 1;
    }
}

// --- Bug 2: 数据竞争 (Race Condition) ---
// 多个线程同时写 Shared Memory 同一地址，且没有原子操作或同步
__global__ void race_kernel(int* out) {
    __shared__ int s_val;
    if (threadIdx.x == 0) s_val = 0;
    __syncthreads();

    // 错误：多个线程同时读写 s_val，结果未定义
    // 正确做法是使用 atomicAdd 或 归约
    s_val += 1;

    __syncthreads();
    if (threadIdx.x == 0) *out = s_val;
}

// --- Bug 3: 非法同步 (Illegal Sync) ---
// 在分支发散区域调用 __syncthreads()，导致死锁或未定义行为
__global__ void illegal_sync_kernel(int* data) {
    int tid = threadIdx.x;

    // 奇数线程进入 if，偶数线程跳过
    if (tid % 2 != 0) {
        data[tid] *= 2;
        // 错误：只有一半线程能到达这里，另一半在外面等
        // 这会导致 Synccheck 报错，或者直接死锁
        __syncthreads();
    } else {
        data[tid] += 1;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <mode>\n", argv[0]);
        printf("  0: Out-of-Bounds Write\n");
        printf("  1: Shared Memory Race\n");
        printf("  2: Illegal Synchronization\n");
        return 1;
    }

    int mode = atoi(argv[1]);
    int N = 1024;
    int* d_data;

    printf("[Host] Starting Bug Generator (Mode %d)...\n", mode);

    if (mode == 0) {
        // --- Case 0: OOB ---
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
        // Launch 1025 threads (N+1)
        oob_kernel<<<1, N + 1>>>(d_data, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Host] OOB Kernel Finished (Did it crash?)\n");
        CUDA_CHECK(cudaFree(d_data));
    }
    else if (mode == 1) {
        // --- Case 1: Race ---
        int* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
        race_kernel<<<1, 32>>>(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Host] Race Kernel Finished (Check Sanitizer output)\n");
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (mode == 2) {
        // --- Case 2: Sync ---
        CUDA_CHECK(cudaMalloc(&d_data, 32 * sizeof(int)));
        illegal_sync_kernel<<<1, 32>>>(d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Host] Sync Kernel Finished\n");
        CUDA_CHECK(cudaFree(d_data));
    }

    // 注意：如果有异步错误，可能要等到这里甚至程序退出时才报错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[Host] Caught Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}