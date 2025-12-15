/**
 * [Module A] 05. Kernel 结构与 ABI 分析
 * 验证目标：
 * 1. 结构体对齐陷阱 (Host vs Device Layout)
 * 2. __noinline__ 与 __forceinline__ 的区别
 * 3. __launch_bounds__ 对寄存器使用的影响
 */

#include <iostream>
#include <cstdio>
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

// ==========================================
// Part 1: 结构体对齐陷阱
// ==========================================

// 这是一个危险的结构体定义
// Host 编译器 (GCC/MSVC) 和 Device 编译器 (NVCC) 可能采用不同的 Padding 策略
struct DangerousStruct {
    char a;      // 1 byte
    // 隐式 Padding? (Host: maybe 0, Device: likely 3)
    int b;       // 4 bytes
};

// 这是一个安全的结构体定义 (C++11 alignas)
struct SafeStruct {
    alignas(4) char a;
    int b;
};

__global__ void alignment_check_kernel(DangerousStruct s1, SafeStruct s2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[Device] DangerousStruct: Offset of b = %lu bytes\n",
               (size_t)&s1.b - (size_t)&s1);
        printf("[Device] SafeStruct:      Offset of b = %lu bytes\n",
               (size_t)&s2.b - (size_t)&s2);

        // 验证值是否正确传递
        printf("[Device] Values: s1.b=%d (Expected 42), s2.b=%d (Expected 100)\n", s1.b, s2.b);
    }
}

// ==========================================
// Part 2: 内联博弈 (__noinline__ vs __forceinline__)
// ==========================================

// 强制不内联：编译器会生成 CAL (Call) 和 RET (Return) 指令
// 这会导致参数压栈，使用 Local Memory (慢!)
__device__ __noinline__ int math_function_noinline(int x) {
    return x * x + 3 * x;
}

// 强制内联：代码直接展开，无函数调用开销，但增加 Kernel 体积
__device__ __forceinline__ int math_function_inline(int x) {
    return x * x + 3 * x;
}

__global__ void test_noinline_kernel(int* out, int x) {
    int tid = threadIdx.x;
    out[tid] = math_function_noinline(x + tid);
}

__global__ void test_forceinline_kernel(int* out, int x) {
    int tid = threadIdx.x;
    out[tid] = math_function_inline(x + tid);
}

// ==========================================
// Part 3: Launch Bounds (资源控制)
// ==========================================

// 1. 无限制 Kernel：编译器会尽可能多用寄存器以优化性能
// 可能会导致每个 SM 只能跑 1-2 个 Block (Low Occupancy)
__global__ void heavy_kernel_default(float* out, int N) {
    int tid = threadIdx.x;
    // 定义大量寄存器变量，模拟高压场景
    float r[50];

    // 初始化
    for(int i=0; i<50; ++i) r[i] = tid * 0.1f + i;

    // 繁重计算
    #pragma unroll
    for(int k=0; k<100; ++k) {
        for(int i=0; i<50; ++i) {
            r[i] = r[i] * r[(i+1)%50] + 1.0f;
        }
    }

    // 写回
    if (tid < N) out[tid] = r[0];
}

// 2. 有限制 Kernel：强制限制寄存器使用量
// 承诺每个 Block 最大 256 线程，且每个 SM 至少要跑 4 个 Block
// 编译器会被迫减少寄存器使用 (Spilling to Local Mem) 或重新调度指令
__global__ void __launch_bounds__(256, 4) heavy_kernel_bounded(float* out, int N) {
    int tid = threadIdx.x;
    float r[50];

    for(int i=0; i<50; ++i) r[i] = tid * 0.1f + i;

    #pragma unroll
    for(int k=0; k<100; ++k) {
        for(int i=0; i<50; ++i) {
            r[i] = r[i] * r[(i+1)%50] + 1.0f;
        }
    }

    if (tid < N) out[tid] = r[0];
}

int main() {
    printf("=== [Module A] 05. Kernel Structure & ABI Analysis ===\n\n");

    // --- Test 1: Alignment ---
    printf("[Host] Checking Structure Layout...\n");
    DangerousStruct ds = {'a', 42};
    SafeStruct ss = {'b', 100};

    printf("[Host]   DangerousStruct: Offset of b = %lu bytes\n", offsetof(DangerousStruct, b));
    printf("[Host]   SafeStruct:      Offset of b = %lu bytes\n", offsetof(SafeStruct, b));

    printf("[Host] Launching Alignment Kernel...\n");
    alignment_check_kernel<<<1, 1>>>(ds, ss);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");

    // --- Test 2: Inlining ---
    // 这个测试主要用于后续的 SASS 分析，运行时看不出区别
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(int)));
    test_noinline_kernel<<<1, 32>>>(d_out, 10);
    test_forceinline_kernel<<<1, 32>>>(d_out, 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[Host] Inlining kernels executed. Run '05_inspect_asm.sh' to see SASS differences.\n\n");

    // --- Test 3: Launch Bounds ---
    // 这个测试主要观察编译时的 Register Usage 输出 (CMake 中配置了 -Xptxas=-v)
    float *d_float_out;
    CUDA_CHECK(cudaMalloc(&d_float_out, 256 * sizeof(float)));

    heavy_kernel_default<<<1, 256>>>(d_float_out, 256);
    heavy_kernel_bounded<<<1, 256>>>(d_float_out, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[Host] Launch Bounds kernels executed.\n");
    printf("       CHECK YOUR COMPILE OUTPUT (Ninja log) for 'ptxas info' lines!\n");
    printf("       You should see different register counts for default vs bounded kernels.\n");

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_float_out));
    return 0;
}