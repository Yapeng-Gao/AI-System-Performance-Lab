#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// 生产级错误检查宏
#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
cudaGetErrorString(err), __FILE__, __LINE__); \
throw std::runtime_error("CUDA Error"); \
} \
} while (0)

// 编译期检查架构 (用于 TMA 等特性)
#define ASPL_Hopper_ARCH 900
#define ASPL_Ampere_ARCH 800

namespace aspl {
    // 向上取整除法
    template <typename T>
    __host__ __device__ inline T ceil_div(T a, T b) {
        return (a + b - 1) / b;
    }
}