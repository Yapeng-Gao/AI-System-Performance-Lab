#include <gtest/gtest.h>
#include "aspl/common/cuda_utils.h"
#include "kernels/cuda/math/reduction_kernel.cuh" // 假设内部头文件

TEST(ReductionTest, SumBasic) {
    int N = 1024;
    float *d_data, *d_out;
    // ... cudaMalloc & Memcpy ...

    // 调用库中的 kernel
    aspl::kernels::launch_reduce_sum(d_data, d_out, N);

    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_out, 1024.0f, 1e-5);
    // ... cudaFree ...
}