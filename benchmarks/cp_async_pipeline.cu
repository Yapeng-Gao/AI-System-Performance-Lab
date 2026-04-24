#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define CUDA_CHECK(x) do { \
cudaError_t err = (x); \
if (err != cudaSuccess) { \
printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
exit(1); \
} \
} while (0)

template<int STAGES>
__global__ void cp_async_pipeline_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // CUDA 13 需要显式的 pipeline shared state + 所属 thread block 组
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pipe_state;
    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);
    float acc = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        pipe.producer_acquire();
        cuda::memcpy_async(
            block,
            &smem[s * blockDim.x + tid],
            &in[gid + s * blockDim.x],
            sizeof(float),
            pipe
        );
        pipe.producer_commit();

        pipe.consumer_wait();

        float x = smem[s * blockDim.x + tid];

#pragma unroll
        for (int k = 0; k < 32; ++k)
            acc = fmaf(x, acc, 1.0f);

        pipe.consumer_release();
    }

    out[gid] = acc;
}

int main() {
    const int N = 1 << 20;
    float *d_in, *d_out;

    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    dim3 block(256);
    dim3 grid(N / block.x);

    cp_async_pipeline_kernel<4>
        <<<grid, block, 4 * block.x * sizeof(float)>>>(d_in, d_out, N);

    cudaDeviceSynchronize();
    printf("[cp.async pipeline] done\n");
    return 0;
}
