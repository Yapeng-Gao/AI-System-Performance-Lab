#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#define CUDA_CHECK(x) do { \
cudaError_t err = (x); \
if (err != cudaSuccess) { \
printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
exit(1); \
} \
} while (0)

__global__ void attention_qk_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    float* __restrict__ out,
    int D
) {
    __shared__ half sQ[64][64];
    __shared__ half sK[64][64];

    int tid = threadIdx.x;
    int qid = blockIdx.x * 64 + tid;

    // Global -> Shared
    for (int d = tid; d < D; d += 64) {
        sQ[tid][d] = Q[qid * D + d];
        sK[tid][d] = K[qid * D + d];
    }
    __syncthreads();

    float acc = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d)
        acc += __half2float(sQ[tid][d]) * __half2float(sK[tid][d]);

    out[qid] = acc;
}

int main() {
    const int QN = 1 << 14;
    const int D  = 64;

    half *d_Q, *d_K;
    float *d_out;

    CUDA_CHECK(cudaMalloc(&d_Q, QN * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, QN * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, QN * sizeof(float)));

    dim3 block(64);
    dim3 grid(QN / 64);

    attention_qk_kernel<<<grid, block>>>(d_Q, d_K, d_out, D);
    cudaDeviceSynchronize();

    printf("[Attention QK] done\n");
    return 0;
}
