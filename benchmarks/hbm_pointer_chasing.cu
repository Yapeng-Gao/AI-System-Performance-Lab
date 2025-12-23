#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CUDA_CHECK(x) do { \
cudaError_t err = (x); \
if (err != cudaSuccess) { \
printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
exit(1); \
} \
} while (0)

__global__ void pointer_chasing_kernel(
    const uint32_t* __restrict__ index,
    const float*    __restrict__ data,
    float*          out,
    int iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = tid;

    float acc = 0.f;
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        idx = index[idx];     // 强制 global 依赖
        acc += data[idx];     // 无法合并、无法预取
    }
    out[tid] = acc;
}

int main() {
    const int N = 1 << 26;          // 256MB 级工作集，远大于 96MB L2
    const int iters = 64;

    uint32_t *d_index;
    float *d_data, *d_out;

    CUDA_CHECK(cudaMalloc(&d_index, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_data,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,   N * sizeof(float)));

    // index 初始化为随机置换（host 侧做即可）
    uint32_t* h_index = new uint32_t[N];
    for (uint32_t i = 0; i < N; ++i)
        h_index[i] = (i * 1315423911u) & (N - 1);

    CUDA_CHECK(cudaMemcpy(d_index, h_index, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(N / block.x);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    pointer_chasing_kernel<<<grid, block>>>(d_index, d_data, d_out, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    pointer_chasing_kernel<<<grid, block>>>(d_index, d_data, d_out, iters);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);

    double bytes = double(iters) * N * sizeof(float);
    double gbps = bytes / 1e9 / (ms / 1e3);

    printf("[HBM Pointer Chasing] %.2f GB/s\n", gbps);

    return 0;
}
