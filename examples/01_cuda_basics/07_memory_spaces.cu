/**
 * [Module A] 07. Memory spaces + UVA
 *
 * A: print pointers (global / __device__ / shared / address-taken local / mapped host)
 *    and read one mapped int (PASS). Address-taken locals go to Local, not on-chip SRAM.
 * B: force spill; print cudaFuncGetAttributes.localSizeBytes (must be > 0).
 * D: compile two add kernels (with/without __restrict__). Contract demo, not a speedup.
 * C (mapped vs device event): not in this binary — bandwidth is B-06.
 * Not CUDA event kernel time (A-08). Spill prescriptions: B-03. UM: B-05.
 */

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

__device__ int g_device_var = 42;

__global__ void address_space_probe(int* d_ptr, int* h_mapped_ptr, int* d_flag) {
    __shared__ int s_var;
    int l_var = 10;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        s_var = 1;
        printf("[Device] === address map (VA, not a wiring diagram) ===\n");
        printf("  cudaMalloc global:     %p\n", static_cast<void*>(d_ptr));
        printf("  __device__ global:     %p\n", static_cast<void*>(&g_device_var));
        printf("  __shared__:            %p\n", static_cast<void*>(&s_var));
        printf("  address-taken local:   %p  (compiler puts this in Local / HBM)\n",
               static_cast<void*>(&l_var));
        printf("  mapped host ptr:       %p\n", static_cast<void*>(h_mapped_ptr));

        const int val = *h_mapped_ptr;
        const int ok = (val == 999) ? 1 : 0;
        *d_flag = ok;
        printf("[Device] mapped host read: %d  expected 999\n", val);
    }
}

__global__ void force_local_memory_spill(float* out, int n, int salt) {
    const int tid = static_cast<int>(threadIdx.x);
    float local_buffer[256];
    for (int i = 0; i < 256; ++i) {
        local_buffer[i] = static_cast<float>(tid + i + salt);
    }
    float acc = 0.0f;
    for (int i = 0; i < 256; ++i) {
        const int idx = (tid + i + salt) & 255;
        acc += local_buffer[idx];
        local_buffer[idx] = acc;
    }
    if (tid < n) {
        out[tid] = acc;
    }
}

__global__ void add_no_restrict(float* a, float* b, float* c, int n) {
    const int idx = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_with_restrict(float* __restrict__ a,
                                  float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    const int idx = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int uva = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, 0));
    printf("[Host] GPU: %s\n", prop.name);
    printf("[Host] Compute Capability: %d.%d  sm_%d%d\n",
           prop.major, prop.minor, prop.major, prop.minor);
    printf("[Host] UnifiedAddressing: %s\n", uva ? "yes" : "no");
    printf("[Host] Not CUDA event kernel time (see A-08).\n");
    printf("[Host] Mapped throughput is B-06; UM fault/prefetch is B-05.\n\n");

    int* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, sizeof(int)));

    int* h_mapped = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_mapped, sizeof(int), cudaHostAllocMapped));
    *h_mapped = 999;

    int* d_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));

    address_space_probe<<<1, 1>>>(d_ptr, h_mapped, d_flag);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_flag = 0;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[Host] mapped read: %s\n\n", h_flag ? "PASS" : "FAIL");
    if (!h_flag) {
        fprintf(stderr, "[Host] mapped host pointer was not readable from the kernel.\n");
        return 1;
    }

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));
    force_local_memory_spill<<<1, 256>>>(d_out, 256, 7);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(force_local_memory_spill)));
    printf("[Host] force_local_memory_spill: regs=%d localSizeBytes=%zu\n",
           attr.numRegs, attr.localSizeBytes);
    if (attr.localSizeBytes == 0) {
        fprintf(stderr,
                "[Host] FAIL: expected localSizeBytes > 0 (array should spill to Local/HBM).\n");
        return 1;
    }
    printf("[Host] spill: PASS  localSizeBytes > 0  (Local is thread-private, physically HBM)\n");
    printf("[Host] SASS: cuobjdump -sass <bin>  then grep/findstr LDL STL\n");
    printf("[Host] How to unspill: B-03. Not this chapter.\n\n");

    const int n = 256;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
    add_no_restrict<<<1, n>>>(da, db, dc, n);
    add_with_restrict<<<1, n>>>(da, db, dc, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[Host] restrict kernels ran (aliasing contract). Not a speedup bill.\n");
    printf("[Host] Do not treat LDG.NC / LDG.128 as guaranteed SASS.\n");

    CUDA_CHECK(cudaFree(d_ptr));
    CUDA_CHECK(cudaFree(d_flag));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFreeHost(h_mapped));
    return 0;
}
