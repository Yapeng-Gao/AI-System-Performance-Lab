/**
 * [Module A] 05. Kernel ABI: params, alignment, inline, launch_bounds
 *
 * Prints Host vs Device offsetof/sizeof (same header), then
 * cudaFuncGetAttributes + occupancy API for default vs __launch_bounds__.
 * Inlining is for cuobjdump (CALL / c[0x0]); not a wall-clock test.
 * Not CUDA event kernel time (A-08). Spill prescriptions: B-03.
 */

#include <cstddef>
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

struct DefaultLayout {
    char a;
    int b;
};

struct AlignedLayout {
    alignas(4) char a;
    int b;
};

#pragma pack(push, 1)
struct PackedLayout {
    char a;
    int b;
};
#pragma pack(pop)

__global__ void alignment_kernel(DefaultLayout d, AlignedLayout al, PackedLayout p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[Device] DefaultLayout: offsetof(b)=%zu sizeof=%zu  b=%d\n",
               offsetof(DefaultLayout, b), sizeof(DefaultLayout), d.b);
        printf("[Device] AlignedLayout: offsetof(b)=%zu sizeof=%zu  b=%d\n",
               offsetof(AlignedLayout, b), sizeof(AlignedLayout), al.b);
        printf("[Device] PackedLayout:  offsetof(b)=%zu sizeof=%zu  b=%d\n",
               offsetof(PackedLayout, b), sizeof(PackedLayout), p.b);
    }
}

__device__ __noinline__ int math_noinline(int x) {
    return x * x + 3 * x;
}

__device__ __forceinline__ int math_forceinline(int x) {
    return x * x + 3 * x;
}

__global__ void test_noinline_kernel(int* out, int x) {
    out[threadIdx.x] = math_noinline(x + static_cast<int>(threadIdx.x));
}

__global__ void test_forceinline_kernel(int* out, int x) {
    out[threadIdx.x] = math_forceinline(x + static_cast<int>(threadIdx.x));
}

__global__ void heavy_kernel_default(float* out, int n) {
    float r[50];
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = 0; i < 50; ++i) {
        r[i] = tid * 0.1f + static_cast<float>(i);
    }
    #pragma unroll
    for (int k = 0; k < 100; ++k) {
        for (int i = 0; i < 50; ++i) {
            r[i] = r[i] * r[(i + 1) % 50] + 1.0f;
        }
    }
    if (tid < n) {
        out[tid] = r[0];
    }
}

__global__ void __launch_bounds__(256, 4) heavy_kernel_bounded(float* out, int n) {
    float r[50];
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = 0; i < 50; ++i) {
        r[i] = tid * 0.1f + static_cast<float>(i);
    }
    #pragma unroll
    for (int k = 0; k < 100; ++k) {
        for (int i = 0; i < 50; ++i) {
            r[i] = r[i] * r[(i + 1) % 50] + 1.0f;
        }
    }
    if (tid < n) {
        out[tid] = r[0];
    }
}

static void print_layout(const char* tag, size_t off, size_t sz) {
    printf("[Host]   %-14s offsetof(b)=%zu sizeof=%zu\n", tag, off, sz);
}

static void print_kernel_abi(const char* name, const void* fn, int block_size) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, block_size, 0));
    printf("[Host] %s: regs=%d local=%zuB  occupancy=%d blocks/SM @ %d threads\n",
           name, attr.numRegs, attr.localSizeBytes, blocks, block_size);
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[Host] GPU: %s\n", prop.name);
    printf("[Host] Compute Capability: %d.%d  sm_%d%d\n",
           prop.major, prop.minor, prop.major, prop.minor);
    printf("[Host] Not CUDA event kernel time (see A-08).\n\n");

    DefaultLayout d = {'a', 42};
    AlignedLayout al = {'b', 100};
    PackedLayout p = {'c', 7};

    printf("[Host] Same header, three layouts:\n");
    print_layout("DefaultLayout", offsetof(DefaultLayout, b), sizeof(DefaultLayout));
    print_layout("AlignedLayout", offsetof(AlignedLayout, b), sizeof(AlignedLayout));
    print_layout("PackedLayout", offsetof(PackedLayout, b), sizeof(PackedLayout));

    alignment_kernel<<<1, 1>>>(d, al, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[Host] Layouts must match Host vs Device. Bug is two definitions, not 'NVCC pads randomly'.\n\n");

    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(int)));
    test_noinline_kernel<<<1, 32>>>(d_out, 10);
    test_forceinline_kernel<<<1, 32>>>(d_out, 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[Host] Inline kernels ran. SASS: cuobjdump -sass <bin>  (CALL / c[0x0]).\n\n");

    float* d_float = nullptr;
    CUDA_CHECK(cudaMalloc(&d_float, 256 * sizeof(float)));
    heavy_kernel_default<<<1, 256>>>(d_float, 256);
    heavy_kernel_bounded<<<1, 256>>>(d_float, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_kernel_abi("heavy_default",
                     reinterpret_cast<const void*>(heavy_kernel_default), 256);
    print_kernel_abi("heavy_bounded",
                     reinterpret_cast<const void*>(heavy_kernel_bounded), 256);
    printf("[Host] launch_bounds is an occupancy contract. Spill fix is B-03.\n");

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_float));
    return 0;
}
