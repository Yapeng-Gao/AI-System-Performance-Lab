/**
 * [Module A] 09. Compute Sanitizer bug generator
 *
 * Intentional bugs for compute-sanitizer (not a performance bench):
 *   0  OOB write              → --tool memcheck
 *   1  SMEM data race         → --tool racecheck
 *   2  illegal __syncwarp mask → --tool synccheck
 *
 * Mode 2 uses NVIDIA sample-style Invalid arguments (thread reaches
 * __syncwarp but is not in the mask). Classic divergent __syncthreads
 * returned 0 errors on RTX 5090 / sm_120 in our first runs.
 *
 * initcheck: optional; not in this binary. Overlap: A-08. Roofline: A-10.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Bug: valid indices are 0..n-1; thread 0 writes data[n].
__global__ void oob_kernel(int* data, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        data[n] = 42;
    }
}

// Bug: many threads RMW the same SMEM location without atomics/sync between RMWs.
__global__ void race_kernel(int* out) {
    __shared__ int s_val;
    if (threadIdx.x == 0) {
        s_val = 0;
    }
    __syncthreads();

    s_val += 1;

    __syncthreads();
    if (threadIdx.x == 0) {
        *out = s_val;
    }
}

// Bug: threads 0..16 reach __syncwarp, but mask only enables 0..15 (thread 16 missing).
// Official synccheck class: Invalid arguments. (illegal_syncwarp sample pattern)
__global__ void illegal_sync_kernel(int* out) {
    __shared__ int smem[32];
    const int tx = static_cast<int>(threadIdx.x);

    if (tx < 17) {
        smem[tx] = tx;
        const unsigned mask = 0x0000ffffu;
        __syncwarp(mask);
    }
    if (tx == 0) {
        *out = smem[0];
    }
}

static void print_usage(const char* argv0) {
    printf("Usage: %s <mode>\n", argv0);
    printf("  0  OOB write                 → compute-sanitizer --tool memcheck\n");
    printf("  1  SMEM race                 → compute-sanitizer --tool racecheck\n");
    printf("  2  illegal __syncwarp mask   → compute-sanitizer --tool synccheck\n");
    printf("Bare run may not crash; hang sanitizer to attribute the bug.\n");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    const int mode = atoi(argv[1]);
    if (mode < 0 || mode > 2) {
        print_usage(argv[0]);
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("sm_%d%d\n", prop.major, prop.minor);
    printf("[Host] Bug generator mode %d (correctness demo, not a timing bench)\n",
           mode);

    if (mode == 0) {
        constexpr int N = 256;
        int* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, static_cast<size_t>(N) * sizeof(int)));
        printf("[Host] Planted: oob_kernel thread0 writes data[N] on length-N buffer\n");
        printf("[Host] Launch: <<<1,32>>> (valid blockDim; OOB is the store, not the grid)\n");
        printf("[Host] Expect: memcheck Invalid __global__ write\n");
        oob_kernel<<<1, 32>>>(d_data, N);
        const cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            printf("[Host] After sync: %s (expected after OOB; sticky error follows)\n",
                   cudaGetErrorString(sync_err));
            // Fatal sticky error: do not CUDA_CHECK free — just best-effort cleanup.
            (void)cudaFree(d_data);
            return 0;
        }
        printf("[Host] Sync returned success (unexpected without sanitizer crash path)\n");
        CUDA_CHECK(cudaFree(d_data));
    } else if (mode == 1) {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
        printf("[Host] Planted: race_kernel SMEM s_val += 1 without atomics\n");
        printf("[Host] Expect: racecheck Hazard (often Warning, not ERROR)\n");
        race_kernel<<<1, 32>>>(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        int h_out = 0;
        CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[Host] Race kernel finished; out=%d (undefined if raced)\n", h_out);
        CUDA_CHECK(cudaFree(d_out));
    } else {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
        printf("[Host] Planted: illegal_sync_kernel threads 0..16 call __syncwarp(mask=0xffff)\n");
        printf("[Host] Expect: synccheck Barrier error / Invalid arguments\n");
        illegal_sync_kernel<<<1, 32>>>(d_out);
        const cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            printf("[Host] After sync: %s\n", cudaGetErrorString(sync_err));
            (void)cudaFree(d_out);
            return 0;
        }
        printf("[Host] Sync returned success (use synccheck to see the hazard)\n");
        CUDA_CHECK(cudaFree(d_out));
    }

    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        printf("[Host] cudaGetLastError: %s\n", cudaGetErrorString(last));
    }

    return 0;
}
