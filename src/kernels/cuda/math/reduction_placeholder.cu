#include <cuda_runtime.h>
#include <cstdio>

namespace aspl {
    namespace kernels {

        __global__ void placeholder_kernel() {
            printf("Hello from ASPL Kernel!\n");
        }

        void launch_placeholder() {
            placeholder_kernel<<<1, 1>>>();
            cudaDeviceSynchronize();
        }

    }
}