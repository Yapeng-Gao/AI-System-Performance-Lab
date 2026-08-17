/**
 * [Module A] 03. CUDA 编程模型物理映射
 * Grid tracer: %smid + occupancy API. Not a kernel timer.
 *
 * 口径：
 * - atomic 序号 = block 内 thread 0 抢到的顺序，不是 GTE 派发日志
 * - clock64 busy-wait 只为把 Block 拉长，不是 event median
 * - Grid 用 cudaOccupancyMaxActiveBlocksPerMultiprocessor，不猜「每 SM 4 个 Block」
 */

#include <cstdio>
#include <cstdint>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__global__ void scheduler_tracer_kernel(
    int* d_block_to_sm,
    int* d_execution_order,
    int* d_global_counter,
    int delay_iters
) {
    if (threadIdx.x == 0) {
        int bid = blockIdx.x;
        d_block_to_sm[bid] = (int)get_smid();
        d_execution_order[bid] = atomicAdd(d_global_counter, 1);

        long long start_clock = clock64();
        while (clock64() - start_clock < delay_iters) {
        }
    }
}

int main() {
    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int num_sms = prop.multiProcessorCount;
    printf("[Host] GPU: %s\n", prop.name);
    printf("[Host] Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("[Host] SM count: %d\n", num_sms);

    const int block_size = 1;
    const int dyn_smem = 0;
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, scheduler_tracer_kernel, block_size, dyn_smem));
    if (blocks_per_sm <= 0) {
        fprintf(stderr, "[Host] occupancy API returned %d blocks/SM\n", blocks_per_sm);
        return 1;
    }

    const int total_waves = 5;
    const int wave = num_sms * blocks_per_sm;
    const int num_blocks = wave * total_waves + 1;  // +1 tail

    printf("[Host] Occupancy (this kernel, blockSize=%d): %d blocks/SM\n",
           block_size, blocks_per_sm);
    printf("[Host] Wave size ≈ %d blocks; launching %d blocks (%d waves + 1 tail)\n",
           wave, num_blocks, total_waves);
    printf("[Host] Note: <<<N,1>>> occupancy is NOT a 256-thread + SMEM kernel.\n");

    int *h_block_to_sm = new int[num_blocks];
    int *h_execution_order = new int[num_blocks];
    int *d_block_to_sm = nullptr;
    int *d_execution_order = nullptr;
    int *d_global_counter = nullptr;

    CUDA_CHECK(cudaMalloc(&d_block_to_sm, num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_execution_order, num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_global_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));

    const int delay_cycles = 100000;
    scheduler_tracer_kernel<<<num_blocks, block_size>>>(
        d_block_to_sm, d_execution_order, d_global_counter, delay_cycles);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_block_to_sm, d_block_to_sm,
                          num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_execution_order, d_execution_order,
                          num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    std::map<int, int> sm_usage;
    int last_block = -1;
    int max_order = -1;
    for (int i = 0; i < num_blocks; ++i) {
        sm_usage[h_block_to_sm[i]]++;
        if (h_execution_order[i] > max_order) {
            max_order = h_execution_order[i];
            last_block = i;
        }
    }

    int min_cnt = num_blocks;
    int max_cnt = 0;
    for (int sm = 0; sm < num_sms; ++sm) {
        int c = sm_usage[sm];
        min_cnt = std::min(min_cnt, c);
        max_cnt = std::max(max_cnt, c);
    }

    printf("\n[Analysis 1] Blocks finished per SM (min=%d max=%d; first 5 SMs):\n",
           min_cnt, max_cnt);
    for (int sm = 0; sm < 5 && sm < num_sms; ++sm) {
        printf("  SM %02d : %d blocks\n", sm, sm_usage[sm]);
    }

    printf("\n[Analysis 2] Tail (atomic order, NOT a GTE log):\n");
    printf("  Last numbered logical Block %d ran on SM %d (order=%d)\n",
           last_block, h_block_to_sm[last_block], max_order);

    printf("\n[Visualizer] logical Block -> SM (first 64):\n");
    for (int i = 0; i < 64 && i < num_blocks; ++i) {
        if (i % 16 == 0) printf("\n  Blocks %03d-%03d: ", i, i + 15);
        printf("%3d ", h_block_to_sm[i]);
    }
    printf("\n\n");

    if ((int)sm_usage.size() == num_sms) {
        printf("[Conclusion] Every SM received at least one Block.\n");
    } else {
        printf("[Conclusion] Only %zu / %d SMs saw a Block. Check device/grid.\n",
               sm_usage.size(), num_sms);
    }

    delete[] h_block_to_sm;
    delete[] h_execution_order;
    CUDA_CHECK(cudaFree(d_block_to_sm));
    CUDA_CHECK(cudaFree(d_execution_order));
    CUDA_CHECK(cudaFree(d_global_counter));
    return 0;
}
