/**
 * [Module A] 03. CUDA 编程模型物理映射
 * Grid Mapper: 可视化 GigaThread Engine 的调度逻辑与物理 SM 映射
 *
 * 技术点：
 * 1. Inline PTX 读取硬件寄存器 %smid
 * 2. Global Atomic 追踪真实的执行顺序 (Execution Order)
 * 3. 模拟负载以观察 Wavefront (波次) 效应
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdio>
#include <cuda_runtime.h>

// --- 1. 生产级错误检查宏 ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- 2. Device 端辅助函数: 读取物理 SM ID ---
// 使用 PTX (Parallel Thread Execution) 汇编直接读取特殊寄存器 %smid
// 这是一个比 CUDA C++ API 更底层的操作，兼容所有架构
__device__ __forceinline__ uint32_t get_smid() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

// --- 3. Kernel 定义 ---
// 记录每个 Block 被分配到了哪个 SM，以及它是第几个开始执行的
__global__ void scheduler_tracer_kernel(
    int* d_block_to_sm,      // 输出: Block ID -> SM ID
    int* d_execution_order,  // 输出: Block ID -> Order (第几个被调度)
    long long* d_start_time, // 输出: Block ID -> Start Timestamp
    int* d_global_counter,   // 全局计数器 (用于生成 Order)
    int delay_iters          // 模拟计算负载，拉长执行时间以形成 Wave
) {
    // 我们只需要 Block 中的第一个线程来记录信息 (代表整个 Block)
    if (threadIdx.x == 0) {
        int bid = blockIdx.x;

        // 1. 获取物理 SM ID
        uint32_t sm_id = get_smid();
        d_block_to_sm[bid] = (int)sm_id;

        // 2. 获取全局执行顺序 (Atomic 操作是序列化的)
        // 这代表了 GigaThread Engine 实际激活 Block 的顺序
        d_execution_order[bid] = atomicAdd(d_global_counter, 1);

        // 3. 记录开始时间 (Cycle Count)
        d_start_time[bid] = clock64();

        // 4. 模拟负载 (Busy Wait)
        // 如果 Kernel 跑得太快，所有 Block 瞬间完成，就看不到 "Wave" 效应了
        // 这里强制让 Block 占用 SM 一段时间
        long long start_clock = clock64();
        while (clock64() - start_clock < delay_iters) {
            // Busy loop
        }
    }
}

int main() {
    std::cout << "[Host] Starting Grid Scheduler Tracer..." << std::endl;

    // --- 1. 获取设备信息 ---
    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int num_sms = prop.multiProcessorCount;
    printf("[Host] GPU: %s, Total SMs: %d\n", prop.name, num_sms);

    // --- 2. 配置 Grid ---
    // 为了观察 Wave 效应，我们需要启动比 SM 数量多得多的 Block
    // 假设每个 SM 能并发跑 4 个 Block (具体取决于资源限制)，我们发射 4 波 (Waves)
    int blocks_per_sm_capacity = 4; // 这是一个估计值，用于制造负载
    int total_waves = 5;
    int num_blocks = num_sms * blocks_per_sm_capacity * total_waves;

    // 加上一个 "Tail" (尾巴)，演示 Tail Effect (最后一个 Block 独占 GPU)
    num_blocks += 1;

    printf("[Host] Launching %d Blocks (approx %d full waves + 1 tail)\n", num_blocks, total_waves * blocks_per_sm_capacity);

    // --- 3. 内存分配 ---
    size_t size_map = num_blocks * sizeof(int);
    size_t size_time = num_blocks * sizeof(long long);

    int *h_block_to_sm = new int[num_blocks];
    int *h_execution_order = new int[num_blocks];
    long long *h_start_time = new long long[num_blocks];

    int *d_block_to_sm, *d_execution_order, *d_global_counter;
    long long *d_start_time;

    CUDA_CHECK(cudaMalloc(&d_block_to_sm, size_map));
    CUDA_CHECK(cudaMalloc(&d_execution_order, size_map));
    CUDA_CHECK(cudaMalloc(&d_start_time, size_time));
    CUDA_CHECK(cudaMalloc(&d_global_counter, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));

    // --- 4. 启动 Kernel ---
    // 模拟约 100,000 cycles 的负载 (取决于主频，约 50-100us)
    int delay_cycles = 100000;

    scheduler_tracer_kernel<<<num_blocks, 1>>>(
        d_block_to_sm,
        d_execution_order,
        d_start_time,
        d_global_counter,
        delay_cycles
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 5. 回传数据 ---
    CUDA_CHECK(cudaMemcpy(h_block_to_sm, d_block_to_sm, size_map, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_execution_order, d_execution_order, size_map, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_start_time, d_start_time, size_time, cudaMemcpyDeviceToHost));

    // --- 6. 结果分析与可视化 (Host Side Analysis) ---

    // 分析 A: 物理 SM 的负载均衡性 (Round-Robin 验证)
    std::map<int, int> sm_usage;
    for (int i = 0; i < num_blocks; i++) {
        sm_usage[h_block_to_sm[i]]++;
    }

    printf("\n[Analysis 1] SM Load Balance (Top 5 & Bottom 5):\n");
    // ... (省略复杂的排序代码，直接打印部分 SM)
    for (int i = 0; i < 5 && i < num_sms; i++) {
        printf("  SM %02d processed %d blocks\n", i, sm_usage[i]);
    }
    printf("  ...\n");

    // 分析 B: 尾部效应 (Tail Effect)
    // 找到执行顺序最后的那个 Block
    int last_executed_block_idx = -1;
    int max_order = -1;

    for(int i=0; i<num_blocks; i++) {
        if(h_execution_order[i] > max_order) {
            max_order = h_execution_order[i];
            last_executed_block_idx = i;
        }
    }

    printf("\n[Analysis 2] Tail Effect Detection:\n");
    printf("  The very last block to run was logical Block %d\n", last_executed_block_idx);
    printf("  It ran on physical SM %d\n", h_block_to_sm[last_executed_block_idx]);
    printf("  Note: While this block was running, other SMs might have been IDLE if the grid size wasn't aligned to waves.\n");

    // 分析 C: 简单的 ASCII 调度图 (Visualizing First 100 Blocks)
    printf("\n[Visualizer] Logical Block ID -> Physical SM ID (First 64 Blocks):\n");
    for (int i = 0; i < 64; i++) {
        if (i % 16 == 0) printf("\n  Blocks %03d-%03d: ", i, i+15);
        printf("%3d ", h_block_to_sm[i]);
    }
    printf("\n\n");

    // 验证逻辑: 检查是否所有 SM 都至少工作了
    bool all_sms_active = sm_usage.size() == num_sms;
    if (all_sms_active) {
        printf("[Conclusion] GigaThread Engine successfully distributed work across ALL %d SMs. \u2705\n", num_sms);
    } else {
        printf("[Conclusion] Some SMs were idle! (Check grid size or device status) \u274C\n");
    }

    // 资源清理
    delete[] h_block_to_sm;
    delete[] h_execution_order;
    delete[] h_start_time;
    CUDA_CHECK(cudaFree(d_block_to_sm));
    CUDA_CHECK(cudaFree(d_execution_order));
    CUDA_CHECK(cudaFree(d_start_time));
    CUDA_CHECK(cudaFree(d_global_counter));

    return 0;
}