/**
 * [Module A] 01. CUDA 核心概念总览
 * 现代版 Hello World：展示 CUDA 12+ 的工程规范
 */

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

// --- 1. 生产级错误检查宏 (必背) ---
// 在 System Engineering 中，Fail Fast 是铁律。
// 任何 CUDA API 返回非 cudaSuccess 时，立即打印文件、行号并退出。
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- 2. Kernel 定义 ---
// __global__ : Host 调用, Device 执行
__global__ void hello_kernel(int* d_data)
{
    // 物理坐标计算:
    // Grid(网格) -> Block(线程块) -> Thread(线程)
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // 全局唯一索引计算 (1D)
    int global_id = bid * blockDim.x + tid;

    // 简单的计算任务
    d_data[global_id] = global_id * 10;

    // --- 硬件感知 ---
    // 仅由第0个线程打印，避免输出爆炸
    // __CUDA_ARCH__ 是编译器宏，代表当前 GPU 的架构版本 (如 800 代表 Ampere, 900 代表 Hopper)
    if (global_id == 0)
    {
#ifdef __CUDA_ARCH__          // 只在 device 编译时可见
        printf("[Device] Kernel running on SM arch sm_%d\n", __CUDA_ARCH__);
#endif
        printf("[Device] GridDim=%d, BlockDim=%d\n", gridDim.x, blockDim.x);
    }
}

int main()
{
    std::cout << "[Host] Starting Modern CUDA Hello World..." << std::endl;

    // --- 3. 硬件查询 (Device Query) ---
    // 在大规模集群中，这一步用于确认你拿到了正确的卡
    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "[Host] GPU Name: " << prop.name << std::endl;
    std::cout << "[Host] SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "[Host] Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // --- 4. 资源分配 (Unified Memory) ---
    // 使用 cudaMallocManaged，数据会在 CPU/GPU 间自动按需迁移 (Page Migration)
    // 这比传统的 cudaMalloc + cudaMemcpy 更符合现代编程范式（虽然手动管理性能更好）
    const int num_elements = 32;
    size_t bytes = num_elements * sizeof(int);
    int* data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));

    // --- 5. 启动 Kernel ---
    // 配置: 1个 Block, 32个 Threads (刚好一个 Warp)
    // 注意: Kernel Launch 是异步的！CPU 发完指令立刻往下走
    std::cout << "[Host] Launching Kernel..." << std::endl;
    hello_kernel<<<1, 32>>>(data);

    // --- 6. 错误检查与同步 ---
    // 捕获 Launch 配置错误 (如 Block 数超限)
    CUDA_CHECK(cudaGetLastError());

    // 强制 CPU 等待 GPU 完成。
    // 如果没有这一行，CPU 可能会在 GPU 还没算完时就去读取 data，导致读到脏数据。
    // 同时，printf 的缓冲区也需要同步才能刷出到屏幕。
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 7. 结果验证 ---
    // 直接在 Host 端读取 Unified Memory
    std::cout << "[Host] Verifying results..." << std::endl;
    bool passed = true;
    for (int i = 0; i < num_elements; i++)
    {
        if (data[i] != i * 10)
        {
            printf("Error at %d: expected %d, got %d\n", i, i * 10, data[i]);
            passed = false;
            break;
        }
    }

    if (passed)
    {
        std::cout << "[Host] Verification PASSED! [OK]" << std::endl;
    }
    else
    {
        std::cout << "[Host] Verification FAILED! [FAIL]" << std::endl;
    }

    // 释放内存
    CUDA_CHECK(cudaFree(data));

    return 0;
}
