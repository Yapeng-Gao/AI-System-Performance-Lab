/**
 * [Module A] 08. 异步执行模型
 * Pipeline Concurrency: 实现 H2D -> Compute -> D2H 三级流水线
 *
 * 验证目标：
 * 1. Pinned Memory (cudaMallocHost) 对异步传输的必要性
 * 2. 多 Stream 并发对掩盖 PCIe 延迟的作用
 * 3. Depth-First 调度策略的优势
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
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

// --- 模拟负载 Kernel ---
// 使用 clock64() 进行忙等待，模拟重计算任务 (Compute Bound)
// 这样可以确保 Kernel 运行足够长的时间，以便在 Profiler 中观察到 Overlap
__global__ void heavy_compute_kernel(float* data, int n, long long delay_clocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];

        // 模拟计算负载
        long long start = clock64();
        while (clock64() - start < delay_clocks) {
            // 做一些无意义的计算防止被编译器优化掉
            val = sinf(val) * cosf(val);
        }

        // 简单的写回
        data[idx] = val + 1.0f;
    }
}

// ==========================================
// Mode 1: 串行基准 (Pageable Memory + Default Stream)
// ==========================================
void run_serial_baseline(int total_elements, int chunk_size, long long kernel_load) {
    size_t total_bytes = total_elements * sizeof(float);
    size_t chunk_bytes = chunk_size * sizeof(float);
    int num_chunks = total_elements / chunk_size;

    // 1. 分配 Pageable Memory (普通 malloc)
    // 这会导致驱动必须介入进行临时 Pinned Buffer 拷贝，无法异步
    float *h_data = (float*)malloc(total_bytes);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));

    // 初始化
    for(int i=0; i<total_elements; ++i) h_data[i] = 1.0f;

    printf("[Serial] Starting processing %d chunks...\n", num_chunks);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_chunks; ++i) {
        int offset = i * chunk_size;

        // 即使写了 cudaMemcpyAsync，对于 Pageable Memory 也会退化为同步
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes,
                                   cudaMemcpyHostToDevice, 0));

        heavy_compute_kernel<<<chunk_size / 256, 256, 0, 0>>>(d_data + offset, chunk_size, kernel_load);

        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes,
                                   cudaMemcpyDeviceToHost, 0));
    }

    CUDA_CHECK(cudaDeviceSynchronize()); // 确保全部完成

    auto end_time = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    printf("[Serial] Total Time: %.2f ms\n", ms);

    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
}

// ==========================================
// Mode 2: 异步流水线 (Pinned Memory + Multi-Streams)
// ==========================================
void run_async_pipeline(int total_elements, int chunk_size, long long kernel_load, int n_streams) {
    size_t total_bytes = total_elements * sizeof(float);
    size_t chunk_bytes = chunk_size * sizeof(float);
    int num_chunks = total_elements / chunk_size;

    // 1. 分配 Pinned Memory (页锁定内存)
    // 这是 DMA 引擎能直接访问的前提
    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, total_bytes)); // Pinned

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));

    // 初始化
    for(int i=0; i<total_elements; ++i) h_data[i] = 1.0f;

    // 2. 创建 Streams
    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; ++i) {
        // cudaStreamNonBlocking: 避免与 Legacy Default Stream 发生隐式同步
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    printf("[Pipeline] Starting processing %d chunks with %d streams...\n", num_chunks, n_streams);
    auto start_time = std::chrono::high_resolution_clock::now();

    // 3. 调度循环 (Depth-First Launch)
    // 这种模式能最大化 Overlap：当 Stream 0 在计算时，Stream 1 在拷贝
    for (int i = 0; i < num_chunks; ++i) {
        int stream_idx = i % n_streams;
        int offset = i * chunk_size;

        // Stage 1: H2D Copy
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes,
                                   cudaMemcpyHostToDevice, streams[stream_idx]));

        // Stage 2: Compute
        heavy_compute_kernel<<<chunk_size / 256, 256, 0, streams[stream_idx]>>>
                            (d_data + offset, chunk_size, kernel_load);

        // Stage 3: D2H Copy
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes,
                                   cudaMemcpyDeviceToHost, streams[stream_idx]));
    }

    // 4. 同步
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    printf("[Pipeline] Total Time: %.2f ms\n", ms);

    // 清理
    for (int i = 0; i < n_streams; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_data));
}

int main() {
    // 参数设置
    // 总数据量: 32MB float
    const int TOTAL_ELEMENTS = 8 * 1024 * 1024;
    // 切块大小: 256KB float (切得太小会导致 Launch Overhead 占比过高，切得太大 Overlap 效果差)
    const int CHUNK_SIZE = 256 * 1024;
    // 模拟计算负载: 约 100000 个时钟周期 (调整此值以平衡 Compute 与 Copy 时间)
    // 理想的 Overlap 是 Compute Time ≈ Copy Time
    const long long KERNEL_LOAD = 100000;
    const int N_STREAMS = 4;

    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("GPU: %s\n", prop.name);
    printf("Data Size: %.2f MB, Chunk Size: %.2f MB\n\n",
           TOTAL_ELEMENTS * sizeof(float) / 1024.0 / 1024.0,
           CHUNK_SIZE * sizeof(float) / 1024.0 / 1024.0);

    // 运行对比
    run_serial_baseline(TOTAL_ELEMENTS, CHUNK_SIZE, KERNEL_LOAD);
    printf("------------------------------------------------\n");
    run_async_pipeline(TOTAL_ELEMENTS, CHUNK_SIZE, KERNEL_LOAD, N_STREAMS);

    return 0;
}