/**
 * [Module A] 10. 性能建模第一性原理
 * Roofline Empirical Prober: 实测硬件的带宽极限 (Bandwidth) 与算力极限 (FLOPs)
 *
 * 原理：
 * 1. Bandwidth Kernel: 使用 float4 向量化读写，最大化内存事务效率。
 * 2. Compute Kernel: 使用寄存器级 FMA (Fused Multiply-Add) 密集计算，隐藏所有延迟。
 */

#include <iostream>
#include <cstdio>
#include <vector>
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

// --- Helper: 计算理论峰值 ---
void print_theoretical_peaks(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("----------------------------------------------------------------\n");
    printf("[Theoretical Peaks] Device: %s (SMs: %d)\n", prop.name, prop.multiProcessorCount);

    // 1. 理论带宽 (GB/s)
    // Memory Clock (KHz) * Bus Width (bits) * 2 (DDR) / 8 (bits->bytes) / 1e9
#if defined(CUDA_VERSION) && CUDA_VERSION < 12000
    // CUDA 11.x 及更早版本
    double mem_clock_ghz = prop.memoryClockRate / 1e6;
    double bus_width_bytes = prop.memoryBusWidth / 8.0;
    double theory_bw = mem_clock_ghz * bus_width_bytes * 2.0;
#else
    // CUDA 12+ 版本：memoryClockRate 和 clockRate 字段已移除
    // 使用 memoryBusWidth 和估算值，或提示用户使用 NVML API
    double bus_width_bytes = prop.memoryBusWidth / 8.0;
    // 对于现代 GPU，通常内存时钟在 1-2 GHz 范围，这里使用保守估算
    // 实际应用中应使用 NVML API 获取准确值
    double mem_clock_ghz = 1.0; // 保守估算，实际值需要通过 NVML 获取
    double theory_bw = mem_clock_ghz * bus_width_bytes * 2.0;
#endif

    // 2. 理论算力 (TFLOPS) - FP32
    // SM Clock (KHz) * SMs * Cores/SM * 2 (FMA) / 1e9
    // 注意: Cores/SM 很难通过 API 直接获取，这里针对 Ampere/Hopper 假设为 128 (FP32+INT32 slots)
    // 严谨计算通常需要查白皮书。这里做估算。
#if defined(CUDA_VERSION) && CUDA_VERSION < 12000
    // CUDA 11.x 及更早版本
    double sm_clock_ghz = prop.clockRate / 1e6;
#else
    // CUDA 12+ 版本：clockRate 字段已移除
    // 使用保守估算，实际应用中应使用 NVML API 获取准确值
    double sm_clock_ghz = 1.5; // 保守估算，实际值需要通过 NVML 获取
#endif
    int cores_per_sm = 128; // Standard for Ampere/Hopper/Ada FP32
    if (prop.major == 7) cores_per_sm = 64; // Volta/Turing

    double theory_flops = sm_clock_ghz * prop.multiProcessorCount * cores_per_sm * 2.0 / 1000.0;

#if defined(CUDA_VERSION) && CUDA_VERSION < 12000
    printf("  > Memory Clock      : %.2f GHz\n", mem_clock_ghz);
    printf("  > Memory Bus Width  : %d-bit\n", prop.memoryBusWidth);
    printf("  > Peak Bandwidth    : %.2f GB/s (Theoretical)\n", theory_bw);
    printf("  > SM Clock          : %.2f GHz (Boost clock may vary)\n", sm_clock_ghz);
    printf("  > Peak FP32 Compute : %.2f TFLOPS (Estimated)\n", theory_flops);
#else
    printf("  > Memory Clock      : N/A (removed in CUDA 12+, using estimate: %.2f GHz)\n", mem_clock_ghz);
    printf("  > Memory Bus Width  : %d-bit\n", prop.memoryBusWidth);
    printf("  > Peak Bandwidth    : %.2f GB/s (Estimated, use NVML for accurate value)\n", theory_bw);
    printf("  > SM Clock          : N/A (removed in CUDA 12+, using estimate: %.2f GHz)\n", sm_clock_ghz);
    printf("  > Peak FP32 Compute : %.2f TFLOPS (Estimated)\n", theory_flops);
    printf("  > Note              : For accurate clock rates, use NVML API\n");
#endif
    printf("----------------------------------------------------------------\n\n");
}

// =========================================================
// Kernel 1: Bandwidth Probe (Memory Bound)
// =========================================================
// 使用 float4 确保生成 128-bit LDG/STG 指令，最大化总线利用率
__global__ void bandwidth_kernel(const float4* __restrict__ input,
                                 float4* __restrict__ output,
                                 size_t n_vectors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop
    for (size_t i = idx; i < n_vectors; i += stride) {
        // 单纯的 Copy，没有任何计算 -> AI (Arithmetic Intensity) = 0
        output[i] = input[i];
    }
}

// =========================================================
// Kernel 2: Compute Probe (Compute Bound)
// =========================================================
// 核心逻辑：读取一次，然后在寄存器里疯狂计算，最后写回一次。
// 极高的 AI 值，确保瓶颈完全在 ALU。
__global__ void compute_floats_kernel(float* data, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到寄存器
    float reg_val = data[idx];

    // 定义多个独立的寄存器变量以增加 ILP (Instruction Level Parallelism)
    // 防止依赖链导致的流水线气泡
    float r0 = reg_val;
    float r1 = reg_val * 0.5f;
    float r2 = reg_val * 0.2f;
    float r3 = reg_val * 0.8f;

    // FMA 常数
    float alpha = 1.00001f;
    float beta  = 0.00001f;

    // 密集计算循环
    // #pragma unroll 告诉编译器展开循环，减少分支指令占比
    #pragma unroll
    for (int i = 0; i < iters; ++i) {
        // FMA: r = r * alpha + beta
        // 4 条独立的指令流，填满流水线
        r0 = __fmaf_rn(r0, alpha, beta);
        r1 = __fmaf_rn(r1, alpha, beta);
        r2 = __fmaf_rn(r2, alpha, beta);
        r3 = __fmaf_rn(r3, alpha, beta);
    }

    // 写回结果 (防止编译器把计算全优化掉)
    data[idx] = r0 + r1 + r2 + r3;
}

int main() {
    int device_id = 0;
    cudaSetDevice(device_id);
    print_theoretical_peaks(device_id);

    // 数据规模：64MB (足够大以避开 L2 Cache，直接测 HBM)
    // 太小的数据会全部驻留在 L2，测出来的是 L2 带宽而不是 HBM 带宽
    const size_t N = 16 * 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    float *d_data_in, *d_data_out;
    CUDA_CHECK(cudaMalloc(&d_data_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_out, bytes));

    // 初始化数据
    // ... (省略初始化赋值，因为我们只关心性能)

    // 创建 Events 用于高精度计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Benchmark 1: Bandwidth ---
    printf("[Micro-Bench 1] Measuring HBM Bandwidth...\n");
    // 使用 float4，元素数量除以 4
    int block_size = 256;
    int grid_size = (N / 4 + block_size - 1) / block_size;

    // Warmup
    bandwidth_kernel<<<grid_size, block_size>>>(
        (float4*)d_data_in, (float4*)d_data_out, N/4);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        bandwidth_kernel<<<grid_size, block_size>>>(
            (float4*)d_data_in, (float4*)d_data_out, N/4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_ms = ms / 10.0;

    // Bandwidth = (Read + Write) / Time
    double measured_bw = (bytes * 2.0) / (avg_ms / 1000.0) / 1e9;
    printf("  > Achieved Bandwidth: %.2f GB/s\n", measured_bw);


    // --- Benchmark 2: Compute ---
    printf("\n[Micro-Bench 2] Measuring FP32 Compute Peak...\n");
    // 计算密集型 Kernel
    // iters 足够大，使得 Loading 的开销可以忽略不计
    // AI = (iters * 4 ops * 2 FMA) / (4 bytes read + 4 bytes write)
    int iters = 1000;

    // 调整 Grid 大小以填满所有 SM
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int sm_count = prop.multiProcessorCount;
    // 假设每个 SM 跑 4 个 Block 比较合适
    grid_size = sm_count * 4;

    // Warmup
    compute_floats_kernel<<<grid_size, block_size>>>(d_data_in, iters);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        compute_floats_kernel<<<grid_size, block_size>>>(d_data_in, iters);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    avg_ms = ms / 10.0;

    // FLOPs Calculation:
    // Total Threads * Iters * 4 (Independent Ops) * 2 (FMA: Mul+Add)
    double total_flops = (double)grid_size * block_size * iters * 4 * 2;
    double measured_tflops = total_flops / (avg_ms / 1000.0) / 1e12;

    printf("  > Achieved Compute  : %.2f TFLOPS\n", measured_tflops);

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_data_in));
    CUDA_CHECK(cudaFree(d_data_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}