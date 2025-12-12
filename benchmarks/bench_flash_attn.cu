#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


// =========================================================================
// ⬇️ 模拟的 FlashAttention Kernel (占位符)
// 在实际项目中，你应该 #include "src/kernels/cuda/nn/flash_attn_fwd.cu"
// =========================================================================
namespace aspl {
namespace kernels {

// 这是一个极其简化的 Dummy Kernel，仅用于验证 Benchmark 流程是否跑通
// 它读取 Q, K, V 并写入 O，防止编译器优化掉
__global__ void mock_flash_attn_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * num_heads * head_dim;

    if (idx < total_elements) {
        // 模拟读取
        half q = Q[idx];
        half k = K[idx];
        half v = V[idx];

        // 模拟一点点计算 (FMA)
        #if __CUDA_ARCH__ >= 530
        half res = __hadd(__hmul(q, k), v);
        #else
        half res = q; // fallback for old GPUs
        #endif

        // 模拟写入
        O[idx] = res;
    }
}

} // namespace kernels
} // namespace aspl

// =========================================================================
// ⬇️ NVBench Benchmark 定义
// =========================================================================

void benchmark_flash_attn(nvbench::state& state) {
    // 1. 获取测试参数 (通过命令行或 NVBench 宏定义)
    const auto batch_size = state.get_int64("Batch");
    const auto seq_len    = state.get_int64("SeqLen");
    const auto num_heads  = state.get_int64("Heads");
    const auto head_dim   = state.get_int64("Dim");

    // 2. 计算显存大小
    // Q, K, V, O 形状均为 [Batch, SeqLen, NumHeads, HeadDim]
    const size_t num_elements = batch_size * seq_len * num_heads * head_dim;
    const size_t bytes_per_tensor = num_elements * sizeof(half);

    // 3. 分配显存 (不计入计时)
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes_per_tensor);
    cudaMalloc(&d_K, bytes_per_tensor);
    cudaMalloc(&d_V, bytes_per_tensor);
    cudaMalloc(&d_O, bytes_per_tensor);

    // 初始化数据 (防止 NaN)
    cudaMemset(d_Q, 0, bytes_per_tensor);
    cudaMemset(d_K, 0, bytes_per_tensor);
    cudaMemset(d_V, 0, bytes_per_tensor);

    // 4. 计算 Kernel Launch 参数
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    // 5. 执行 Benchmark 循环
    // NVBench 会自动处理 Warmup (预热) 和多次迭代取平均
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        // 获取当前流
        cudaStream_t stream = launch.get_stream();

        // 启动 Kernel
        aspl::kernels::mock_flash_attn_fwd_kernel<<<blocks, threads, 0, stream>>>(
            d_Q, d_K, d_V, d_O,
            (int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim
        );
    });

    // 6. 报告性能指标 (Throughput & Bandwidth)

    // 估算 FLOPs (标准 Attention 约为 4 * B * H * N^2 * D)
    // 注意：这里是近似公式，用于生成 TFLOPS 报告
    double flop_count = 4.0 * batch_size * num_heads * seq_len * seq_len * head_dim;

    // 估算 IO (Read Q,K,V + Write O)
    // 实际 FlashAttention IO 会远小于标准 Attention，这里按标准 IO 计算 Baseline
    double mem_bytes = 4.0 * bytes_per_tensor;

    state.add_element_count(num_elements, "Elements");
    state.add_global_memory_reads(3 * bytes_per_tensor);  // Read Q, K, V
    state.add_global_memory_writes(1 * bytes_per_tensor); // Write O

    // 清理资源
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

// 注册 Benchmark 并定义参数轴
NVBENCH_BENCH(benchmark_flash_attn)
    .add_int64_axis("Batch", {1, 8, 16})      // 不同的 Batch Size
    .add_int64_axis("SeqLen", {512, 1024})    // 不同的序列长度
    .add_int64_axis("Heads", {12})            // 12 头 (如 GPT-2)
    .add_int64_axis("Dim", {64});             // Head Dimension 64