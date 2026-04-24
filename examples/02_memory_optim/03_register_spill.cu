/**
 * [Module B] 13. 寄存器管理：Register Spilling 与 Occupancy
 *
 * 目标：
 * - 构造“寄存器压力”并观察 spill 对性能的影响
 * - 对比不同策略：
 *   1) baseline（寄存器压力较低）
 *   2) high-reg / spill-prone（寄存器压力极高，可能触发 local spill）
 *   3) launch_bounds（提示编译器的驻留约束，可能改善/也可能诱发 spill）
 *
 * 工程用法（建议）：
 * - 编译时加：-Xptxas=-v 观察 reg / spill loads / spill stores
 * - 运行时看：同一输入规模下的 kernel time（CUDA Event）
 *
 * 注意：
 * - “是否 spill”与“spill 严重程度”取决于架构、编译器版本、优化级别、以及资源（shared/reg）配置；
 *   本示例的目的是提供一个稳定的实验框架与对比方法，而不是保证在所有机器上必然 spill。
 */
 
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// 为了让编译器更难“证明无用并优化掉”，用 volatile + 数据依赖链。
template <int REGS_PER_THREAD>
__device__ __forceinline__ float reg_pressure_body(const float* __restrict__ in,
                                                   int tid,
                                                   int iters) {
    // 关键：一个较大的“线程私有数组”会把压力推到寄存器/本地内存决策边界附近。
    // 说明：数组是否放进寄存器由编译器决定；当寄存器不够时会溢出到 local memory。
    volatile float r[REGS_PER_THREAD];

    float seed = in[tid];
    #pragma unroll
    for (int i = 0; i < REGS_PER_THREAD; ++i) {
        r[i] = seed + float(i) * 0.001f;
    }

    float acc = 0.0f;
    // 用动态索引制造更长的 live range 与更难优化的访问模式
    // （并不追求“算法意义”，只追求放大寄存器压力/溢出影响）
    for (int t = 0; t < iters; ++t) {
        int j = (t * 13 + (tid & 31)) % REGS_PER_THREAD;
        float x = r[j];
        acc = fmaf(x, acc, 1.0f);
        r[j] = acc * 0.0001f + x; // 写回，防止编译器把 r 视为只读常量
    }

    return acc;
}

template <int REGS_PER_THREAD>
__global__ void reg_pressure_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n,
                                    int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    out[tid] = reg_pressure_body<REGS_PER_THREAD>(in, tid, iters);
}

// 同一个 kernel，但用 launch_bounds 给编译器一个“驻留契约”提示（可能会改变寄存器分配/调度）
template <int REGS_PER_THREAD, int MAX_THREADS, int MIN_BLOCKS>
__global__ __launch_bounds__(MAX_THREADS, MIN_BLOCKS)
void reg_pressure_kernel_lb(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n,
                            int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = reg_pressure_body<REGS_PER_THREAD>(in, tid, iters);
}

template <class LaunchFn>
static float time_kernel(LaunchFn launch, int warmup, int iters) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (int i = 0; i < warmup; ++i) launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s, stream));
    for (int i = 0; i < iters; ++i) launch(stream);
    CUDA_CHECK(cudaEventRecord(e, stream));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return ms / float(iters);
}

int main(int argc, char** argv) {
    // 默认参数：足够大以产生稳定时间，但避免 Windows TDR（按需可调小）
    int n = 1 << 20;       // 1,048,576
    int inner_iters = 256; // 每线程循环次数（放大差异）
    int warmup = 5;
    int repeat = 20;

    if (argc >= 2) n = std::atoi(argv[1]);
    if (argc >= 3) inner_iters = std::atoi(argv[2]);

    printf("=== [Module B] 13. Register Spilling & Occupancy Trade-off ===\n");
    printf("N=%d, inner_iters=%d\n", n, inner_iters);
    printf("Tip: build with -Xptxas=-v and compare reg/spill logs across variants.\n\n");

    std::vector<float> h_in(n, 1.0f);
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // Variant A: baseline（较低寄存器压力）
    auto launch_a = [&](cudaStream_t stream) {
        reg_pressure_kernel<32><<<grid, block, 0, stream>>>(d_in, d_out, n, inner_iters);
    };

    // Variant B: high-reg（更大寄存器压力，可能触发 spill）
    auto launch_b = [&](cudaStream_t stream) {
        reg_pressure_kernel<256><<<grid, block, 0, stream>>>(d_in, d_out, n, inner_iters);
    };

    // Variant C: launch_bounds（提示：每 SM 至少驻留 2 个 block；不保证一定更快）
    auto launch_c = [&](cudaStream_t stream) {
        reg_pressure_kernel_lb<256, 256, 2><<<grid, block, 0, stream>>>(d_in, d_out, n, inner_iters);
    };

    float ms_a = time_kernel([&](cudaStream_t s) { launch_a(s); }, warmup, repeat);
    float ms_b = time_kernel([&](cudaStream_t s) { launch_b(s); }, warmup, repeat);
    float ms_c = time_kernel([&](cudaStream_t s) { launch_c(s); }, warmup, repeat);

    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[A] baseline (REGS=32)    : %.4f ms\n", ms_a);
    printf("[B] high-reg (REGS=256)   : %.4f ms\n", ms_b);
    printf("[C] launch_bounds (2 blks): %.4f ms\n", ms_c);
    printf("\nInterpretation:\n");
    printf("- If [B] is much slower and ptxas shows spill loads/stores, you hit spilling on hot path.\n");
    printf("- If [C] changes reg/spill behavior, it demonstrates launch_bounds being a 'contract' not a free lunch.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

