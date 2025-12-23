/**
 * [Module A] 06. CUDA 工具链详解
 * NVRTC (Runtime Compilation) 实战：动态构建、编译与加载 Kernel
 *
 * 场景模拟：
 * 我们需要一个 SAXPY (Y = a*X + Y) 算子。
 * 传统方式：写一个通用 Kernel，'a' 作为参数传入。编译器无法做常量折叠。
 * NVRTC方式：运行时已知 'a=2.5'，我们将其硬编码进源码字符串，触发编译器极致优化。
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>

// --- 错误检查宏 (针对 NVRTC 和 Driver API) ---
#define NVRTC_CHECK(call) \
    do { \
        nvrtcResult result = call; \
        if (result != NVRTC_SUCCESS) { \
            std::cerr << "NVRTC Error: " << nvrtcGetErrorString(result) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CU_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* msg; \
            cuGetErrorName(result, &msg); \
            std::cerr << "Driver API Error: " << msg << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Runtime API Error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

// --- 1. Kernel 源码模版 ---
// 注意：这是 C++ 字符串，不是 .cu 文件
// 我们使用占位符 %SCALE% 来演示"运行时特化"
const char* saxpy_kernel_source_template = R"(
extern "C" __global__
void saxpy_specialized(float* x, float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // %SCALE% 会在运行时被替换为具体的数值 (如 2.5)
        // 编译器会直接生成 FMUL R1, R2, 2.5 (立即数乘法)，比读取 Constant Memory 更快
        out[i] = %SCALE% * x[i] + y[i];
    }
}
)";

int main() {
    std::cout << "[Host] Starting NVRTC JIT Compilation Demo..." << std::endl;

    // --- 步骤 0: 初始化 Driver API ---
    // NVRTC 生成的 PTX 必须通过 Driver API (libcuda.so) 加载
    CU_CHECK(cuInit(0));
    CUdevice cuDevice;
    CU_CHECK(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    // CUDA 13.1+ requires CUctxCreateParams parameter (can be NULL for default)
    CU_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));

    // --- 步骤 1: 准备源码 (Runtime Specialization) ---
    // 假设运行时决定 scale = 5.0f
    float runtime_scale = 5.0f;
    std::string source = saxpy_kernel_source_template;

    // 简单的字符串替换，模拟模板特化
    // 使用 stringstream 格式化浮点数，确保正确的格式（如 5.0f 而不是 5.000000）
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << runtime_scale << "f";
    std::string scale_str = ss.str();

    // 替换所有出现的 %SCALE%
    size_t pos = 0;
    while ((pos = source.find("%SCALE%", pos)) != std::string::npos) {
        source.replace(pos, 7, scale_str);
        pos += scale_str.length(); // 移动到替换后的位置
    }

    std::cout << "[NVRTC] Specialized Source Code generated:\n"
              << "   out[i] = " << scale_str << " * x[i] + y[i];" << std::endl;

    // --- 步骤 2: 创建 NVRTC 程序 ---
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog,
                                   source.c_str(),
                                   "saxpy_specialized.cu", // 虚拟文件名
                                   0, NULL, NULL));        // 不包含外部头文件

    // --- 步骤 3: 编译为 PTX ---
    // 动态获取当前 GPU 的 Compute Capability，生成匹配架构的 PTX
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int cc_major = prop.major;
    int cc_minor = prop.minor;

    std::stringstream arch_opt_ss;
    arch_opt_ss << "--gpu-architecture=compute_" << cc_major << cc_minor;
    std::string arch_opt = arch_opt_ss.str();

    const char* opts[] = {
        arch_opt.c_str(),
        "--use_fast_math"
    };

    nvrtcResult compile_res = nvrtcCompileProgram(prog, 2, opts);

    // 获取编译日志 (如果有错误或警告)
    size_t logSize;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1) {
        std::vector<char> log(logSize);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        std::cout << "[NVRTC Log]:\n" << log.data() << std::endl;
    }
    if (compile_res != NVRTC_SUCCESS) exit(1);

    // 获取 PTX 二进制
    size_t ptxSize;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    std::cout << "[NVRTC] PTX generated (" << ptxSize << " bytes)." << std::endl;

    // --- 步骤 4: 加载到 GPU (Driver API) ---
    CUmodule module;
    CUfunction kernel;
    CU_CHECK(cuModuleLoadData(&module, ptx.data()));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "saxpy_specialized"));

    // --- 步骤 5: 准备数据与执行 ---
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // 使用 Runtime API 分配内存 (Runtime 和 Driver API 可以混用)
    float *d_x, *d_y, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // 初始化数据
    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y(N, 2.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    // Launch 参数 (void* 数组的数组)
    void* args[] = { &d_x, &d_y, &d_out, (void*)&N };

    // Launch Kernel (1 Block, 1024 Threads)
    CU_CHECK(cuLaunchKernel(kernel,
                            1, 1, 1,    // Grid Dim
                            1024, 1, 1, // Block Dim
                            0, NULL,    // Shared Mem, Stream
                            args, NULL));

    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 步骤 6: 验证结果 ---
    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < N; i++) {
        // 期望结果: 5.0 * 1.0 + 2.0 = 7.0
        if (abs(h_out[i] - 7.0f) > 1e-5) {
            correct = false;
            printf("Error at %d: %f != 7.0\n", i, h_out[i]);
            break;
        }
    }

    if (correct) std::cout << "[Host] Verification PASSED! Result is 7.0" << std::endl;
    else std::cout << "[Host] Verification FAILED!" << std::endl;

    // 清理
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}