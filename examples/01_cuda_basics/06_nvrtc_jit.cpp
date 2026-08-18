/**
 * [Module A] 06. Toolchain: NVRTC JIT + Driver API load
 *
 * Runtime-specialize a SAXPY scale into the source string, compile PTX
 * for this GPU's compute_XY, load with cuModuleLoadData, launch, check 7.0.
 * Host chrono around nvrtcCompileProgram is compile wall, not kernel event (A-08).
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <nvrtc.h>
#include <sstream>
#include <string>
#include <vector>

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
            const char* msg = nullptr; \
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

const char* saxpy_kernel_source_template = R"(
extern "C" __global__
void saxpy_specialized(float* x, float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = %SCALE% * x[i] + y[i];
    }
}
)";

int main() {
    CU_CHECK(cuInit(0));
    CUdevice cuDevice;
    CU_CHECK(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    CU_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[Host] GPU: %s\n", prop.name);
    printf("[Host] Compute Capability: %d.%d  sm_%d%d\n",
           prop.major, prop.minor, prop.major, prop.minor);
    printf("[Host] Not CUDA event kernel time (see A-08).\n");

    const float runtime_scale = 5.0f;
    std::string source = saxpy_kernel_source_template;
    std::stringstream scale_ss;
    scale_ss << std::fixed << std::setprecision(1) << runtime_scale << "f";
    const std::string scale_str = scale_ss.str();
    size_t pos = 0;
    while ((pos = source.find("%SCALE%", pos)) != std::string::npos) {
        source.replace(pos, 7, scale_str);
        pos += scale_str.length();
    }
    std::cout << "[NVRTC] specialized: out[i] = " << scale_str
              << " * x[i] + y[i];\n";

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), "saxpy_specialized.cu",
                                   0, nullptr, nullptr));

    std::stringstream arch_opt_ss;
    arch_opt_ss << "--gpu-architecture=compute_" << prop.major << prop.minor;
    const std::string arch_opt = arch_opt_ss.str();
    const char* opts[] = {arch_opt.c_str(), "--use_fast_math"};
    std::cout << "[NVRTC] arch: " << arch_opt << std::endl;

    const auto t0 = std::chrono::steady_clock::now();
    const nvrtcResult compile_res = nvrtcCompileProgram(prog, 2, opts);
    const auto t1 = std::chrono::steady_clock::now();
    const double compile_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t logSize = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1) {
        std::vector<char> log(logSize);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        std::cout << "[NVRTC Log]:\n" << log.data() << std::endl;
    }
    if (compile_res != NVRTC_SUCCESS) {
        exit(1);
    }

    size_t ptxSize = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    printf("[NVRTC] compile host-ms: %.2f  (nvrtcCompileProgram wall, not kernel event)\n",
           compile_ms);
    printf("[NVRTC] PTX bytes: %zu\n", ptxSize);

    CUmodule module;
    CUfunction kernel;
    CU_CHECK(cuModuleLoadData(&module, ptx.data()));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "saxpy_specialized"));

    int n = 1024;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 2.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    void* args[] = {&d_x, &d_y, &d_out, &n};
    CU_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1024, 1, 1, 0, nullptr, args, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_out[i] - 7.0f) > 1e-5f) {
            correct = false;
            printf("[Host] mismatch at %d: %f != 7.0\n", i, h_out[i]);
            break;
        }
    }
    printf("[Host] verify: %s  expected 5.0*1.0+2.0=7.0\n",
           correct ? "PASS" : "FAIL");

    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    CU_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
    CU_CHECK(cuCtxDestroy(cuContext));
    return correct ? 0 : 1;
}
