/**
 * [Module B] B-05. Unified Memory：Page Fault / Prefetch / Advise
 *
 * 目标：
 * - 对比 UM 三种策略在同一 kernel 下的时间差异：
 *   1) fault-only   : 只用 cudaMallocManaged（按需迁移）
 *   2) prefetch     : kernel 前执行 cudaMemPrefetchAsync
 *   3) advise       : 在 prefetch 基础上增加 cudaMemAdvise 提示
 *
 * 输出：
 * - first / median / p95 / mean（毫秒）
 * - 便于直接贴入文章或脚本聚合的 CSV 行
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",            \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__);  \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class UmMode {
  FaultOnly = 0,
  Prefetch = 1,
  Advise = 2,
};

static const char* mode_name(UmMode m) {
  switch (m) {
    case UmMode::FaultOnly: return "fault";
    case UmMode::Prefetch:  return "prefetch";
    case UmMode::Advise:    return "advise";
    default:                return "unknown";
  }
}

static UmMode parse_mode(const char* s) {
  if (std::strcmp(s, "fault") == 0) return UmMode::FaultOnly;
  if (std::strcmp(s, "prefetch") == 0) return UmMode::Prefetch;
  if (std::strcmp(s, "advise") == 0) return UmMode::Advise;
  std::fprintf(stderr, "Invalid --mode=%s (expected fault|prefetch|advise)\n", s);
  std::exit(EXIT_FAILURE);
}

static float mean_of(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  return std::accumulate(v.begin(), v.end(), 0.0f) / float(v.size());
}

static float percentile_of(std::vector<float> v, float p) {
  if (v.empty()) return 0.0f;
  p = std::max(0.0f, std::min(100.0f, p));
  std::sort(v.begin(), v.end());
  const float pos = (p / 100.0f) * float(v.size() - 1);
  const size_t lo = (size_t)pos;
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const float t = pos - float(lo);
  return v[lo] * (1.0f - t) + v[hi] * t;
}

static float median_of(const std::vector<float>& v) {
  return percentile_of(v, 50.0f);
}

static void print_usage(const char* prog) {
  std::printf(
      "Usage:\n"
      "  %s [--n N] [--iters I] [--mode fault|prefetch|advise] [--runs R]\n"
      "     [--warmup W] [--device D] [--csv-only]\n"
      "\n"
      "Args:\n"
      "  --n         Number of float elements (default: 16777216, ~64MB)\n"
      "  --iters     Loop count inside kernel (default: 32)\n"
      "  --mode      Unified Memory strategy (default: fault)\n"
      "  --runs      Number of measured runs (default: 5)\n"
      "  --warmup    Warmup runs not included in stats (default: 1)\n"
      "  --device    CUDA device id (default: 0)\n"
      "  --csv-only  Print only CSV output line\n",
      prog);
}

__global__ void um_touch_kernel(float* __restrict__ data, int n, int iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float acc = 0.0f;
  for (int r = 0; r < iters; ++r) {
    for (int i = tid; i < n; i += stride) {
      float x = data[i];
      acc = fmaf(x, 1.000001f, acc);
    }
  }
  if (tid < n) data[tid] = acc;
}

template <class LaunchFn>
static float time_once_ms(LaunchFn launch, cudaStream_t stream) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s, stream));
  launch(stream);
  CUDA_CHECK(cudaEventRecord(e, stream));
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

int main(int argc, char** argv) {
  int n = 1 << 24;   // 16,777,216 floats -> 64MB
  int iters = 32;
  UmMode mode = UmMode::FaultOnly;
  int runs = 5;
  int warmup = 1;
  int device = 0;
  bool csv_only = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      mode = parse_mode(argv[++i]);
    } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
      runs = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      device = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--csv-only") == 0) {
      csv_only = true;
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  n = std::max(1, n);
  iters = std::max(1, iters);
  runs = std::max(1, runs);
  warmup = std::max(0, warmup);

  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const size_t bytes = (size_t)n * sizeof(float);
  float* data = nullptr;
  CUDA_CHECK(cudaMallocManaged(&data, bytes));

  // 初始化在 CPU 上进行，确保首轮 GPU 访问可观察 UM 行为差异。
  for (int i = 0; i < n; ++i) data[i] = float((i * 131) & 1023) * 0.001f;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const dim3 block(256);
  const dim3 grid(std::min(65535, (n + (int)block.x - 1) / (int)block.x));

  auto apply_mode = [&](UmMode m) {
    if (m == UmMode::FaultOnly) return;
    if (m == UmMode::Advise) {
      CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, device));
      CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, device));
      CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, device));
    }
    CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, device, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  auto launch = [&](cudaStream_t s) {
    um_touch_kernel<<<grid, block, 0, s>>>(data, n, iters);
    CUDA_CHECK(cudaGetLastError());
  };

  for (int i = 0; i < warmup; ++i) {
    apply_mode(mode);
    (void)time_once_ms([&](cudaStream_t s) { launch(s); }, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::vector<float> times;
  times.reserve((size_t)runs);
  for (int i = 0; i < runs; ++i) {
    apply_mode(mode);
    float ms = time_once_ms([&](cudaStream_t s) { launch(s); }, stream);
    times.push_back(ms);
  }

  const float first = times.front();
  const float median = median_of(times);
  const float p95 = percentile_of(times, 95.0f);
  const float mean = mean_of(times);

  if (!csv_only) {
    std::printf("=== [Module B] B-05 Unified Memory Page Fault/Pefetch/Advise ===\n");
    std::printf("GPU: %s (device=%d)\n", prop.name, device);
    std::printf("n=%d (%.2f MB), iters=%d, mode=%s, warmup=%d, runs=%d\n",
                n, bytes / (1024.0 * 1024.0), iters, mode_name(mode), warmup, runs);
    std::printf("Result: first=%.4f ms, median=%.4f ms, p95=%.4f ms, mean=%.4f ms\n",
                first, median, p95, mean);
  }

  std::printf("CSV,um_mode=%s,n=%d,iters=%d,warmup=%d,runs=%d,first_ms=%.4f,median_ms=%.4f,p95_ms=%.4f,mean_ms=%.4f\n",
              mode_name(mode), n, iters, warmup, runs, first, median, p95, mean);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(data));
  return 0;
}

