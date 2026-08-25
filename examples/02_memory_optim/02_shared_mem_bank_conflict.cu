/**
 * [Module B] B-02. Shared Memory：Bank Conflict / Padding / XOR Swizzle
 *
 * 同一 32×32 tile 的列访问三种物理布局：
 *   naive    : tile[32][32]，tile[tid][col] → 32-way bank conflict
 *   padding  : tile[32][33]，同行跨度 33，列访问错开 bank
 *   swizzle  : tile[32][32]，读 tile[row][col ^ row]
 *   modes    : 三档一次跑齐 + 相对 naive 加速比 CSV
 *
 * 主证据：CUDA event warmup + 多次 run → median
 * 刻意不做：TMA / ldmatrix / cp.async（→ B-08 / B-07）
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",           \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

enum class Mode { Naive = 0, Padding = 1, Swizzle = 2, Modes = 3 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Naive: return "naive";
    case Mode::Padding: return "padding";
    case Mode::Swizzle: return "swizzle";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "naive") == 0) return Mode::Naive;
  if (std::strcmp(s, "padding") == 0) return Mode::Padding;
  if (std::strcmp(s, "swizzle") == 0) return Mode::Swizzle;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr, "Invalid --mode=%s (expected naive|padding|swizzle|modes)\n", s);
  std::exit(EXIT_FAILURE);
}

static float percentile_of(const std::vector<float>& in, float p) {
  if (in.empty()) return 0.0f;
  std::vector<float> v = in;
  p = std::max(0.0f, std::min(100.0f, p));
  std::sort(v.begin(), v.end());
  const float pos = (p / 100.0f) * float(v.size() - 1);
  const size_t lo = (size_t)pos;
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const float t = pos - float(lo);
  return v[lo] * (1.0f - t) + v[hi] * t;
}

static float median_of(const std::vector<float>& v) { return percentile_of(v, 50.0f); }

static float mean_of(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  return std::accumulate(v.begin(), v.end(), 0.0f) / float(v.size());
}

constexpr int kTile = 32;

__global__ void kernel_naive(float* out, int iters) {
  __shared__ float tile[kTile][kTile];
  const int tid = threadIdx.x;
  for (int i = 0; i < kTile; ++i) tile[tid][i] = float(tid + i);
  __syncthreads();

  float val = 0.0f;
  for (int k = 0; k < iters; ++k) {
    const int col = k & (kTile - 1);
    val += tile[tid][col];
  }
  out[blockIdx.x * kTile + tid] = val;
}

__global__ void kernel_padding(float* out, int iters) {
  __shared__ float tile[kTile][kTile + 1];
  const int tid = threadIdx.x;
  for (int i = 0; i < kTile; ++i) tile[tid][i] = float(tid + i);
  __syncthreads();

  float val = 0.0f;
  for (int k = 0; k < iters; ++k) {
    const int col = k & (kTile - 1);
    val += tile[tid][col];
  }
  out[blockIdx.x * kTile + tid] = val;
}

__global__ void kernel_swizzle(float* out, int iters) {
  __shared__ float tile[kTile][kTile];
  const int tid = threadIdx.x;
  for (int i = 0; i < kTile; ++i) tile[tid][i] = float(tid + i);
  __syncthreads();

  float val = 0.0f;
  for (int k = 0; k < iters; ++k) {
    const int col = k & (kTile - 1);
    const int phys = col ^ tid;
    val += tile[tid][phys];
  }
  out[blockIdx.x * kTile + tid] = val;
}

struct BenchConfig {
  Mode mode = Mode::Modes;
  int grid = 2048;
  int iters = 8192;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static BenchConfig parse_args(int argc, char** argv) {
  BenchConfig c;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--mode") == 0) {
      c.mode = parse_mode(need("--mode"));
    } else if (std::strcmp(argv[i], "--grid") == 0) {
      c.grid = std::atoi(need("--grid"));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      c.iters = std::atoi(need("--iters"));
    } else if (std::strcmp(argv[i], "--runs") == 0) {
      c.runs = std::atoi(need("--runs"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      c.warmup = std::atoi(need("--warmup"));
    } else if (std::strcmp(argv[i], "--device") == 0) {
      c.device = std::atoi(need("--device"));
    } else if (std::strcmp(argv[i], "--csv-only") == 0) {
      c.csv_only = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf(
          "Usage: %s --mode <naive|padding|swizzle|modes> [options]\n"
          "  --grid <n>     CTA count (default 2048; block=32)\n"
          "  --iters <n>    column-scan repeats (default 8192)\n"
          "  --runs <n>     timed runs (default 7)\n"
          "  --warmup <n>   warmup (default 2)\n"
          "  --device <id>\n"
          "  --csv-only\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (c.grid <= 0 || c.iters <= 0 || c.runs <= 0 || c.warmup < 0) {
    std::fprintf(stderr, "Invalid numeric args\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

template <typename Launch>
static float time_launch_ms(Launch&& launch, int warmup, int runs,
                            std::vector<float>* samples) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  samples->clear();
  samples->reserve(runs);
  for (int i = 0; i < runs; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples->push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return median_of(*samples);
}

static float run_mode(const BenchConfig& c, Mode mode, float* d_out,
                      std::vector<float>* samples) {
  const int grid = c.grid;
  const int block = kTile;
  switch (mode) {
    case Mode::Naive:
      return time_launch_ms(
          [&]() { kernel_naive<<<grid, block>>>(d_out, c.iters); }, c.warmup,
          c.runs, samples);
    case Mode::Padding:
      return time_launch_ms(
          [&]() { kernel_padding<<<grid, block>>>(d_out, c.iters); }, c.warmup,
          c.runs, samples);
    case Mode::Swizzle:
      return time_launch_ms(
          [&]() { kernel_swizzle<<<grid, block>>>(d_out, c.iters); }, c.warmup,
          c.runs, samples);
    default:
      std::fprintf(stderr, "run_mode: unexpected mode\n");
      std::exit(EXIT_FAILURE);
  }
}

static void print_row(const BenchConfig& c, Mode mode, float med,
                      const std::vector<float>& samples, float naive_ms) {
  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean = mean_of(samples);
  const float speedup = (med > 0.f && naive_ms > 0.f) ? (naive_ms / med) : 0.f;
  if (c.csv_only) {
    std::printf("%s,%.6f,%.4f\n", mode_name(mode), med, speedup);
    return;
  }
  std::printf(
      "mode=%-8s  first=%.4f  median=%.4f  p95=%.4f  mean=%.4f ms  vs_naive=%.3fx\n",
      mode_name(mode), first, med, p95, mean, speedup);
}

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  if (!cfg.csv_only) {
    std::printf("GPU: %s | sm_%d%d\n", prop.name, prop.major, prop.minor);
    std::printf("grid=%d block=%d iters=%d | runs=%d warmup=%d\n", cfg.grid, kTile,
                cfg.iters, cfg.runs, cfg.warmup);
    std::printf("speedup = naive_median / mode_median  (SMEM column scan)\n");
  }

  float* d_out = nullptr;
  const size_t out_bytes = size_t(cfg.grid) * size_t(kTile) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMemset(d_out, 0, out_bytes));

  auto run_one = [&](Mode m, float naive_ms) {
    std::vector<float> samples;
    const float med = run_mode(cfg, m, d_out, &samples);
    print_row(cfg, m, med, samples, naive_ms);
    return med;
  };

  if (cfg.mode == Mode::Modes) {
    std::vector<float> s_naive, s_pad, s_sw;
    const float naive_ms = run_mode(cfg, Mode::Naive, d_out, &s_naive);
    const float pad_ms = run_mode(cfg, Mode::Padding, d_out, &s_pad);
    const float sw_ms = run_mode(cfg, Mode::Swizzle, d_out, &s_sw);
    if (!cfg.csv_only) {
      std::printf("\n=== modes (speedup = naive_median / mode_median) ===\n");
    }
    print_row(cfg, Mode::Naive, naive_ms, s_naive, naive_ms);
    print_row(cfg, Mode::Padding, pad_ms, s_pad, naive_ms);
    print_row(cfg, Mode::Swizzle, sw_ms, s_sw, naive_ms);
    std::printf("\nmode,median_ms,speedup_vs_naive\n");
    std::printf("naive,%.6f,%.4f\n", naive_ms, 1.0f);
    std::printf("padding,%.6f,%.4f\n", pad_ms,
                (pad_ms > 0.f) ? (naive_ms / pad_ms) : 0.f);
    std::printf("swizzle,%.6f,%.4f\n", sw_ms,
                (sw_ms > 0.f) ? (naive_ms / sw_ms) : 0.f);
  } else {
    std::vector<float> s_ref;
    const float naive_ms =
        (cfg.mode == Mode::Naive) ? 0.f : run_mode(cfg, Mode::Naive, d_out, &s_ref);
    if (cfg.mode != Mode::Naive && !cfg.csv_only) {
      std::printf("(naive reference median=%.4f ms)\n", naive_ms);
    }
    const float ref = (cfg.mode == Mode::Naive) ? 1.f : naive_ms;
    if (cfg.mode == Mode::Naive) {
      std::vector<float> samples;
      const float med = run_mode(cfg, Mode::Naive, d_out, &samples);
      print_row(cfg, Mode::Naive, med, samples, med);
    } else {
      run_one(cfg.mode, ref);
    }
  }

  float probe = 0.f;
  CUDA_CHECK(cudaMemcpy(&probe, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  if (!cfg.csv_only) std::printf("probe_out0=%.1f\n", probe);
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
