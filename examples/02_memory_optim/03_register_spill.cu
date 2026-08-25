/**
 * [Module B] B-03. 寄存器：Spilling / Occupancy / launch_bounds
 *
 * 同一压力核三档：
 *   baseline       : REGS=32，低寄存器压力对照
 *   highreg        : REGS=256，推 spill / 掉 occupancy
 *   launch_bounds  : REGS=256 + __launch_bounds__(256, 2)
 *   modes          : 三档一次跑齐 + 相对 baseline 加速比 CSV
 *
 * 主证据：CUDA event warmup + 多次 run → median
 * 运行时旁证：cudaFuncGetAttributes (numRegs / localSizeBytes)
 *              cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * 刻意不做：L2 lock / TMA / cp.async（→ B-04 / B-08 / B-07）
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

enum class Mode { Baseline = 0, Highreg = 1, LaunchBounds = 2, Modes = 3 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Baseline: return "baseline";
    case Mode::Highreg: return "highreg";
    case Mode::LaunchBounds: return "launch_bounds";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "baseline") == 0) return Mode::Baseline;
  if (std::strcmp(s, "highreg") == 0) return Mode::Highreg;
  if (std::strcmp(s, "launch_bounds") == 0) return Mode::LaunchBounds;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected baseline|highreg|launch_bounds|modes)\n",
               s);
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

template <int REGS_PER_THREAD>
__device__ __forceinline__ float reg_pressure_body(const float* __restrict__ in,
                                                   int tid, int iters) {
  volatile float r[REGS_PER_THREAD];
  const float seed = in[tid];
#pragma unroll
  for (int i = 0; i < REGS_PER_THREAD; ++i) {
    r[i] = seed + float(i) * 0.001f;
  }

  float acc = 0.0f;
  for (int t = 0; t < iters; ++t) {
    const int j = (t * 13 + (tid & 31)) % REGS_PER_THREAD;
    const float x = r[j];
    acc = fmaf(x, acc, 1.0f);
    r[j] = acc * 0.0001f + x;
  }
  return acc;
}

template <int REGS_PER_THREAD>
__global__ void kernel_pressure(const float* __restrict__ in, float* __restrict__ out,
                                int n, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = reg_pressure_body<REGS_PER_THREAD>(in, tid, iters);
}

template <int REGS_PER_THREAD, int MAX_THREADS, int MIN_BLOCKS>
__global__ __launch_bounds__(MAX_THREADS, MIN_BLOCKS) void kernel_pressure_lb(
    const float* __restrict__ in, float* __restrict__ out, int n, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = reg_pressure_body<REGS_PER_THREAD>(in, tid, iters);
}

struct KernelMeta {
  int num_regs = 0;
  size_t local_bytes = 0;
  int occ_blocks = 0;
};

template <typename Kernel>
static KernelMeta query_meta(Kernel k, int block) {
  KernelMeta m;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, k));
  m.num_regs = attr.numRegs;
  m.local_bytes = attr.localSizeBytes;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&m.occ_blocks, k, block, 0));
  return m;
}

static KernelMeta meta_of(Mode mode, int block) {
  switch (mode) {
    case Mode::Baseline:
      return query_meta(kernel_pressure<32>, block);
    case Mode::Highreg:
      return query_meta(kernel_pressure<256>, block);
    case Mode::LaunchBounds:
      return query_meta(kernel_pressure_lb<256, 256, 2>, block);
    default:
      return {};
  }
}

struct BenchConfig {
  Mode mode = Mode::Modes;
  int n = 1 << 20;
  int iters = 256;
  int block = 256;
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
    } else if (std::strcmp(argv[i], "--n") == 0) {
      c.n = std::atoi(need("--n"));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      c.iters = std::atoi(need("--iters"));
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
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
          "Usage: %s --mode <baseline|highreg|launch_bounds|modes> [options]\n"
          "  --n <n>        elements (default 1048576)\n"
          "  --iters <n>    inner loop (default 256)\n"
          "  --block <n>    threads/block (default 256)\n"
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
  if (c.n <= 0 || c.iters <= 0 || c.block <= 0 || c.runs <= 0 || c.warmup < 0) {
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

static float run_mode(const BenchConfig& c, Mode mode, const float* d_in, float* d_out,
                      std::vector<float>* samples) {
  const int grid = (c.n + c.block - 1) / c.block;
  switch (mode) {
    case Mode::Baseline:
      return time_launch_ms(
          [&]() { kernel_pressure<32><<<grid, c.block>>>(d_in, d_out, c.n, c.iters); },
          c.warmup, c.runs, samples);
    case Mode::Highreg:
      return time_launch_ms(
          [&]() { kernel_pressure<256><<<grid, c.block>>>(d_in, d_out, c.n, c.iters); },
          c.warmup, c.runs, samples);
    case Mode::LaunchBounds:
      return time_launch_ms(
          [&]() {
            kernel_pressure_lb<256, 256, 2><<<grid, c.block>>>(d_in, d_out, c.n, c.iters);
          },
          c.warmup, c.runs, samples);
    default:
      std::fprintf(stderr, "run_mode: unexpected mode\n");
      std::exit(EXIT_FAILURE);
  }
}

static void print_row(const BenchConfig& c, Mode mode, float med,
                      const std::vector<float>& samples, float baseline_ms,
                      const KernelMeta& meta) {
  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean = mean_of(samples);
  const float speedup =
      (med > 0.f && baseline_ms > 0.f) ? (baseline_ms / med) : 0.f;
  if (c.csv_only) {
    std::printf("%s,%.6f,%.4f,%d,%zu,%d\n", mode_name(mode), med, speedup, meta.num_regs,
                meta.local_bytes, meta.occ_blocks);
    return;
  }
  std::printf(
      "mode=%-13s  first=%.4f  median=%.4f  p95=%.4f  mean=%.4f ms  "
      "vs_base=%.3fx  regs=%d  localB=%zu  occ_blk/SM=%d\n",
      mode_name(mode), first, med, p95, mean, speedup, meta.num_regs, meta.local_bytes,
      meta.occ_blocks);
}

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  if (!cfg.csv_only) {
    std::printf("GPU: %s | sm_%d%d\n", prop.name, prop.major, prop.minor);
    std::printf("RF: regsPerSM=%d  maxThreadsPerSM=%d  regsPerBlock=%d\n",
                prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor,
                prop.regsPerBlock);
    std::printf("n=%d block=%d iters=%d | runs=%d warmup=%d\n", cfg.n, cfg.block,
                cfg.iters, cfg.runs, cfg.warmup);
    std::printf("speedup = baseline_median / mode_median  (register pressure)\n");
    std::printf("Tip: rebuild with -Xptxas=-v to compare compile-time spill loads/stores.\n");
  }

  std::vector<float> h_in(size_t(cfg.n), 1.0f);
  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size_t(cfg.n) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, size_t(cfg.n) * sizeof(float)));

  auto run_one = [&](Mode m, float baseline_ms) {
    std::vector<float> samples;
    const KernelMeta meta = meta_of(m, cfg.block);
    const float med = run_mode(cfg, m, d_in, d_out, &samples);
    print_row(cfg, m, med, samples, (m == Mode::Baseline) ? med : baseline_ms, meta);
    return med;
  };

  if (cfg.mode == Mode::Modes) {
    std::vector<float> s0, s1, s2;
    const KernelMeta m0 = meta_of(Mode::Baseline, cfg.block);
    const KernelMeta m1 = meta_of(Mode::Highreg, cfg.block);
    const KernelMeta m2 = meta_of(Mode::LaunchBounds, cfg.block);
    const float t0 = run_mode(cfg, Mode::Baseline, d_in, d_out, &s0);
    const float t1 = run_mode(cfg, Mode::Highreg, d_in, d_out, &s1);
    const float t2 = run_mode(cfg, Mode::LaunchBounds, d_in, d_out, &s2);
    if (!cfg.csv_only) {
      std::printf("\n=== modes (speedup = baseline_median / mode_median) ===\n");
    }
    print_row(cfg, Mode::Baseline, t0, s0, t0, m0);
    print_row(cfg, Mode::Highreg, t1, s1, t0, m1);
    print_row(cfg, Mode::LaunchBounds, t2, s2, t0, m2);
    std::printf("\nmode,median_ms,speedup_vs_baseline,num_regs,local_bytes,occ_blocks\n");
    std::printf("baseline,%.6f,%.4f,%d,%zu,%d\n", t0, 1.0f, m0.num_regs, m0.local_bytes,
                m0.occ_blocks);
    std::printf("highreg,%.6f,%.4f,%d,%zu,%d\n", t1, (t1 > 0.f) ? (t0 / t1) : 0.f,
                m1.num_regs, m1.local_bytes, m1.occ_blocks);
    std::printf("launch_bounds,%.6f,%.4f,%d,%zu,%d\n", t2, (t2 > 0.f) ? (t0 / t2) : 0.f,
                m2.num_regs, m2.local_bytes, m2.occ_blocks);
  } else if (cfg.mode == Mode::Baseline) {
    run_one(Mode::Baseline, 0.f);
  } else {
    std::vector<float> s_ref;
    const float baseline_ms = run_mode(cfg, Mode::Baseline, d_in, d_out, &s_ref);
    if (!cfg.csv_only) {
      std::printf("(baseline reference median=%.4f ms)\n", baseline_ms);
    }
    run_one(cfg.mode, baseline_ms);
  }

  float probe = 0.f;
  CUDA_CHECK(cudaMemcpy(&probe, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  if (!cfg.csv_only) std::printf("probe_out0=%.3f\n", probe);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
