/**
 * [Module C] C-03. Atomics 与 Contention：争用曲线、分层 staging 与 warp 聚合
 *
 * 工作负载（对齐 NVIDIA filtering Pro Tip 简化版）：
 *   输入 int 数组，值域 [0,999]；谓词 in[i] < thresh 控制 hit_rate。
 *   命中则对「同一全局计数器」做 +1（最大同址争用形态）。
 *
 * 模式：
 *   naive    : 每命中线程直接 atomicAdd(global)
 *   smem     : 命中 → atomicAdd(SMEM) → block 一次 atomicAdd(global)
 *   agg      : coalesced_threads 聚合后每活跃组一次 atomicAdd(global)（承接 C-02）
 *   agg_smem : 聚合写 SMEM，再每 block 一次 global（可选定点）
 *   sweep    : 扫 hit_rate∈{0.05,0.125,0.25,0.5,1.0}，主曲线 agg/naive、smem/naive
 *   modes    : 定点全表（默认 hit_rate=1.0 高争用）
 *
 * 主证据：CUDA event median。硬件：不限 sm_90+；主路径 32-bit int 计数。
 *
 * 注意：同 warp 同地址相同增量的 naive atomicAdd 可能被 NVCC 自动聚合；
 *       若本机 naive≈agg，属诚实对照（见大纲 TL;DR④），勿强行制造假差距。
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",             \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__);   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class Mode {
  Naive = 0,
  Smem = 1,
  Agg = 2,
  AggSmem = 3,
  Sweep = 4,
  Modes = 5,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Naive: return "naive";
    case Mode::Smem: return "smem";
    case Mode::Agg: return "agg";
    case Mode::AggSmem: return "agg_smem";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "naive") == 0) return Mode::Naive;
  if (std::strcmp(s, "smem") == 0) return Mode::Smem;
  if (std::strcmp(s, "agg") == 0) return Mode::Agg;
  if (std::strcmp(s, "agg_smem") == 0) return Mode::AggSmem;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected "
               "naive|smem|agg|agg_smem|sweep|modes)\n",
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

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Naive: every hitting thread issues one global atomicAdd.
__global__ void kernel_count_naive(const int* __restrict__ in,
                                   unsigned long long* __restrict__ out, int n,
                                   int thresh) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    if (in[i] < thresh) {
      atomicAdd(out, 1ULL);
    }
  }
}

// Shared staging: hit → SMEM atomic; then one global atomic per block.
__global__ void kernel_count_smem(const int* __restrict__ in,
                                  unsigned long long* __restrict__ out, int n,
                                  int thresh) {
  __shared__ unsigned long long block_ctr;
  if (threadIdx.x == 0) {
    block_ctr = 0;
  }
  __syncthreads();

  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    if (in[i] < thresh) {
      atomicAdd(&block_ctr, 1ULL);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && block_ctr != 0) {
    atomicAdd(out, block_ctr);
  }
}

// Warp-aggregated via coalesced_threads (C-02 form): one atomic per active group.
__global__ void kernel_count_agg(const int* __restrict__ in,
                                 unsigned long long* __restrict__ out, int n,
                                 int thresh) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    if (in[i] < thresh) {
      cg::coalesced_group g = cg::coalesced_threads();
      if (g.thread_rank() == 0) {
        atomicAdd(out, static_cast<unsigned long long>(g.size()));
      }
    }
  }
}

// Aggregated into SMEM, then one global flush per block (optional对照).
__global__ void kernel_count_agg_smem(const int* __restrict__ in,
                                      unsigned long long* __restrict__ out, int n,
                                      int thresh) {
  __shared__ unsigned long long block_ctr;
  if (threadIdx.x == 0) {
    block_ctr = 0;
  }
  __syncthreads();

  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    if (in[i] < thresh) {
      cg::coalesced_group g = cg::coalesced_threads();
      if (g.thread_rank() == 0) {
        atomicAdd(&block_ctr, static_cast<unsigned long long>(g.size()));
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && block_ctr != 0) {
    atomicAdd(out, block_ctr);
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 1 << 24;
  int block = 256;
  int grid = 0;           // 0 = auto ~SMs*8
  float hit_rate = 1.0f;  // for fixed modes
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <naive|smem|agg|agg_smem|sweep|modes> [options]\n"
      "  --n <elems>         element count (default 1<<24)\n"
      "  --block <thr>       threads/block (default 256)\n"
      "  --grid <blocks>     grid size (default 0 = auto ~SMs*8)\n"
      "  --hit-rate <r>      hit fraction in (0,1] for fixed modes (default 1.0)\n"
      "  --runs <n>          timed runs (default 7)\n"
      "  --warmup <n>        warmup runs (default 2)\n"
      "  --device <id>       GPU id (default 0)\n"
      "  --csv-only          only print CSV line(s)\n",
      prog);
}

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
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
    } else if (std::strcmp(argv[i], "--grid") == 0) {
      c.grid = std::atoi(need("--grid"));
    } else if (std::strcmp(argv[i], "--hit-rate") == 0) {
      c.hit_rate = static_cast<float>(std::atof(need("--hit-rate")));
    } else if (std::strcmp(argv[i], "--runs") == 0) {
      c.runs = std::atoi(need("--runs"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      c.warmup = std::atoi(need("--warmup"));
    } else if (std::strcmp(argv[i], "--device") == 0) {
      c.device = std::atoi(need("--device"));
    } else if (std::strcmp(argv[i], "--csv-only") == 0) {
      c.csv_only = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (c.n <= 0) {
    std::fprintf(stderr, "ERROR: --n must be positive\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.block < 32 || c.block > 1024 || (c.block % 32) != 0) {
    std::fprintf(stderr, "ERROR: --block must be in [32,1024], multiple of 32\n");
    std::exit(EXIT_FAILURE);
  }
  if (!(c.hit_rate > 0.0f && c.hit_rate <= 1.0f)) {
    std::fprintf(stderr, "ERROR: --hit-rate must be in (0,1]\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

template <typename LaunchFn>
static float time_launch_ms(LaunchFn&& launch, int warmup, int runs,
                            std::vector<float>* samples) {
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  samples->clear();
  samples->reserve(static_cast<size_t>(runs));
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

static int auto_grid(int sm_count) { return std::max(1, sm_count * 8); }

// hit_rate → thresh on values in [0,999]; exact expect when n % 1000 == 0.
static int thresh_from_hit_rate(float hit_rate) {
  int t = static_cast<int>(std::lround(static_cast<double>(hit_rate) * 1000.0));
  if (t < 1) t = 1;
  if (t > 1000) t = 1000;
  return t;
}

static void init_host_pattern(std::vector<int>* h, int n) {
  h->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*h)[static_cast<size_t>(i)] = i % 1000;  // 0..999
  }
}

static long long host_expect_hits(int n, int thresh) {
  // Exactly: for each complete period of 1000, thresh hits; plus remainder.
  const long long full = n / 1000;
  const int rem = n % 1000;
  long long hits = full * static_cast<long long>(thresh);
  hits += std::min(rem, thresh);
  return hits;
}

static void verify_count(unsigned long long* d_out, long long expect, bool quiet,
                         const char* tag) {
  unsigned long long got = 0;
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  if (static_cast<long long>(got) != expect) {
    std::fprintf(stderr, "ERROR: %s count mismatch: got=%llu expect=%lld\n", tag,
                 (unsigned long long)got, expect);
    std::exit(EXIT_FAILURE);
  }
  if (!quiet) {
    std::printf("verify %s OK (hits=%lld)\n", tag, expect);
  }
}

static void launch_kernel(Mode mode, int grid, int block, const int* d_in,
                          unsigned long long* d_out, int n, int thresh) {
  switch (mode) {
    case Mode::Naive:
      kernel_count_naive<<<grid, block>>>(d_in, d_out, n, thresh);
      break;
    case Mode::Smem:
      kernel_count_smem<<<grid, block>>>(d_in, d_out, n, thresh);
      break;
    case Mode::Agg:
      kernel_count_agg<<<grid, block>>>(d_in, d_out, n, thresh);
      break;
    case Mode::AggSmem:
      kernel_count_agg_smem<<<grid, block>>>(d_in, d_out, n, thresh);
      break;
    default:
      std::fprintf(stderr, "ERROR: launch_kernel bad mode\n");
      std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cudaGetLastError());
}

static float run_one(Mode mode, int grid, int block, int* d_in,
                     unsigned long long* d_out, int n, int thresh, long long expect,
                     int warmup, int runs, bool verify, bool quiet, const char* tag,
                     std::vector<float>* samples) {
  if (verify) {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    launch_kernel(mode, grid, block, d_in, d_out, n, thresh);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify_count(d_out, expect, quiet, tag);
  }
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    launch_kernel(mode, grid, block, d_in, d_out, n, thresh);
  };
  return time_launch_ms(launch, warmup, runs, samples);
}

static void print_row(const char* tag, float hit_rate, int thresh, int block, int grid,
                      int n, float med_ms, const std::vector<float>& samples,
                      bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%.4f,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", tag, hit_rate, thresh, block,
                grid, n, med_ms, p10, p90);
  } else {
    std::printf("%-8s hit=%.3f thresh=%4d block=%4d grid=%4d | median=%.4f ms "
                "(p10=%.4f p90=%.4f)\n",
                tag, hit_rate, thresh, block, grid, med_ms, p10, p90);
  }
}

static const char* kCsvHeader =
    "tag,hit_rate,thresh,block,grid,n,median_ms,p10_ms,p90_ms";

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int grid = (cfg.grid > 0) ? cfg.grid : auto_grid(prop.multiProcessorCount);
  const bool quiet = cfg.csv_only;

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("mode=%s n=%d block=%d grid=%d hit_rate=%.3f runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.n, cfg.block, grid, cfg.hit_rate, cfg.runs,
                cfg.warmup);
  }

  std::vector<int> h_in;
  init_host_pattern(&h_in, cfg.n);

  int* d_in = nullptr;
  unsigned long long* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, size_t(cfg.n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size_t(cfg.n) * sizeof(int),
                        cudaMemcpyHostToDevice));

  std::vector<float> samples;

  auto run_tagged = [&](Mode m, float hit_rate, bool verify) {
    const int thresh = thresh_from_hit_rate(hit_rate);
    const long long expect = host_expect_hits(cfg.n, thresh);
    const char* tag = mode_name(m);
    const float ms =
        run_one(m, grid, cfg.block, d_in, d_out, cfg.n, thresh, expect, cfg.warmup,
                cfg.runs, verify, quiet, tag, &samples);
    print_row(tag, hit_rate, thresh, cfg.block, grid, cfg.n, ms, samples, cfg.csv_only);
    return ms;
  };

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) {
      std::printf("%s\n", kCsvHeader);
    } else {
      std::printf("\n== sweep hit_rate (main curve: agg/naive, smem/naive) ==\n");
    }
    const float kRates[] = {0.05f, 0.125f, 0.25f, 0.5f, 1.0f};
    bool first = true;
    for (float hr : kRates) {
      const float ms_naive = run_tagged(Mode::Naive, hr, first);
      first = false;
      const float ms_smem = run_tagged(Mode::Smem, hr, false);
      const float ms_agg = run_tagged(Mode::Agg, hr, false);
      if (cfg.csv_only) {
        std::printf("speedup_agg,%.4f,0,%d,%d,%d,%.6f,,\n", hr, cfg.block, grid, cfg.n,
                    (ms_agg > 0.f) ? (ms_naive / ms_agg) : 0.f);
        std::printf("speedup_smem,%.4f,0,%d,%d,%d,%.6f,,\n", hr, cfg.block, grid, cfg.n,
                    (ms_smem > 0.f) ? (ms_naive / ms_smem) : 0.f);
      } else {
        std::printf("  speedup @hit=%.3f: agg/naive=%.3fx  smem/naive=%.3fx\n", hr,
                    (ms_agg > 0.f) ? (ms_naive / ms_agg) : 0.f,
                    (ms_smem > 0.f) ? (ms_naive / ms_smem) : 0.f);
      }
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) {
      std::printf("%s\n", kCsvHeader);
    } else {
      std::printf("\n== modes (fixed-point @ hit_rate=%.3f) ==\n", cfg.hit_rate);
    }
    const float ms_naive = run_tagged(Mode::Naive, cfg.hit_rate, true);
    const float ms_smem = run_tagged(Mode::Smem, cfg.hit_rate, true);
    const float ms_agg = run_tagged(Mode::Agg, cfg.hit_rate, true);
    const float ms_as = run_tagged(Mode::AggSmem, cfg.hit_rate, true);
    if (!cfg.csv_only) {
      std::printf("speedup agg/naive=%.3fx  smem/naive=%.3fx  agg_smem/naive=%.3fx\n",
                  (ms_agg > 0.f) ? (ms_naive / ms_agg) : 0.f,
                  (ms_smem > 0.f) ? (ms_naive / ms_smem) : 0.f,
                  (ms_as > 0.f) ? (ms_naive / ms_as) : 0.f);
    }
  } else {
    if (cfg.csv_only) {
      std::printf("%s\n", kCsvHeader);
    }
    run_tagged(cfg.mode, cfg.hit_rate, true);
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
