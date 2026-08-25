/**
 * [Module B] B-01. Global Memory：合并访问、对齐与 float4
 *
 * 模式：
 *   misaligned : 连续 copy，但读侧基址 offset=1 float（破坏对齐）
 *   aligned    : 对齐连续 float copy（对照基线）
 *   float4     : 显式 float4 向量化 copy（同 useful payload）
 *   ldg_nt     : float4 + __ldcs（streaming hint；可选对照）
 *   modes      : 一次跑齐三档（+可选 ldg_nt）并打印相对 aligned 加速比 CSV
 *
 * 主证据：CUDA event warmup + 多次 run → median 时延 → 有效带宽（R+W）
 * 口径：绝对 GB/s 可能含 L2 → 主看相对 aligned 的加速比 / 形状
 * 硬件：不限 sm_90+（合并/对齐/向量化是全架构问题）
 * 刻意不做：async copy / L2 persistence / TMA（→ B-07 / B-04 / B-08）
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

enum class Mode {
  Misaligned = 0,
  Aligned = 1,
  Float4 = 2,
  LdgNt = 3,
  Modes = 4,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Misaligned: return "misaligned";
    case Mode::Aligned: return "aligned";
    case Mode::Float4: return "float4";
    case Mode::LdgNt: return "ldg_nt";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "misaligned") == 0) return Mode::Misaligned;
  if (std::strcmp(s, "aligned") == 0) return Mode::Aligned;
  if (std::strcmp(s, "float4") == 0) return Mode::Float4;
  if (std::strcmp(s, "ldg_nt") == 0) return Mode::LdgNt;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected misaligned|aligned|float4|ldg_nt|modes)\n",
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

// ---------------------------------------------------------------------------
// Kernels：GMEM→GMEM copy；写回可见存储防 DCE；尾块 clamp
// ---------------------------------------------------------------------------

// 对齐连续 float copy。每线程处理 grid-stride 个元素。
__global__ void kernel_aligned(const float* __restrict__ in, float* __restrict__ out,
                               int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = in[i] + 1.0f;  // +1 防纯 load DCE；计入 useful R+W
  }
}

// 错位：读侧从 in+offset 开始，写侧仍连续。offset 默认 1 → 破坏 16B/128B 向量对齐。
__global__ void kernel_misaligned(const float* __restrict__ in, float* __restrict__ out,
                                  int n, int offset) {
  const int stride = blockDim.x * gridDim.x;
  // 可写元素数：out[0..n-1]，但读 in[offset..offset+n-1] 需要缓冲多 offset 个 float
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = in[i + offset] + 1.0f;
  }
}

__global__ void kernel_float4(const float4* __restrict__ in, float4* __restrict__ out,
                              int n_vec) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    float4 v = in[i];
    v.x += 1.0f;
    v.y += 1.0f;
    v.z += 1.0f;
    v.w += 1.0f;
    out[i] = v;
  }
}

// streaming / evict-first hint（__ldcs → ld.cs）；不是合并替代品
__global__ void kernel_ldg_nt(const float4* __restrict__ in, float4* __restrict__ out,
                              int n_vec) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    float4 v = __ldcs(&in[i]);
    v.x += 1.0f;
    v.y += 1.0f;
    v.z += 1.0f;
    v.w += 1.0f;
    out[i] = v;
  }
}

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <misaligned|aligned|float4|ldg_nt|modes> [options]\n"
      "  --n <elems>          floats to copy (default 1<<24; must be %%4==0 for float4)\n"
      "  --offset <k>         misaligned read offset in floats (default 1)\n"
      "  --block <threads>    block size (default 256)\n"
      "  --runs <n>           timed runs (default 7)\n"
      "  --warmup <n>         warmup runs (default 2)\n"
      "  --device <id>        GPU id (default 0)\n"
      "  --with-ldg-nt        include ldg_nt row in --mode modes (optional)\n"
      "  --csv-only           only print CSV line(s)\n",
      prog);
}

struct BenchConfig {
  Mode mode = Mode::Modes;
  int n = 1 << 24;  // 64 MiB floats → 128 MiB R+W useful
  int offset = 1;
  int block = 256;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
  bool with_ldg_nt = false;
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
    } else if (std::strcmp(argv[i], "--offset") == 0) {
      c.offset = std::atoi(need("--offset"));
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
    } else if (std::strcmp(argv[i], "--runs") == 0) {
      c.runs = std::atoi(need("--runs"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      c.warmup = std::atoi(need("--warmup"));
    } else if (std::strcmp(argv[i], "--device") == 0) {
      c.device = std::atoi(need("--device"));
    } else if (std::strcmp(argv[i], "--with-ldg-nt") == 0) {
      c.with_ldg_nt = true;
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
  if (c.n <= 0 || c.block <= 0 || c.runs <= 0 || c.warmup < 0 || c.offset < 0) {
    std::fprintf(stderr, "Invalid numeric args\n");
    std::exit(EXIT_FAILURE);
  }
  if ((c.mode == Mode::Float4 || c.mode == Mode::LdgNt || c.mode == Mode::Modes) &&
      (c.n % 4) != 0) {
    std::fprintf(stderr, "--n must be divisible by 4 for float4/ldg_nt/modes\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.mode == Mode::Misaligned && c.offset <= 0) {
    std::fprintf(stderr, "--offset must be > 0 for misaligned\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

static double useful_bytes(int n) {
  // R+W；kernel 对每个元素读一次写一次（+1 写回）
  return double(n) * sizeof(float) * 2.0;
}

static double gbps_from_ms(double bytes, float med_ms) {
  if (med_ms <= 0.f) return 0.0;
  return (bytes / (1024.0 * 1024.0 * 1024.0)) / (double(med_ms) / 1000.0);
}

template <typename Launch>
static float time_launch_ms(Launch&& launch, int warmup, int runs,
                            std::vector<float>* samples) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
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
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples->push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return median_of(*samples);
}

struct DeviceBuf {
  float* d_in = nullptr;
  float* d_out = nullptr;
  int n = 0;
  int offset_pad = 0;  // extra floats at end of d_in for misaligned
};

static void alloc_bufs(DeviceBuf* b, int n, int offset) {
  b->n = n;
  b->offset_pad = std::max(0, offset);
  const size_t in_bytes = size_t(n + b->offset_pad) * sizeof(float);
  const size_t out_bytes = size_t(n) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&b->d_in, in_bytes));
  CUDA_CHECK(cudaMalloc(&b->d_out, out_bytes));
  CUDA_CHECK(cudaMemset(b->d_in, 0, in_bytes));
  CUDA_CHECK(cudaMemset(b->d_out, 0, out_bytes));
}

static void free_bufs(DeviceBuf* b) {
  if (b->d_in) CUDA_CHECK(cudaFree(b->d_in));
  if (b->d_out) CUDA_CHECK(cudaFree(b->d_out));
  b->d_in = b->d_out = nullptr;
}

static int pick_grid(int n, int block) {
  // 适度占满 SM；grid-stride 覆盖尾块
  const int want = (n + block - 1) / block;
  return std::min(want, 65535);
}

static float run_mode(const BenchConfig& c, Mode mode, DeviceBuf& buf,
                      std::vector<float>* samples) {
  const int grid = pick_grid(c.n, c.block);
  const int n_vec = c.n / 4;
  const int grid_vec = pick_grid(n_vec, c.block);

  switch (mode) {
    case Mode::Aligned:
      return time_launch_ms(
          [&]() {
            kernel_aligned<<<grid, c.block>>>(buf.d_in, buf.d_out, c.n);
            CUDA_CHECK(cudaGetLastError());
          },
          c.warmup, c.runs, samples);
    case Mode::Misaligned:
      return time_launch_ms(
          [&]() {
            kernel_misaligned<<<grid, c.block>>>(buf.d_in, buf.d_out, c.n, c.offset);
            CUDA_CHECK(cudaGetLastError());
          },
          c.warmup, c.runs, samples);
    case Mode::Float4:
      return time_launch_ms(
          [&]() {
            kernel_float4<<<grid_vec, c.block>>>(
                reinterpret_cast<const float4*>(buf.d_in),
                reinterpret_cast<float4*>(buf.d_out), n_vec);
            CUDA_CHECK(cudaGetLastError());
          },
          c.warmup, c.runs, samples);
    case Mode::LdgNt:
      return time_launch_ms(
          [&]() {
            kernel_ldg_nt<<<grid_vec, c.block>>>(
                reinterpret_cast<const float4*>(buf.d_in),
                reinterpret_cast<float4*>(buf.d_out), n_vec);
            CUDA_CHECK(cudaGetLastError());
          },
          c.warmup, c.runs, samples);
    default:
      std::fprintf(stderr, "run_mode: unexpected mode\n");
      std::exit(EXIT_FAILURE);
  }
}

static void print_row(const BenchConfig& c, Mode mode, float med,
                      const std::vector<float>& samples, float aligned_ms) {
  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean = mean_of(samples);
  const double gbps = gbps_from_ms(useful_bytes(c.n), med);
  const float speedup = (med > 0.f && aligned_ms > 0.f) ? (aligned_ms / med) : 0.f;

  if (c.csv_only) {
    std::printf("%s,%.6f,%.4f,%.4f\n", mode_name(mode), med, gbps, speedup);
    return;
  }
  std::printf(
      "mode=%-12s  first=%.4f  median=%.4f  p95=%.4f  mean=%.4f ms  "
      "~%.2f GB/s (R+W)  vs_aligned=%.3fx\n",
      mode_name(mode), first, med, p95, mean, gbps, speedup);
}

static void run_modes(const BenchConfig& c, DeviceBuf& buf) {
  struct Row {
    Mode mode;
    float ms;
    double gbps;
    std::vector<float> samples;
  };
  std::vector<Row> rows;
  rows.reserve(4);

  auto push = [&](Mode m) {
    Row r;
    r.mode = m;
    r.ms = run_mode(c, m, buf, &r.samples);
    r.gbps = gbps_from_ms(useful_bytes(c.n), r.ms);
    rows.push_back(std::move(r));
  };

  push(Mode::Misaligned);
  push(Mode::Aligned);
  push(Mode::Float4);
  if (c.with_ldg_nt) push(Mode::LdgNt);

  float aligned_ms = 0.f;
  for (const Row& r : rows) {
    if (r.mode == Mode::Aligned) aligned_ms = r.ms;
  }

  if (!c.csv_only) {
    std::printf("\n=== modes (speedup = aligned_median / mode_median) ===\n");
  }
  for (const Row& r : rows) {
    print_row(c, r.mode, r.ms, r.samples, aligned_ms);
  }

  // Machine-friendly block → docs/results/B-01_modes.csv
  std::printf("\nmode,median_ms,gbps,speedup_vs_aligned\n");
  for (const Row& r : rows) {
    const float sp = (r.ms > 0.f && aligned_ms > 0.f) ? (aligned_ms / r.ms) : 0.f;
    std::printf("%s,%.6f,%.4f,%.4f\n", mode_name(r.mode), r.ms, r.gbps, sp);
  }
}

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

  if (!cfg.csv_only) {
    std::printf("GPU: %s | sm_%d%d\n", prop.name, prop.major, prop.minor);
    std::printf("n=%d floats (%.2f MiB) | offset=%d | block=%d | runs=%d warmup=%d\n",
                cfg.n, double(cfg.n) * sizeof(float) / (1024.0 * 1024.0), cfg.offset,
                cfg.block, cfg.runs, cfg.warmup);
    std::printf("useful GB/s = 2 * n * sizeof(float) / median_s  (R+W; may include L2)\n");
  }

  DeviceBuf buf{};
  const int need_offset =
      (cfg.mode == Mode::Misaligned || cfg.mode == Mode::Modes) ? cfg.offset : 0;
  alloc_bufs(&buf, cfg.n, need_offset);

  std::vector<float> samples;
  if (cfg.mode == Mode::Modes) {
    run_modes(cfg, buf);
  } else {
    const float med = run_mode(cfg, cfg.mode, buf, &samples);
    // single-mode: vs_aligned only meaningful if we also timed aligned; print 1.0 for aligned
    float aligned_ref = (cfg.mode == Mode::Aligned) ? med : 0.f;
    if (cfg.mode != Mode::Aligned) {
      // one extra aligned timing so single-mode debug still has a relative number
      std::vector<float> align_samples;
      aligned_ref = run_mode(cfg, Mode::Aligned, buf, &align_samples);
      if (!cfg.csv_only) {
        std::printf("(aligned reference median=%.4f ms)\n", aligned_ref);
      }
    }
    print_row(cfg, cfg.mode, med, samples, aligned_ref);
  }

  // Touch one host word so host-side DCE cannot drop the device work
  float probe = 0.f;
  CUDA_CHECK(cudaMemcpy(&probe, buf.d_out, sizeof(float), cudaMemcpyDeviceToHost));
  if (!cfg.csv_only) {
    std::printf("probe_out0=%.1f\n", probe);
  }

  free_bufs(&buf);
  return 0;
}
