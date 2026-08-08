/**
 * [Module C] C-05. Kernel Fusion 代价边界：少写回 vs 寄存器 / occupancy
 *
 * 工作负载：float 点式链，第 s 级 y = a_s*y + b_s，偶级再 ReLU。
 *   naive : k 个独立 kernel，中间结果写 global（双缓冲）
 *   fused : 单核，中间量留寄存器
 *   fat   : 同 fused 语义，另造大量 live 临时（写 sink 防 DCE）抬寄存器压力
 *
 * 模式：
 *   naive|fused|fat : 定点（默认 k）
 *   sweep           : 扫 k∈{2,3,4,6,8} 上 naive vs fused（主曲线；fat 不进 sweep）
 *   modes           : 定点 k 上 naive/fused/fat + occupancy
 *
 * 计时：一对 CUDA event 包住整条链（naive = 连续 launch k 核）。
 * 主证据：median。硬件：不限 sm_90+。
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",             \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__);   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class Mode { Naive = 0, Fused = 1, Fat = 2, Sweep = 3, Modes = 4 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Naive: return "naive";
    case Mode::Fused: return "fused";
    case Mode::Fat: return "fat";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "naive") == 0) return Mode::Naive;
  if (std::strcmp(s, "fused") == 0) return Mode::Fused;
  if (std::strcmp(s, "fat") == 0) return Mode::Fat;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr, "Invalid --mode=%s (expected naive|fused|fat|sweep|modes)\n", s);
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
// Device: stage op (must match host)
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ float stage_op(float v, int s) {
  const float a = 1.0f + 0.01f * float(s + 1);
  const float b = 0.001f * float(s + 1);
  v = a * v + b;
  if ((s & 1) == 0) {
    v = v > 0.0f ? v : 0.0f;
  }
  return v;
}

__global__ void kernel_stage(const float* __restrict__ in, float* __restrict__ out, int n,
                             int stage) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    out[i] = stage_op(in[i], stage);
  }
}

__global__ void kernel_fused(const float* __restrict__ in, float* __restrict__ out, int n,
                             int k) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    float v = in[i];
    for (int s = 0; s < k; ++s) {
      v = stage_op(v, s);
    }
    out[i] = v;
  }
}

// Fat: same math as fused for out[i]; many live temps accumulated into sink (防 DCE).
#ifndef C05_FAT_TEMPS
#define C05_FAT_TEMPS 48
#endif

__global__ void kernel_fat(const float* __restrict__ in, float* __restrict__ out, int n,
                           int k, float* __restrict__ sink) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    float v = in[i];
    for (int s = 0; s < k; ++s) {
      v = stage_op(v, s);
    }

    // Live register pressure probe: temps depend on i / lane / t; must contribute to sink.
    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < C05_FAT_TEMPS; ++t) {
      float tmp = v;
      tmp = tmp * (1.0f + 1.0e-4f * float(t + 1)) + float((i + t) & 1023) * 1.0e-3f;
      tmp = stage_op(tmp, t % (k > 0 ? k : 1));
      acc += tmp;
    }
    out[i] = v;
    // One atomic per thread is heavy; fold into warp then one atomic — still keeps acc live.
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if ((threadIdx.x & 31) == 0) {
      atomicAdd(sink, acc);
    }
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 1 << 24;
  int k = 4;
  int block = 256;
  int grid = 0;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <naive|fused|fat|sweep|modes> [options]\n"
      "  --n <elems>      element count (default 1<<24)\n"
      "  --k <stages>     chain length for fixed modes (default 4)\n"
      "  --block <thr>    threads/block (default 256)\n"
      "  --grid <blocks>  grid size (default 0 = auto ~SMs*8)\n"
      "  --runs <n>       timed runs (default 7)\n"
      "  --warmup <n>     warmup runs (default 2)\n"
      "  --device <id>    GPU id (default 0)\n"
      "  --csv-only       only print CSV line(s)\n",
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
    } else if (std::strcmp(argv[i], "--k") == 0) {
      c.k = std::atoi(need("--k"));
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
    } else if (std::strcmp(argv[i], "--grid") == 0) {
      c.grid = std::atoi(need("--grid"));
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
  if (c.k < 2 || c.k > 16) {
    std::fprintf(stderr, "ERROR: --k must be in [2,16]\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.block < 32 || c.block > 1024 || (c.block % 32) != 0) {
    std::fprintf(stderr, "ERROR: --block must be in [32,1024], multiple of 32\n");
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

static int query_occ(const void* kernel, int block) {
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block, 0));
  return blocks_per_sm;
}

static void host_chain(const std::vector<float>& in, std::vector<float>* out, int k) {
  out->resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    float v = in[i];
    for (int s = 0; s < k; ++s) {
      v = stage_op(v, s);
    }
    (*out)[i] = v;
  }
}

static void verify_out(const float* d_out, const std::vector<float>& expect, bool quiet,
                       const char* tag) {
  std::vector<float> got(expect.size());
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, expect.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const int ncheck = (int)std::min<size_t>(expect.size(), 4096);
  // stride sample + first/last
  auto check_at = [&](int i) {
    const float e = expect[static_cast<size_t>(i)];
    const float g = got[static_cast<size_t>(i)];
    const float tol = 1e-4f * (std::fabs(e) + 1.0f);
    if (std::fabs(g - e) > tol) {
      std::fprintf(stderr, "ERROR: %s mismatch at i=%d got=%g expect=%g\n", tag, i, g, e);
      std::exit(EXIT_FAILURE);
    }
  };
  for (int i = 0; i < ncheck; ++i) check_at(i);
  check_at((int)expect.size() - 1);
  if (!quiet) std::printf("verify %s OK (sampled)\n", tag);
}

static const char* kCsvHeader =
    "tag,k,n,block,grid,median_ms,p10_ms,p90_ms,occ_blocks_per_sm";

static void print_row(const char* tag, int k, int n, int block, int grid, float med_ms,
                      const std::vector<float>& samples, int occ, bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d\n", tag, k, n, block, grid, med_ms, p10, p90,
                occ);
  } else {
    std::printf("%-6s k=%2d n=%d block=%d grid=%d | median=%.4f ms "
                "(p10=%.4f p90=%.4f) occ_bpsm=%d\n",
                tag, k, n, block, grid, med_ms, p10, p90, occ);
  }
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int grid = (cfg.grid > 0) ? cfg.grid : auto_grid(prop.multiProcessorCount);
  const bool quiet = cfg.csv_only;

  const int occ_stage = query_occ((void*)kernel_stage, cfg.block);
  const int occ_fused = query_occ((void*)kernel_fused, cfg.block);
  const int occ_fat = query_occ((void*)kernel_fat, cfg.block);

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("mode=%s n=%d k=%d block=%d grid=%d runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.n, cfg.k, cfg.block, grid, cfg.runs, cfg.warmup);
    std::printf("occupancy blocks/SM @block=%d: stage=%d fused=%d fat=%d (FAT_TEMPS=%d)\n",
                cfg.block, occ_stage, occ_fused, occ_fat, C05_FAT_TEMPS);
  }

  std::vector<float> h_in(static_cast<size_t>(cfg.n));
  for (int i = 0; i < cfg.n; ++i) {
    h_in[static_cast<size_t>(i)] = float((i % 1000) - 500) * 0.01f;
  }

  float *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr, *d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size_t(cfg.n) * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<float> samples;
  std::vector<float> h_expect;

  auto run_naive = [&](int k, bool verify) -> float {
    auto launch = [&]() {
      float* a = d_tmp;
      float* b = d_out;
      CUDA_CHECK(cudaMemcpy(a, d_in, size_t(cfg.n) * sizeof(float), cudaMemcpyDeviceToDevice));
      for (int s = 0; s < k; ++s) {
        kernel_stage<<<grid, cfg.block>>>(a, b, cfg.n, s);
        float* t = a;
        a = b;
        b = t;
      }
      if (a != d_out) {
        CUDA_CHECK(cudaMemcpy(d_out, a, size_t(cfg.n) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
      }
      CUDA_CHECK(cudaGetLastError());
    };
    if (verify) {
      host_chain(h_in, &h_expect, k);
      launch();
      CUDA_CHECK(cudaDeviceSynchronize());
      verify_out(d_out, h_expect, quiet, "naive");
    }
    return time_launch_ms(launch, cfg.warmup, cfg.runs, &samples);
  };

  auto run_fused = [&](int k, bool verify) -> float {
    auto launch = [&]() {
      kernel_fused<<<grid, cfg.block>>>(d_in, d_out, cfg.n, k);
      CUDA_CHECK(cudaGetLastError());
    };
    if (verify) {
      host_chain(h_in, &h_expect, k);
      launch();
      CUDA_CHECK(cudaDeviceSynchronize());
      verify_out(d_out, h_expect, quiet, "fused");
    }
    return time_launch_ms(launch, cfg.warmup, cfg.runs, &samples);
  };

  auto run_fat = [&](int k, bool verify) -> float {
    auto launch = [&]() {
      CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(float)));
      kernel_fat<<<grid, cfg.block>>>(d_in, d_out, cfg.n, k, d_sink);
      CUDA_CHECK(cudaGetLastError());
    };
    if (verify) {
      host_chain(h_in, &h_expect, k);
      launch();
      CUDA_CHECK(cudaDeviceSynchronize());
      verify_out(d_out, h_expect, quiet, "fat");
      // touch sink so host keeps allocation live
      float sink_h = 0;
      CUDA_CHECK(cudaMemcpy(&sink_h, d_sink, sizeof(float), cudaMemcpyDeviceToHost));
      if (!quiet) std::printf("fat sink=%g (pressure probe only)\n", sink_h);
    }
    return time_launch_ms(launch, cfg.warmup, cfg.runs, &samples);
  };

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else std::printf("\n== sweep k (main curve: fused/naive; fat not in sweep) ==\n");
    const int ks[] = {2, 3, 4, 6, 8};
    bool first = true;
    for (int k : ks) {
      const float ms_n = run_naive(k, first);
      first = false;
      print_row("naive", k, cfg.n, cfg.block, grid, ms_n, samples, occ_stage, cfg.csv_only);
      const float ms_f = run_fused(k, false);
      print_row("fused", k, cfg.n, cfg.block, grid, ms_f, samples, occ_fused, cfg.csv_only);
      const float sp = (ms_f > 0.f) ? (ms_n / ms_f) : 0.f;
      if (cfg.csv_only) {
        std::printf("speedup_fused,%d,%d,%d,%d,%.6f,,,0\n", k, cfg.n, cfg.block, grid, sp);
      } else {
        std::printf("  fused/naive @k=%d: %.3fx\n", k, sp);
      }
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else std::printf("\n== modes (fixed k=%d; includes fat) ==\n", cfg.k);
    const float ms_n = run_naive(cfg.k, true);
    print_row("naive", cfg.k, cfg.n, cfg.block, grid, ms_n, samples, occ_stage, cfg.csv_only);
    const float ms_f = run_fused(cfg.k, true);
    print_row("fused", cfg.k, cfg.n, cfg.block, grid, ms_f, samples, occ_fused, cfg.csv_only);
    const float ms_fat = run_fat(cfg.k, true);
    print_row("fat", cfg.k, cfg.n, cfg.block, grid, ms_fat, samples, occ_fat, cfg.csv_only);
    if (!cfg.csv_only) {
      std::printf("speedup fused/naive=%.3fx  fat/fused=%.3fx  "
                  "(fat/fused>1 => pressure hurt)\n",
                  (ms_f > 0.f) ? (ms_n / ms_f) : 0.f,
                  (ms_f > 0.f) ? (ms_fat / ms_f) : 0.f);
      std::printf("occ_bpsm stage=%d fused=%d fat=%d\n", occ_stage, occ_fused, occ_fat);
    }
  } else if (cfg.mode == Mode::Naive) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_naive(cfg.k, true);
    print_row("naive", cfg.k, cfg.n, cfg.block, grid, ms, samples, occ_stage, cfg.csv_only);
  } else if (cfg.mode == Mode::Fused) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_fused(cfg.k, true);
    print_row("fused", cfg.k, cfg.n, cfg.block, grid, ms, samples, occ_fused, cfg.csv_only);
  } else if (cfg.mode == Mode::Fat) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_fat(cfg.k, true);
    print_row("fat", cfg.k, cfg.n, cfg.block, grid, ms, samples, occ_fat, cfg.csv_only);
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_sink));
  return EXIT_SUCCESS;
}
