/**
 * [Module C] C-07. Persistent Kernel：常驻网格、拉活与何时别再 launch
 *
 * 主命题：同一批短任务，一任务一发 <<<1,block>>> vs occupancy 常驻网格
 * + block leader atomicAdd 拉活。不重测 Graph，不用 grid.sync。
 *
 * 模式：
 *   launch      : 每个 task 一次 <<<1, block>>>（基线）
 *   persistent  : occupancy×SM 网格，leader 拉 1 task 直到做完
 *   sweep       : 扫 n_tasks（work=1）主曲线
 *   sweep_work  : 固定 n_tasks，扫 work（收益收窄）
 *   oversub     : 同一 persistent 核，网格 = occ×SM×factor
 *   modes       : 定点 + occupancy 参数 + oversub 一行
 *
 * 主证据：CUDA event median；event 与 kernel 同 stream。
 * 硬件：不限 sm_90+。
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

enum class Mode { Launch = 0, Persistent = 1, Sweep = 2, SweepWork = 3, Oversub = 4, Modes = 5 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Launch: return "launch";
    case Mode::Persistent: return "persistent";
    case Mode::Sweep: return "sweep";
    case Mode::SweepWork: return "sweep_work";
    case Mode::Oversub: return "oversub";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "launch") == 0) return Mode::Launch;
  if (std::strcmp(s, "persistent") == 0) return Mode::Persistent;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "sweep_work") == 0) return Mode::SweepWork;
  if (std::strcmp(s, "oversub") == 0) return Mode::Oversub;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected launch|persistent|sweep|sweep_work|oversub|modes)\n",
               s);
  std::exit(EXIT_FAILURE);
}

static float percentile_of(const std::vector<float>& in, float p) {
  if (in.empty()) return 0.0f;
  std::vector<float> v = in;
  p = std::max(0.0f, std::min(100.0f, p));
  std::sort(v.begin(), v.end());
  const float pos = (p / 100.0f) * float(v.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const float t = pos - float(lo);
  return v[lo] * (1.0f - t) + v[hi] * t;
}

static float median_of(const std::vector<float>& v) { return percentile_of(v, 50.0f); }

__device__ __host__ inline float task_value(int task, int work) {
  float v = 0.001f * float(task + 1);
  for (int w = 0; w < work; ++w) {
    v = v * 1.0000001f + 0.0000001f;
  }
  return v;
}

__global__ void kernel_one_task(float* __restrict__ out, int task, int work) {
  const float v = task_value(task, work);
  if (threadIdx.x == 0) out[task] = v;
}

__global__ void kernel_persistent(float* __restrict__ out, int n_tasks, int work,
                                  unsigned int* __restrict__ next) {
  __shared__ unsigned int idx;
  while (true) {
    if (threadIdx.x == 0) {
      idx = atomicAdd(next, 1u);
    }
    __syncthreads();
    if (idx >= static_cast<unsigned int>(n_tasks)) return;
    const float v = task_value(static_cast<int>(idx), work);
    if (threadIdx.x == 0) out[idx] = v;
    __syncthreads();
  }
}

struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n_tasks = 4096;
  int work = 1;
  int block = 256;
  int grid = 0;  // 0 = occupancy auto (persist / oversub only)
  int oversub_factor = 8;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <launch|persistent|sweep|sweep_work|oversub|modes> [options]\n"
      "  --n-tasks <n>         task count (default 4096)\n"
      "  --work <iters>        FMA loops per task (default 1)\n"
      "  --block <thr>         threads/block (default 256)\n"
      "  --grid <blocks>       persist/oversub grid (default 0 = occupancy x SMs)\n"
      "  --oversub-factor <n>  oversub multiplier (default 8)\n"
      "  --runs <n>            timed runs (default 7)\n"
      "  --warmup <n>          warmup runs (default 2)\n"
      "  --device <id>         GPU id (default 0)\n"
      "  --csv-only            only print CSV line(s)\n",
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
    } else if (std::strcmp(argv[i], "--n-tasks") == 0) {
      c.n_tasks = std::atoi(need("--n-tasks"));
    } else if (std::strcmp(argv[i], "--work") == 0) {
      c.work = std::atoi(need("--work"));
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
    } else if (std::strcmp(argv[i], "--grid") == 0) {
      c.grid = std::atoi(need("--grid"));
    } else if (std::strcmp(argv[i], "--oversub-factor") == 0) {
      c.oversub_factor = std::atoi(need("--oversub-factor"));
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
  if (c.n_tasks < 1 || c.work < 0 || c.oversub_factor < 1) {
    std::fprintf(stderr, "ERROR: invalid --n-tasks / --work / --oversub-factor\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.block < 32 || c.block > 1024 || (c.block % 32) != 0) {
    std::fprintf(stderr, "ERROR: --block must be in [32,1024], multiple of 32\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.runs < 1 || c.warmup < 0) {
    std::fprintf(stderr, "ERROR: invalid --runs / --warmup\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

static int occupancy_grid(int block, int sms) {
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_persistent,
                                                           block, /*dynSmem=*/0));
  if (blocks_per_sm < 1) blocks_per_sm = 1;
  return blocks_per_sm * std::max(1, sms);
}

static int persist_grid_of(const BenchConfig& cfg, int occ_grid) {
  if (cfg.grid > 0) return cfg.grid;
  return occ_grid;
}

template <typename SetupFn, typename LaunchFn>
static float time_on_stream(cudaStream_t stream, SetupFn&& setup, LaunchFn&& launch, int warmup,
                            int runs, std::vector<float>* samples) {
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; ++i) {
    setup();
    launch();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  samples->clear();
  samples->reserve(static_cast<size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    setup();
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples->push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return median_of(*samples);
}

static void launch_one_each(cudaStream_t stream, float* out, int n_tasks, int work, int block) {
  for (int t = 0; t < n_tasks; ++t) {
    kernel_one_task<<<1, block, 0, stream>>>(out, t, work);
  }
  CUDA_CHECK(cudaGetLastError());
}

static void launch_persistent(cudaStream_t stream, float* out, int n_tasks, int work,
                              unsigned int* next, int grid, int block) {
  kernel_persistent<<<grid, block, 0, stream>>>(out, n_tasks, work, next);
  CUDA_CHECK(cudaGetLastError());
}

static void verify_out(float* d_out, int n_tasks, int work, bool quiet, const char* tag) {
  std::vector<float> got(static_cast<size_t>(n_tasks));
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, size_t(n_tasks) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float tol = 1e-3f;
  for (int i = 0; i < n_tasks; ++i) {
    const float expect = task_value(i, work);
    if (std::fabs(got[static_cast<size_t>(i)] - expect) >
        tol * (std::fabs(expect) + 1.0f)) {
      std::fprintf(stderr, "ERROR: %s mismatch at %d got=%g expect=%g\n", tag, i,
                   got[static_cast<size_t>(i)], expect);
      std::exit(EXIT_FAILURE);
    }
  }
  if (!quiet) {
    std::printf("verify %s OK (n_tasks=%d work=%d)\n", tag, n_tasks, work);
  }
}

static const char* kCsvHeader =
    "tag,n_tasks,work,grid,blocks_per_sm,median_ms,p10_ms,p90_ms";

static void print_row(const char* tag, int n_tasks, int work, int grid, int blocks_per_sm,
                      float med_ms, const std::vector<float>& samples, bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", tag, n_tasks, work, grid, blocks_per_sm,
                med_ms, p10, p90);
  } else {
    std::printf("%-12s n_tasks=%6d work=%d grid=%5d b/SM=%d | median=%.4f ms "
                "(p10=%.4f p90=%.4f)\n",
                tag, n_tasks, work, grid, blocks_per_sm, med_ms, p10, p90);
  }
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

  const int occ_grid = occupancy_grid(cfg.block, prop.multiProcessorCount);
  const int blocks_per_sm = occ_grid / std::max(1, prop.multiProcessorCount);
  const int persist_grid = persist_grid_of(cfg, occ_grid);
  const int oversub_grid = persist_grid * cfg.oversub_factor;
  const bool quiet = cfg.csv_only;

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("mode=%s n_tasks=%d work=%d block=%d persist_grid=%d (occ=%d, b/SM=%d) "
                "oversub_grid=%d factor=%d runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.n_tasks, cfg.work, cfg.block, persist_grid, occ_grid,
                blocks_per_sm, oversub_grid, cfg.oversub_factor, cfg.runs, cfg.warmup);
    std::printf("launch grid is always 1 (1 task = 1 launch). "
                "Do not batch tasks into one kernel.\n");
  }

  int alloc_tasks = cfg.n_tasks;
  if (cfg.mode == Mode::Sweep) alloc_tasks = 16384;
  float* d_out = nullptr;
  unsigned int* d_next = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, size_t(alloc_tasks) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_next, sizeof(unsigned int)));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<float> samples;

  auto setup_launch = [&]() {};
  auto setup_persist = [&]() {
    CUDA_CHECK(cudaMemsetAsync(d_next, 0, sizeof(unsigned int), stream));
  };

  auto run_launch = [&](int n_tasks, int work, bool verify) -> float {
    auto launch = [&]() { launch_one_each(stream, d_out, n_tasks, work, cfg.block); };
    if (verify) {
      setup_launch();
      launch();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      verify_out(d_out, n_tasks, work, quiet, "launch");
    }
    return time_on_stream(stream, setup_launch, launch, cfg.warmup, cfg.runs, &samples);
  };

  auto run_persist = [&](int n_tasks, int work, int grid, const char* tag, bool verify) -> float {
    auto launch = [&]() {
      launch_persistent(stream, d_out, n_tasks, work, d_next, grid, cfg.block);
    };
    if (verify) {
      setup_persist();
      launch();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      verify_out(d_out, n_tasks, work, quiet, tag);
    }
    return time_on_stream(stream, setup_persist, launch, cfg.warmup, cfg.runs, &samples);
  };

  auto print_speedup = [&](const char* label, int n_tasks, int work, float ms_l, float ms_p) {
    const float sp = (ms_p > 0.f) ? (ms_l / ms_p) : 0.f;
    if (cfg.csv_only) {
      std::printf("speedup_launch_persist,%d,%d,0,0,%.6f,0,0\n", n_tasks, work, sp);
    } else {
      std::printf("  launch/persistent %s: %.3fx\n", label, sp);
    }
  };

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else std::printf("\n== sweep n_tasks (main curve: launch/persistent @ work=%d) ==\n",
                     cfg.work);
    const int tasks_list[] = {64, 256, 1024, 4096, 16384};
    for (int nt : tasks_list) {
      const float ms_l = run_launch(nt, cfg.work, /*verify=*/true);
      print_row("launch", nt, cfg.work, /*grid=*/1, /*bps=*/0, ms_l, samples, cfg.csv_only);
      const float ms_p = run_persist(nt, cfg.work, persist_grid, "persistent", true);
      print_row("persistent", nt, cfg.work, persist_grid, blocks_per_sm, ms_p, samples,
                cfg.csv_only);
      char buf[64];
      std::snprintf(buf, sizeof(buf), "@n_tasks=%d", nt);
      print_speedup(buf, nt, cfg.work, ms_l, ms_p);
    }
  } else if (cfg.mode == Mode::SweepWork) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else
      std::printf("\n== sweep_work (fixed n_tasks=%d; benefit should shrink) ==\n",
                  cfg.n_tasks);
    const int work_list[] = {0, 1, 8, 64, 512, 4096};
    bool first = true;
    for (int w : work_list) {
      const float ms_l = run_launch(cfg.n_tasks, w, first);
      first = false;
      print_row("launch", cfg.n_tasks, w, 1, 0, ms_l, samples, cfg.csv_only);
      const float ms_p = run_persist(cfg.n_tasks, w, persist_grid, "persistent", true);
      print_row("persistent", cfg.n_tasks, w, persist_grid, blocks_per_sm, ms_p, samples,
                cfg.csv_only);
      char buf[64];
      std::snprintf(buf, sizeof(buf), "@work=%d", w);
      print_speedup(buf, cfg.n_tasks, w, ms_l, ms_p);
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else
      std::printf("\n== modes (n_tasks=%d work=%d) ==\n", cfg.n_tasks, cfg.work);
    const float ms_l = run_launch(cfg.n_tasks, cfg.work, true);
    print_row("launch", cfg.n_tasks, cfg.work, 1, 0, ms_l, samples, cfg.csv_only);
    const float ms_p = run_persist(cfg.n_tasks, cfg.work, persist_grid, "persistent", true);
    print_row("persistent", cfg.n_tasks, cfg.work, persist_grid, blocks_per_sm, ms_p, samples,
              cfg.csv_only);
    const float ms_o = run_persist(cfg.n_tasks, cfg.work, oversub_grid, "oversub", true);
    print_row("oversub", cfg.n_tasks, cfg.work, oversub_grid, blocks_per_sm, ms_o, samples,
              cfg.csv_only);
    if (!cfg.csv_only) {
      std::printf("speedup launch/persistent=%.3fx\n", (ms_p > 0.f) ? (ms_l / ms_p) : 0.f);
      std::printf("oversub/persistent=%.3fx  (same A/B = tA/tB; >1 => oversub slower)\n",
                  (ms_p > 0.f) ? (ms_o / ms_p) : 0.f);
    } else {
      print_speedup("modes", cfg.n_tasks, cfg.work, ms_l, ms_p);
    }
  } else if (cfg.mode == Mode::Launch) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_launch(cfg.n_tasks, cfg.work, true);
    print_row("launch", cfg.n_tasks, cfg.work, 1, 0, ms, samples, cfg.csv_only);
  } else if (cfg.mode == Mode::Persistent) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_persist(cfg.n_tasks, cfg.work, persist_grid, "persistent", true);
    print_row("persistent", cfg.n_tasks, cfg.work, persist_grid, blocks_per_sm, ms, samples,
              cfg.csv_only);
  } else if (cfg.mode == Mode::Oversub) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms_p = run_persist(cfg.n_tasks, cfg.work, persist_grid, "persistent", true);
    print_row("persistent", cfg.n_tasks, cfg.work, persist_grid, blocks_per_sm, ms_p, samples,
              cfg.csv_only);
    const float ms_o = run_persist(cfg.n_tasks, cfg.work, oversub_grid, "oversub", true);
    print_row("oversub", cfg.n_tasks, cfg.work, oversub_grid, blocks_per_sm, ms_o, samples,
              cfg.csv_only);
    if (!cfg.csv_only) {
      std::printf("oversub/persistent=%.3fx  (>1 => oversub slower)\n",
                  (ms_p > 0.f) ? (ms_o / ms_p) : 0.f);
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_next));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
