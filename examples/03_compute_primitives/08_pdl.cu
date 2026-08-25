/**
 * [Module C] C-08. PDL：同流依赖核的提前启动与何时叠不上
 *
 * 主命题：同 stream 上有数据依赖的 K1→K2，串行等退休 vs
 * Programmatic Dependent Launch 允许次核提前 boot。
 * 重叠来自 K1 trigger 之后的 tail ∥ K2 wait 之前的独立 work。
 *
 * 模式：
 *   serial      : 同流 <<<K1>>> 再 <<<K2>>>（无 PDL attribute）
 *   pdl         : K1 trigger+tail；K2 LaunchKernelEx + wait；半 occupancy
 *   sweep       : 固定 tail，扫 K2 work
 *   sweep_tail  : 固定 work，扫 K1 tail
 *   modes       : 定点 + pdl_full（满 occupancy 反例）
 *
 * 主证据：CUDA event median；event 与 kernel 同 stream。
 * 硬件：sm_90+。CC 不够清晰退出。
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

enum class Mode { Serial = 0, Pdl = 1, Sweep = 2, SweepTail = 3, Modes = 4 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Serial: return "serial";
    case Mode::Pdl: return "pdl";
    case Mode::Sweep: return "sweep";
    case Mode::SweepTail: return "sweep_tail";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "serial") == 0) return Mode::Serial;
  if (std::strcmp(s, "pdl") == 0) return Mode::Pdl;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "sweep_tail") == 0) return Mode::SweepTail;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected serial|pdl|sweep|sweep_tail|modes)\n", s);
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

__device__ __host__ inline float burn(float v, int iters) {
  for (int w = 0; w < iters; ++w) {
    v = v * 1.0000001f + 0.0000001f;
  }
  return v;
}

__device__ inline void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ inline void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
}

__global__ void kernel_primary(float* __restrict__ out, float* __restrict__ scratch, int n,
                               int tail) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = 0.001f * float(i + 1);
  }
  pdl_trigger();
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    acc = burn(acc + 1.f, tail);
  }
  if (threadIdx.x == 0) scratch[blockIdx.x] = acc;
}

__global__ void kernel_secondary(const float* __restrict__ out, float* __restrict__ sink,
                                 int n, int work) {
  const int stride = blockDim.x * gridDim.x;
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    acc = burn(acc + 1.f, work);
  }
  pdl_wait();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    sink[i] = out[i] * 2.f + acc * 0.f;
  }
}

struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 1 << 20;
  int work = 512;
  int tail = 512;
  int block = 256;
  int grid = 0;  // 0 = half occupancy
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <serial|pdl|sweep|sweep_tail|modes> [options]\n"
      "  --n <elems>     element count (default 1048576)\n"
      "  --work <iters>  K2 independent FMA after boot / before wait (default 512)\n"
      "  --tail <iters>  K1 FMA after trigger (default 512)\n"
      "  --block <thr>   threads/block (default 256)\n"
      "  --grid <blocks> override half-occupancy grid (default 0 = auto)\n"
      "  --runs <n>      timed runs (default 7)\n"
      "  --warmup <n>    warmup runs (default 2)\n"
      "  --device <id>   GPU id (default 0)\n"
      "  --csv-only      only print CSV line(s)\n",
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
    } else if (std::strcmp(argv[i], "--work") == 0) {
      c.work = std::atoi(need("--work"));
    } else if (std::strcmp(argv[i], "--tail") == 0) {
      c.tail = std::atoi(need("--tail"));
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
  if (c.n < 1 || c.work < 0 || c.tail < 0) {
    std::fprintf(stderr, "ERROR: invalid --n / --work / --tail\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.block < 32 || c.block > 1024 || (c.block % 32) != 0) {
    std::fprintf(stderr, "ERROR: --block must be in [32,1024], multiple of 32\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

static int occupancy_grid(int block, int sms) {
  int b1 = 0, b2 = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, kernel_primary, block, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b2, kernel_secondary, block, 0));
  const int bps = std::max(1, std::min(b1, b2));
  return bps * std::max(1, sms);
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

static void launch_serial(cudaStream_t stream, float* out, float* scratch, float* sink, int n,
                          int tail, int work, int grid, int block) {
  kernel_primary<<<grid, block, 0, stream>>>(out, scratch, n, tail);
  kernel_secondary<<<grid, block, 0, stream>>>(out, sink, n, work);
  CUDA_CHECK(cudaGetLastError());
}

static void launch_pdl(cudaStream_t stream, float* out, float* scratch, float* sink, int n,
                       int tail, int work, int grid, int block) {
  kernel_primary<<<grid, block, 0, stream>>>(out, scratch, n, tail);

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel_secondary, out, sink, n, work));
  CUDA_CHECK(cudaGetLastError());
}

static void verify_sink(float* d_sink, int n, bool quiet, const char* tag) {
  std::vector<float> got(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(got.data(), d_sink, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
  const float tol = 1e-3f;
  const int step = std::max(1, n / 4096);
  for (int i = 0; i < n; i += step) {
    const float expect = 0.002f * float(i + 1);
    if (std::fabs(got[static_cast<size_t>(i)] - expect) >
        tol * (std::fabs(expect) + 1.0f)) {
      std::fprintf(stderr, "ERROR: %s mismatch at %d got=%g expect=%g\n", tag, i,
                   got[static_cast<size_t>(i)], expect);
      std::exit(EXIT_FAILURE);
    }
  }
  if (!quiet) std::printf("verify %s OK (n=%d)\n", tag, n);
}

static const char* kCsvHeader = "tag,n,work,tail,grid,median_ms,p10_ms,p90_ms";

static void print_row(const char* tag, int n, int work, int tail, int grid, float med_ms,
                      const std::vector<float>& samples, bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", tag, n, work, tail, grid, med_ms, p10, p90);
  } else {
    std::printf("%-10s n=%d work=%d tail=%d grid=%d | median=%.4f ms (p10=%.4f p90=%.4f)\n",
                tag, n, work, tail, grid, med_ms, p10, p90);
  }
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int cc = prop.major * 10 + prop.minor;
  if (prop.major < 9) {
    std::fprintf(stderr,
                 "ERROR: C-08 PDL needs sm_90+ (got sm_%d%d / %s). "
                 "This is not an A-08 stream chapter.\n",
                 prop.major, prop.minor, prop.name);
    return EXIT_FAILURE;
  }

  const int occ_grid = occupancy_grid(cfg.block, prop.multiProcessorCount);
  const int blocks_per_sm = occ_grid / std::max(1, prop.multiProcessorCount);
  const int half_grid = (cfg.grid > 0) ? cfg.grid : std::max(1, occ_grid / 2);
  const int full_grid = occ_grid;
  const bool quiet = cfg.csv_only;

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d | CC=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, cc);
    std::printf("mode=%s n=%d work=%d tail=%d block=%d half_grid=%d full_grid=%d "
                "(occ=%d, b/SM=%d) runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.n, cfg.work, cfg.tail, cfg.block, half_grid, full_grid,
                occ_grid, blocks_per_sm, cfg.runs, cfg.warmup);
    std::printf("PDL overlap is K1 tail || K2 independent work. "
                "Do not drop the K2 wait.\n");
  }

  float* d_out = nullptr;
  float* d_sink = nullptr;
  float* d_scratch = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sink, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scratch, size_t(full_grid) * sizeof(float)));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::vector<float> samples;

  auto setup = [&]() {
    CUDA_CHECK(cudaMemsetAsync(d_sink, 0, size_t(cfg.n) * sizeof(float), stream));
  };

  auto run_serial = [&](int work, int tail, int grid, bool verify) -> float {
    auto launch = [&]() {
      launch_serial(stream, d_out, d_scratch, d_sink, cfg.n, tail, work, grid, cfg.block);
    };
    if (verify) {
      setup();
      launch();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      verify_sink(d_sink, cfg.n, quiet, "serial");
    }
    return time_on_stream(stream, setup, launch, cfg.warmup, cfg.runs, &samples);
  };

  auto run_pdl = [&](int work, int tail, int grid, const char* tag, bool verify) -> float {
    auto launch = [&]() {
      launch_pdl(stream, d_out, d_scratch, d_sink, cfg.n, tail, work, grid, cfg.block);
    };
    if (verify) {
      setup();
      launch();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      verify_sink(d_sink, cfg.n, quiet, tag);
    }
    return time_on_stream(stream, setup, launch, cfg.warmup, cfg.runs, &samples);
  };

  auto print_speedup = [&](const char* label, int work, int tail, float ms_s, float ms_p) {
    const float sp = (ms_p > 0.f) ? (ms_s / ms_p) : 0.f;
    if (cfg.csv_only) {
      std::printf("speedup_serial_pdl,%d,%d,%d,0,%.6f,0,0\n", cfg.n, work, tail, sp);
    } else {
      std::printf("  serial/pdl %s: %.3fx\n", label, sp);
    }
  };

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else
      std::printf("\n== sweep work (fixed tail=%d, half_grid=%d) ==\n", cfg.tail, half_grid);
    const int work_list[] = {0, 1, 8, 64, 512, 4096};
    for (int w : work_list) {
      const float ms_s = run_serial(w, cfg.tail, half_grid, true);
      print_row("serial", cfg.n, w, cfg.tail, half_grid, ms_s, samples, cfg.csv_only);
      const float ms_p = run_pdl(w, cfg.tail, half_grid, "pdl", true);
      print_row("pdl", cfg.n, w, cfg.tail, half_grid, ms_p, samples, cfg.csv_only);
      char buf[64];
      std::snprintf(buf, sizeof(buf), "@work=%d", w);
      print_speedup(buf, w, cfg.tail, ms_s, ms_p);
    }
  } else if (cfg.mode == Mode::SweepTail) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else
      std::printf("\n== sweep_tail (fixed work=%d, half_grid=%d) ==\n", cfg.work, half_grid);
    const int tail_list[] = {0, 1, 8, 64, 512, 4096};
    for (int t : tail_list) {
      const float ms_s = run_serial(cfg.work, t, half_grid, true);
      print_row("serial", cfg.n, cfg.work, t, half_grid, ms_s, samples, cfg.csv_only);
      const float ms_p = run_pdl(cfg.work, t, half_grid, "pdl", true);
      print_row("pdl", cfg.n, cfg.work, t, half_grid, ms_p, samples, cfg.csv_only);
      char buf[64];
      std::snprintf(buf, sizeof(buf), "@tail=%d", t);
      print_speedup(buf, cfg.work, t, ms_s, ms_p);
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else
      std::printf("\n== modes (work=%d tail=%d) ==\n", cfg.work, cfg.tail);
    const float ms_s = run_serial(cfg.work, cfg.tail, half_grid, true);
    print_row("serial", cfg.n, cfg.work, cfg.tail, half_grid, ms_s, samples, cfg.csv_only);
    const float ms_p = run_pdl(cfg.work, cfg.tail, half_grid, "pdl", true);
    print_row("pdl", cfg.n, cfg.work, cfg.tail, half_grid, ms_p, samples, cfg.csv_only);
    const float ms_f = run_pdl(cfg.work, cfg.tail, full_grid, "pdl_full", true);
    print_row("pdl_full", cfg.n, cfg.work, cfg.tail, full_grid, ms_f, samples, cfg.csv_only);
    if (!cfg.csv_only) {
      std::printf("speedup serial/pdl=%.3fx\n", (ms_p > 0.f) ? (ms_s / ms_p) : 0.f);
      std::printf("pdl_full/pdl=%.3fx  (>1 => full occupancy slower; not more overlap)\n",
                  (ms_p > 0.f) ? (ms_f / ms_p) : 0.f);
    } else {
      print_speedup("modes", cfg.work, cfg.tail, ms_s, ms_p);
    }
  } else if (cfg.mode == Mode::Serial) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_serial(cfg.work, cfg.tail, half_grid, true);
    print_row("serial", cfg.n, cfg.work, cfg.tail, half_grid, ms, samples, cfg.csv_only);
  } else if (cfg.mode == Mode::Pdl) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_pdl(cfg.work, cfg.tail, half_grid, "pdl", true);
    print_row("pdl", cfg.n, cfg.work, cfg.tail, half_grid, ms, samples, cfg.csv_only);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_scratch));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
