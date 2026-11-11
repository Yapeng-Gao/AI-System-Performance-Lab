/**
 * [Module C] C-09. Named Barrier：子集会合、单边 arrive 与何时叠不上
 *
 * 主命题：一个 CTA 里 warp0 生产、warp1 消费。
 *   cta   : 两道 __syncthreads 把 tail 和 work 串成关键路径
 *   named : 写完 SMEM 就 barrier.arrive，tail / work 在 arrive 之后，才能叠
 * 反例：
 *   named_pre : 长 FMA 放在 arrive 之前，应贴 cta
 *   indep     : 两组互不依赖，kernel 墙钟仍由慢组决定
 *
 * 主证据：CUDA event median；event 与 kernel 同 stream。
 * 硬件：sm_70+。每个 SM 一个 block，避免别的 block 把 barrier 上空档补上。
 * 编号 1 / 2。0 留给 __syncthreads。计数含两边线程。
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

enum class Mode { Cta = 0, Named = 1, Sweep = 2, SweepTail = 3, Modes = 4 };

static constexpr int kBlock = 64;
static constexpr int kCount = 64;
static constexpr int kFull = 1;
static constexpr int kEmpty = 2;
static constexpr int kIndepBlock = 256;
static constexpr int kIndepCount = 128;
static constexpr int kIndepPhases = 128;
static constexpr int kIndepWorkA = 2048;
static constexpr int kIndepWorkB = 32;

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Cta: return "cta";
    case Mode::Named: return "named";
    case Mode::Sweep: return "sweep";
    case Mode::SweepTail: return "sweep_tail";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "cta") == 0) return Mode::Cta;
  if (std::strcmp(s, "named") == 0) return Mode::Named;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "sweep_tail") == 0) return Mode::SweepTail;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected cta|named|sweep|sweep_tail|modes)\n", s);
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

__device__ inline void barrier_sync(int id, int count) {
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ inline void barrier_arrive(int id, int count) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__global__ void kernel_cta(unsigned long long* __restrict__ sums, float* __restrict__ accs,
                           int phases, int tail, int work) {
  __shared__ int token;
  const bool producer = (threadIdx.x >> 5) == 0;
  const int lane = threadIdx.x & 31;
  float acc = 1.f;
  unsigned long long sum = 0;
  for (int p = 0; p < phases; ++p) {
    if (producer) {
      acc = burn(acc, tail);
      if (lane == 0) token = p + 1;
    }
    __syncthreads();
    if (!producer) {
      if (lane == 0) sum += static_cast<unsigned>(token);
      acc = burn(acc, work);
    }
    __syncthreads();
  }
  if (lane == 0 && !producer) sums[blockIdx.x] = sum;
  if (lane == 0) accs[blockIdx.x * 2 + (producer ? 0 : 1)] = acc;
}

template <bool kPre>
__global__ void kernel_named(unsigned long long* __restrict__ sums, float* __restrict__ accs,
                             int phases, int tail, int work) {
  __shared__ int token;
  const bool producer = (threadIdx.x >> 5) == 0;
  const int lane = threadIdx.x & 31;
  float acc = 1.f;
  unsigned long long sum = 0;
  if (producer) {
    for (int p = 0; p < phases; ++p) {
      if (p > 0) barrier_sync(kEmpty, kCount);
      if (kPre) acc = burn(acc, tail);
      if (lane == 0) token = p + 1;
      barrier_arrive(kFull, kCount);
      if (!kPre) acc = burn(acc, tail);
    }
  } else {
    for (int p = 0; p < phases; ++p) {
      barrier_sync(kFull, kCount);
      if (lane == 0) sum += static_cast<unsigned>(token);
      if (kPre) acc = burn(acc, work);
      barrier_arrive(kEmpty, kCount);
      if (!kPre) acc = burn(acc, work);
    }
  }
  if (lane == 0 && !producer) sums[blockIdx.x] = sum;
  if (lane == 0) accs[blockIdx.x * 2 + (producer ? 0 : 1)] = acc;
}

__global__ void kernel_indep_cta(float* __restrict__ accs, int phases, int work_a, int work_b) {
  const bool group_a = (threadIdx.x >> 5) < 4;
  const int lane = threadIdx.x & 31;
  const int iters = group_a ? work_a : work_b;
  float acc = 1.f;
  for (int p = 0; p < phases; ++p) {
    acc = burn(acc, iters);
    __syncthreads();
  }
  if (lane == 0) accs[blockIdx.x * 8 + (threadIdx.x >> 5)] = acc;
}

__global__ void kernel_indep_named(float* __restrict__ accs, int phases, int work_a, int work_b) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool group_a = warp < 4;
  const int iters = group_a ? work_a : work_b;
  const int bar = group_a ? kFull : kEmpty;
  float acc = 1.f;
  for (int p = 0; p < phases; ++p) {
    acc = burn(acc, iters);
    barrier_sync(bar, kIndepCount);
  }
  if (lane == 0) accs[blockIdx.x * 8 + warp] = acc;
}

struct BenchConfig {
  Mode mode = Mode::Sweep;
  int phases = 1024;
  int work = 512;
  int tail = 512;
  int grid = 0;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <cta|named|sweep|sweep_tail|modes> [options]\n"
      "  --phases <n>    handshake iterations (default 1024)\n"
      "  --work <iters>  consumer FMA after arrive (default 512)\n"
      "  --tail <iters>  producer FMA after arrive (default 512)\n"
      "  --grid <blocks> override one-block-per-SM (default 0 = auto)\n"
      "  --runs <n>      timed runs (default 7)\n"
      "  --warmup <n>    warmup runs (default 2)\n"
      "  --device <id>   GPU id (default 0)\n"
      "  --csv-only      only print CSV line(s)\n"
      "Block is fixed at 64 (1 producer warp + 1 consumer warp).\n",
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
    } else if (std::strcmp(argv[i], "--phases") == 0) {
      c.phases = std::atoi(need("--phases"));
    } else if (std::strcmp(argv[i], "--work") == 0) {
      c.work = std::atoi(need("--work"));
    } else if (std::strcmp(argv[i], "--tail") == 0) {
      c.tail = std::atoi(need("--tail"));
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
  if (c.phases < 1 || c.phases > 100000 || c.work < 0 || c.tail < 0) {
    std::fprintf(stderr, "ERROR: invalid --phases / --work / --tail\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
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

static unsigned long long expected_sum(int phases) {
  return static_cast<unsigned long long>(phases) * static_cast<unsigned long long>(phases + 1) / 2ull;
}

static void verify_sums(unsigned long long* d_sums, int grid, int phases, bool quiet,
                        const char* tag) {
  std::vector<unsigned long long> got(static_cast<size_t>(grid));
  CUDA_CHECK(cudaMemcpy(got.data(), d_sums, size_t(grid) * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  const unsigned long long expect = expected_sum(phases);
  for (int i = 0; i < grid; ++i) {
    if (got[static_cast<size_t>(i)] != expect) {
      std::fprintf(stderr, "ERROR: %s checksum block %d got=%llu expect=%llu\n", tag, i,
                   static_cast<unsigned long long>(got[static_cast<size_t>(i)]),
                   static_cast<unsigned long long>(expect));
      std::exit(EXIT_FAILURE);
    }
  }
  if (!quiet) std::printf("verify %s OK (grid=%d sum=%llu)\n", tag, grid, expect);
}

static void verify_finite(float* d_accs, int n, const char* tag) {
  std::vector<float> got(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(got.data(), d_accs, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(got[static_cast<size_t>(i)])) {
      std::fprintf(stderr, "ERROR: %s non-finite acc at %d\n", tag, i);
      std::exit(EXIT_FAILURE);
    }
  }
}

static void print_row(const char* tag, int phases, int work, int tail, int grid, float med_ms,
                      const std::vector<float>& samples, bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (med_ms < 0.05f) {
    std::fprintf(stderr,
                 "WARN: %s median %.4f ms < 0.05 ms. Increase --phases before reading the ratio.\n",
                 tag, med_ms);
  }
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", tag, phases, work, tail, grid, med_ms, p10,
                p90);
  } else {
    std::printf("%-12s phases=%d work=%d tail=%d grid=%d | median=%.4f ms (p10=%.4f p90=%.4f)\n",
                tag, phases, work, tail, grid, med_ms, p10, p90);
  }
}

static void print_speedup(const char* label, float ms_cta, float ms_named, bool csv_only) {
  const float sp = (ms_named > 0.f) ? (ms_cta / ms_named) : 0.f;
  if (csv_only) {
    std::printf("speedup,%s,%.6f\n", label, sp);
  } else {
    std::printf("  cta/named %s: %.3fx\n", label, sp);
  }
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  if (prop.major < 7) {
    std::fprintf(stderr,
                 "ERROR: C-09 named barrier subset sync needs sm_70+ (got sm_%d%d / %s). "
                 "On sm_6x and below, unaligned barrier.sync is defined as aligned.\n",
                 prop.major, prop.minor, prop.name);
    return EXIT_FAILURE;
  }

  const int grid = (cfg.grid > 0) ? cfg.grid : std::max(1, prop.multiProcessorCount);
  const bool quiet = cfg.csv_only;
  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d | grid=%d (1 block/SM unless --grid)\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount, grid);
    std::printf("mode=%s phases=%d work=%d tail=%d block=%d count=%d full=%d empty=%d "
                "runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.phases, cfg.work, cfg.tail, kBlock, kCount, kFull, kEmpty,
                cfg.runs, cfg.warmup);
    std::printf("Overlap is post-arrive tail || post-arrive work. "
                "Do not use barrier 0. Count includes both warps.\n");
  }

  unsigned long long* d_sums = nullptr;
  float* d_accs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sums, size_t(grid) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_accs, size_t(grid) * 8 * sizeof(float)));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::vector<float> samples;

  auto setup = [&]() {
    CUDA_CHECK(cudaMemsetAsync(d_sums, 0, size_t(grid) * sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(d_accs, 0, size_t(grid) * 8 * sizeof(float), stream));
  };

  auto run_cta = [&](int work, int tail, bool verify) -> float {
    auto launch = [&]() {
      kernel_cta<<<grid, kBlock, 0, stream>>>(d_sums, d_accs, cfg.phases, tail, work);
      CUDA_CHECK(cudaGetLastError());
    };
    if (verify) {
      setup();
      launch();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      verify_sums(d_sums, grid, cfg.phases, quiet, "cta");
    }
    return time_on_stream(stream, setup, launch, cfg.warmup, cfg.runs, &samples);
  };

  auto run_named = [&](int work, int tail, bool pre, const char* tag, bool verify) -> float {
    auto launch = [&]() {
      if (pre) {
        kernel_named<true><<<grid, kBlock, 0, stream>>>(d_sums, d_accs, cfg.phases, tail, work);
      } else {
        kernel_named<false><<<grid, kBlock, 0, stream>>>(d_sums, d_accs, cfg.phases, tail, work);
      }
      CUDA_CHECK(cudaGetLastError());
    };
    if (verify) {
      setup();
      launch();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      verify_sums(d_sums, grid, cfg.phases, quiet, tag);
    }
    return time_on_stream(stream, setup, launch, cfg.warmup, cfg.runs, &samples);
  };

  auto run_pair = [&](int work, int tail, bool verify) {
    const float ms_c = run_cta(work, tail, verify);
    print_row("cta", cfg.phases, work, tail, grid, ms_c, samples, cfg.csv_only);
    const float ms_n = run_named(work, tail, false, "named", verify);
    print_row("named", cfg.phases, work, tail, grid, ms_n, samples, cfg.csv_only);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "work=%d,tail=%d", work, tail);
    print_speedup(buf, ms_c, ms_n, cfg.csv_only);
    if (!cfg.csv_only) {
      std::printf("PAIR,%d,%d,%.6f,%.6f,%.6f\n", work, tail, ms_c, ms_n,
                  (ms_n > 0.f) ? (ms_c / ms_n) : 0.f);
    }
  };

  if (cfg.mode == Mode::Sweep || cfg.mode == Mode::SweepTail) {
    if (cfg.csv_only) {
      std::printf("tag,phases,work,tail,grid,median_ms,p10_ms,p90_ms\n");
    }
    const int list[] = {0, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    if (cfg.mode == Mode::Sweep) {
      if (!quiet) {
        std::printf("\n== sweep work (fixed tail=%d, phases=%d, grid=%d) ==\n", cfg.tail,
                    cfg.phases, grid);
      }
      for (int w : list) run_pair(w, cfg.tail, true);
    } else {
      if (!quiet) {
        std::printf("\n== sweep_tail (fixed work=%d, phases=%d, grid=%d) ==\n", cfg.work,
                    cfg.phases, grid);
      }
      for (int t : list) run_pair(cfg.work, t, true);
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) std::printf("tag,phases,work,tail,grid,median_ms,p10_ms,p90_ms\n");
    else std::printf("\n== modes (work=%d tail=%d) ==\n", cfg.work, cfg.tail);
    run_pair(cfg.work, cfg.tail, true);
    const float ms_pre = run_named(cfg.work, cfg.tail, true, "named_pre", true);
    print_row("named_pre", cfg.phases, cfg.work, cfg.tail, grid, ms_pre, samples, cfg.csv_only);

    if (!quiet) {
      std::printf("\n== indep (block=%d work_a=%d work_b=%d phases=%d) kernel wall should ~1x ==\n",
                  kIndepBlock, kIndepWorkA, kIndepWorkB, kIndepPhases);
    }
    auto launch_ic = [&]() {
      kernel_indep_cta<<<grid, kIndepBlock, 0, stream>>>(d_accs, kIndepPhases, kIndepWorkA,
                                                        kIndepWorkB);
      CUDA_CHECK(cudaGetLastError());
    };
    auto launch_in = [&]() {
      kernel_indep_named<<<grid, kIndepBlock, 0, stream>>>(d_accs, kIndepPhases, kIndepWorkA,
                                                          kIndepWorkB);
      CUDA_CHECK(cudaGetLastError());
    };
    setup();
    launch_ic();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    verify_finite(d_accs, grid * 8, "indep_cta");
    const float ms_ic = time_on_stream(stream, setup, launch_ic, cfg.warmup, cfg.runs, &samples);
    print_row("indep_cta", kIndepPhases, kIndepWorkA, kIndepWorkB, grid, ms_ic, samples,
              cfg.csv_only);
    setup();
    launch_in();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    verify_finite(d_accs, grid * 8, "indep_named");
    const float ms_in = time_on_stream(stream, setup, launch_in, cfg.warmup, cfg.runs, &samples);
    print_row("indep_named", kIndepPhases, kIndepWorkA, kIndepWorkB, grid, ms_in, samples,
              cfg.csv_only);
    print_speedup("indep", ms_ic, ms_in, cfg.csv_only);
  } else if (cfg.mode == Mode::Cta) {
    if (cfg.csv_only) std::printf("tag,phases,work,tail,grid,median_ms,p10_ms,p90_ms\n");
    const float ms = run_cta(cfg.work, cfg.tail, true);
    print_row("cta", cfg.phases, cfg.work, cfg.tail, grid, ms, samples, cfg.csv_only);
  } else if (cfg.mode == Mode::Named) {
    if (cfg.csv_only) std::printf("tag,phases,work,tail,grid,median_ms,p10_ms,p90_ms\n");
    const float ms = run_named(cfg.work, cfg.tail, false, "named", true);
    print_row("named", cfg.phases, cfg.work, cfg.tail, grid, ms, samples, cfg.csv_only);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_accs));
  CUDA_CHECK(cudaFree(d_sums));
  return EXIT_SUCCESS;
}
