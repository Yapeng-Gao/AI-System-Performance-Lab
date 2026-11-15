/**
 * [Module D] D-03. Elementwise：手写 a+b 达到带宽顶
 *
 * a[i] = (float)(i & 255), b[i] = (float)((3 * i) & 255)。和在 0..510，float 精确。
 * n 是 256 的倍数。
 *
 *   scalar : 标量加载/存储
 *   vec4   : float4 加载/存储
 *   sweep  : 扫 n
 *   modes  : 定点两条
 *
 * 有效字节 = 3 * n * 4（两读一写）。无官方单算子。主证据：CUDA event median。
 * D-04～D-10 短章共用这套网格与计时口径。
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
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",            \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__);  \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class Mode { Scalar = 0, Vec4 = 1, Sweep = 2, Modes = 3 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Scalar: return "scalar";
    case Mode::Vec4: return "vec4";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "scalar") == 0) return Mode::Scalar;
  if (std::strcmp(s, "vec4") == 0) return Mode::Vec4;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr, "Invalid --mode=%s (expected scalar|vec4|sweep|modes)\n", s);
  std::exit(EXIT_FAILURE);
}

struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 1 << 24;
  int block = 256;
  int grid = 0;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <scalar|vec4|sweep|modes> [options]\n"
      "  --n <elems>      multiple of 256 (default 1<<24)\n"
      "  --block <thr>    default 256\n"
      "  --grid <blocks>  default 0 = min(SMs*8, tiles)\n"
      "  --runs / --warmup / --device / --csv-only\n",
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
    if (std::strcmp(argv[i], "--mode") == 0) c.mode = parse_mode(need("--mode"));
    else if (std::strcmp(argv[i], "--n") == 0) c.n = std::atoi(need("--n"));
    else if (std::strcmp(argv[i], "--block") == 0) c.block = std::atoi(need("--block"));
    else if (std::strcmp(argv[i], "--grid") == 0) c.grid = std::atoi(need("--grid"));
    else if (std::strcmp(argv[i], "--runs") == 0) c.runs = std::atoi(need("--runs"));
    else if (std::strcmp(argv[i], "--warmup") == 0) c.warmup = std::atoi(need("--warmup"));
    else if (std::strcmp(argv[i], "--device") == 0) c.device = std::atoi(need("--device"));
    else if (std::strcmp(argv[i], "--csv-only") == 0) c.csv_only = true;
    else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (c.n <= 0 || (c.n % 256) != 0) {
    std::fprintf(stderr, "ERROR: --n must be a positive multiple of 256\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.block <= 0 || c.runs <= 0 || c.warmup < 0) {
    std::fprintf(stderr, "ERROR: --block/--runs/--warmup out of range\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

static float median_of(std::vector<float> v) {
  if (v.empty()) return 0.f;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  if (n % 2 == 1) return v[n / 2];
  return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

__global__ void kernel_fill(float* a, float* b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = static_cast<float>(i & 255);
    b[i] = static_cast<float>((3 * i) & 255);
  }
}

__global__ void kernel_scalar(const float* a, const float* b, float* c, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

__global__ void kernel_vec4(const float* a, const float* b, float* c, int n) {
  const int n4 = n >> 2;
  const float4* a4 = reinterpret_cast<const float4*>(a);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  float4* c4 = reinterpret_cast<float4*>(c);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 x = a4[i];
    const float4 y = b4[i];
    c4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

static int choose_grid(int sm, int n, int block, int requested) {
  if (requested > 0) return requested;
  const int cap = std::max(1, sm * 8);
  const int tiles = std::max(1, (n + block - 1) / block);
  return std::min(cap, tiles);
}

// Two reads + one write.
static double useful_gbps(int n, float ms) {
  if (ms <= 0.f) return 0.0;
  return (double)n * 12.0 / (double)ms * 1e-6;
}

struct Buffers {
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
  int n = 0;
  int grid = 0;
  int block = 256;
  cudaStream_t stream = nullptr;
};

static void alloc_buffers(Buffers& b, int n, int grid, int block, cudaStream_t stream) {
  b.n = n;
  b.grid = grid;
  b.block = block;
  b.stream = stream;
  CUDA_CHECK(cudaMalloc(&b.a, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.b, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.c, (size_t)n * sizeof(float)));
  const int fill_grid = (n + block - 1) / block;
  kernel_fill<<<fill_grid, block, 0, stream>>>(b.a, b.b, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void free_buffers(Buffers& b) {
  cudaFree(b.a);
  cudaFree(b.b);
  cudaFree(b.c);
  b.a = b.b = b.c = nullptr;
}

static void check_add(Buffers& b, const char* tag) {
  // Spot-check head/mid/tail on host. Values are exact integers in float.
  const int probes[] = {0, 1, 255, b.n / 2, b.n - 2, b.n - 1};
  std::vector<float> hc(6);
  for (int k = 0; k < 6; ++k) {
    const int i = probes[k];
    if (i < 0 || i >= b.n) continue;
    float got = 0.f;
    CUDA_CHECK(cudaMemcpy(&got, b.c + i, sizeof(float), cudaMemcpyDeviceToHost));
    const float expect = static_cast<float>((i & 255) + ((3 * i) & 255));
    if (got != expect) {
      std::fprintf(stderr, "VERIFY FAIL %s at i=%d: got %g expect %g\n", tag, i, got, expect);
      std::exit(EXIT_FAILURE);
    }
  }
}

template <typename Launch>
static float time_median(Buffers& b, int runs, int warmup, Launch&& launch) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> samples;
  samples.reserve((size_t)runs);
  for (int i = 0; i < warmup + runs; ++i) {
    CUDA_CHECK(cudaEventRecord(start, b.stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop, b.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (i >= warmup) samples.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return median_of(samples);
}

static float time_scalar(Buffers& b, int runs, int warmup) {
  const float ms = time_median(b, runs, warmup, [&] {
    kernel_scalar<<<b.grid, b.block, 0, b.stream>>>(b.a, b.b, b.c, b.n);
  });
  check_add(b, "scalar");
  return ms;
}

static float time_vec4(Buffers& b, int runs, int warmup) {
  // grid for vec4 indexes n/4 elements; keep same CTA count as scalar for occupancy shape.
  const float ms = time_median(b, runs, warmup, [&] {
    kernel_vec4<<<b.grid, b.block, 0, b.stream>>>(b.a, b.b, b.c, b.n);
  });
  check_add(b, "vec4");
  return ms;
}

static void warn_if_tiny(float ms) {
  if (ms < 0.05f) std::fprintf(stderr, "WARN: median %.4f ms < 0.05 ms; raise --n\n", ms);
}

struct Row {
  float scalar = 0.f, vec4 = 0.f;
};

static void print_row(int n, const Row& r, bool csv_only) {
  const double sp = r.scalar / r.vec4;
  if (!csv_only) {
    std::printf(
        "n=%d scalar=%.4f ms (%.1f GB/s) vec4=%.4f ms (%.1f GB/s) scalar/vec4=%.3f\n", n, r.scalar,
        useful_gbps(n, r.scalar), r.vec4, useful_gbps(n, r.vec4), sp);
  }
  std::printf("SWEEP,%d,%.4f,%.4f,%.4f,%.1f,%.1f\n", n, r.scalar, r.vec4, sp,
              useful_gbps(n, r.scalar), useful_gbps(n, r.vec4));
  warn_if_tiny(r.vec4);
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int sm = prop.multiProcessorCount;

  if (!cfg.csv_only) {
    std::printf("GPU: %s  sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor, sm);
    std::printf("mode=%s n=%d block=%d runs=%d warmup=%d  bytes=3*n*4\n", mode_name(cfg.mode),
                cfg.n, cfg.block, cfg.runs, cfg.warmup);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto make = [&](int n) {
    // For vec4 path, tiles counted in float elements still; grid capped by SMs*8.
    const int grid = choose_grid(sm, n, cfg.block, cfg.grid);
    Buffers b;
    alloc_buffers(b, n, grid, cfg.block, stream);
    return b;
  };

  auto run_both = [&](Buffers& b) {
    Row r;
    r.scalar = time_scalar(b, cfg.runs, cfg.warmup);
    r.vec4 = time_vec4(b, cfg.runs, cfg.warmup);
    return r;
  };

  if (cfg.mode == Mode::Sweep) {
    if (!cfg.csv_only) {
      std::printf("CSV: n,scalar_ms,vec4_ms,scalar_over_vec4,scalar_GBps,vec4_GBps\n");
    }
    std::printf("SWEEP,n,scalar_ms,vec4_ms,scalar_over_vec4,scalar_GBps,vec4_GBps\n");
    const int sizes[] = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 27, 1 << 28};
    for (int n : sizes) {
      Buffers b = make(n);
      print_row(n, run_both(b), cfg.csv_only);
      free_buffers(b);
    }
  } else if (cfg.mode == Mode::Modes) {
    Buffers b = make(cfg.n);
    if (!cfg.csv_only) std::printf("modes n=%d\n", cfg.n);
    print_row(cfg.n, run_both(b), cfg.csv_only);
    free_buffers(b);
  } else if (cfg.mode == Mode::Scalar) {
    Buffers b = make(cfg.n);
    const float ms = time_scalar(b, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("n=%d scalar=%.4f ms (%.1f GB/s)\n", cfg.n, ms, useful_gbps(cfg.n, ms));
    }
    std::printf("SINGLE,scalar,%d,%.4f,%.1f\n", cfg.n, ms, useful_gbps(cfg.n, ms));
    warn_if_tiny(ms);
    free_buffers(b);
  } else {
    Buffers b = make(cfg.n);
    const float ms = time_vec4(b, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("n=%d vec4=%.4f ms (%.1f GB/s)\n", cfg.n, ms, useful_gbps(cfg.n, ms));
    }
    std::printf("SINGLE,vec4,%d,%.4f,%.1f\n", cfg.n, ms, useful_gbps(cfg.n, ms));
    warn_if_tiny(ms);
    free_buffers(b);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
