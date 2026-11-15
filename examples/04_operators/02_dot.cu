/**
 * [Module D] D-02. Dot：手写点积达到 cublasSdot
 *
 * x[i] = (float)(i & 255), y[i] = (float)((3 * i) & 255)。
 * n 是 256 的倍数。host double 真值 = (n/256) * sum_{k=0..255} k*((3k)&255)。
 * 相对误差须 <= 1e-3。
 *
 *   scalar  : grid-stride float 乘加，block 内 shfl+SMEM，每 block 一次 atomicAdd
 *   vec4    : float4 成对加载再乘加
 *   cublas  : cublasSdot，POINTER_MODE_DEVICE，handle/stream 在计时外
 *   sweep   : 扫 n
 *   modes   : 定点三条路径
 *
 * 主证据：CUDA event median。不做 GEMV。
 */

#include <cublas_v2.h>
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

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t st__ = (call);                                                \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS Error: status=%d at %s:%d\n", (int)st__,     \
                   __FILE__, __LINE__);                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class Mode { Scalar = 0, Vec4 = 1, Cublas = 2, Sweep = 3, Modes = 4 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Scalar: return "scalar";
    case Mode::Vec4: return "vec4";
    case Mode::Cublas: return "cublas";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "scalar") == 0) return Mode::Scalar;
  if (std::strcmp(s, "vec4") == 0) return Mode::Vec4;
  if (std::strcmp(s, "cublas") == 0) return Mode::Cublas;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr, "Invalid --mode=%s (expected scalar|vec4|cublas|sweep|modes)\n", s);
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
      "Usage: %s --mode <scalar|vec4|cublas|sweep|modes> [options]\n"
      "  --n <elems>      element count, multiple of 256 (default 1<<24)\n"
      "  --block <thr>    threads/block (default 256, must be 256)\n"
      "  --grid <blocks>  grid size (default 0 = min(SMs*8, tiles))\n"
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
  if (c.n <= 0 || (c.n % 256) != 0) {
    std::fprintf(stderr, "ERROR: --n must be a positive multiple of 256\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.block != 256) {
    std::fprintf(stderr, "ERROR: --block must be 256\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.runs <= 0 || c.warmup < 0) {
    std::fprintf(stderr, "ERROR: --runs/--warmup out of range\n");
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

// One period 0..255 of x*y with the fill pattern above.
static double period_dot() {
  double s = 0.0;
  for (int k = 0; k < 256; ++k) {
    s += (double)k * (double)((3 * k) & 255);
  }
  return s;
}

static double host_ref(int n, double period) { return (double)(n / 256) * period; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  v = warp_sum(v);
  __shared__ float warp_sums[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  float x = 0.f;
  if (warp == 0) {
    x = (lane < 8) ? warp_sums[lane] : 0.f;
    x = warp_sum(x);
  }
  return x;
}

__device__ __forceinline__ float acc_scalar(const float* x, const float* y, int n) {
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    acc += x[i] * y[i];
  }
  return acc;
}

__device__ __forceinline__ float acc_vec4(const float* x, const float* y, int n) {
  const int n4 = n >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* y4 = reinterpret_cast<const float4*>(y);
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 a = x4[i];
    const float4 b = y4[i];
    acc += (a.x * b.x + a.y * b.y) + (a.z * b.z + a.w * b.w);
  }
  return acc;
}

__global__ void kernel_scalar(const float* x, const float* y, float* out, int n) {
  const float sum = block_sum(acc_scalar(x, y, n));
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void kernel_vec4(const float* x, const float* y, float* out, int n) {
  const float sum = block_sum(acc_vec4(x, y, n));
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void kernel_fill(float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = static_cast<float>(i & 255);
    y[i] = static_cast<float>((3 * i) & 255);
  }
}

static int choose_grid(int sm, int n, int block, int requested) {
  if (requested > 0) return requested;
  const int cap = std::max(1, sm * 8);
  const int tiles = std::max(1, (n + block - 1) / block);
  return std::min(cap, tiles);
}

// Two vectors: useful bytes = 2 * n * 4.
static double useful_gbps(int n, float ms) {
  if (ms <= 0.f) return 0.0;
  return (double)n * 8.0 / (double)ms * 1e-6;
}

struct Buffers {
  float* x = nullptr;
  float* y = nullptr;
  float* out = nullptr;
  int n = 0;
  int grid = 0;
  int block = 256;
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
};

static void alloc_buffers(Buffers& b, int n, int grid, int block, cudaStream_t stream,
                          cublasHandle_t handle) {
  b.n = n;
  b.grid = grid;
  b.block = block;
  b.stream = stream;
  b.handle = handle;
  CUDA_CHECK(cudaMalloc(&b.x, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.y, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.out, sizeof(float)));
  const int fill_grid = (n + block - 1) / block;
  kernel_fill<<<fill_grid, block, 0, stream>>>(b.x, b.y, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void free_buffers(Buffers& b) {
  cudaFree(b.x);
  cudaFree(b.y);
  cudaFree(b.out);
  b.x = b.y = b.out = nullptr;
}

static double check_dot(Buffers& b, double ref, const char* tag) {
  float got = 0.f;
  CUDA_CHECK(cudaMemcpy(&got, b.out, sizeof(float), cudaMemcpyDeviceToHost));
  const double rel = std::abs((double)got - ref) / std::max(1.0, std::abs(ref));
  if (rel > 1e-3) {
    std::fprintf(stderr, "VERIFY FAIL %s: got %.9g ref %.9g rel %.3e\n", tag, got, ref, rel);
    std::exit(EXIT_FAILURE);
  }
  return rel;
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

static float time_scalar(Buffers& b, int runs, int warmup, double ref, double* rel) {
  const float ms = time_median(b, runs, warmup, [&] {
    CUDA_CHECK(cudaMemsetAsync(b.out, 0, sizeof(float), b.stream));
    kernel_scalar<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.out, b.n);
  });
  *rel = check_dot(b, ref, "scalar");
  return ms;
}

static float time_vec4(Buffers& b, int runs, int warmup, double ref, double* rel) {
  const float ms = time_median(b, runs, warmup, [&] {
    CUDA_CHECK(cudaMemsetAsync(b.out, 0, sizeof(float), b.stream));
    kernel_vec4<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.out, b.n);
  });
  *rel = check_dot(b, ref, "vec4");
  return ms;
}

static float time_cublas(Buffers& b, int runs, int warmup, double ref, double* rel) {
  const float ms = time_median(b, runs, warmup, [&] {
    CUBLAS_CHECK(cublasSdot(b.handle, b.n, b.x, 1, b.y, 1, b.out));
  });
  *rel = check_dot(b, ref, "cublas");
  return ms;
}

static void warn_if_tiny(float ms) {
  if (ms < 0.05f) {
    std::fprintf(stderr, "WARN: median %.4f ms < 0.05 ms; raise --n\n", ms);
  }
}

struct Row {
  float scalar = 0.f, vec4 = 0.f, cublas = 0.f;
  double max_rel = 0.0;
};

static void print_row(int n, const Row& r, bool csv_only) {
  const double v_sp = r.scalar / r.vec4;
  const double c_sp = r.scalar / r.cublas;
  const double vc = r.vec4 / r.cublas;
  if (!csv_only) {
    std::printf(
        "n=%d scalar=%.4f ms (%.1f GB/s) vec4=%.4f ms (%.1f GB/s) cublas=%.4f ms (%.1f GB/s) "
        "scalar/vec4=%.3f scalar/cublas=%.3f vec4/cublas=%.3f max_rel=%.3e\n",
        n, r.scalar, useful_gbps(n, r.scalar), r.vec4, useful_gbps(n, r.vec4), r.cublas,
        useful_gbps(n, r.cublas), v_sp, c_sp, vc, r.max_rel);
  }
  std::printf("SWEEP,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e\n", n, r.scalar, r.vec4, r.cublas, v_sp,
              c_sp, vc, r.max_rel);
  warn_if_tiny(r.vec4);
}

static Row run_all(Buffers& b, int runs, int warmup, double ref) {
  Row row{};
  double rel = 0.0;
  row.scalar = time_scalar(b, runs, warmup, ref, &rel);
  row.max_rel = rel;
  row.vec4 = time_vec4(b, runs, warmup, ref, &rel);
  row.max_rel = std::max(row.max_rel, rel);
  row.cublas = time_cublas(b, runs, warmup, ref, &rel);
  row.max_rel = std::max(row.max_rel, rel);
  return row;
}

static void run_single(Buffers& b, Mode m, int runs, int warmup, double ref, bool csv_only) {
  double rel = 0.0;
  float ms = 0.f;
  const char* tag = mode_name(m);
  switch (m) {
    case Mode::Scalar: ms = time_scalar(b, runs, warmup, ref, &rel); break;
    case Mode::Vec4: ms = time_vec4(b, runs, warmup, ref, &rel); break;
    case Mode::Cublas: ms = time_cublas(b, runs, warmup, ref, &rel); break;
    default: return;
  }
  if (!csv_only) {
    std::printf("n=%d %s=%.4f ms (%.1f GB/s) rel=%.3e\n", b.n, tag, ms, useful_gbps(b.n, ms), rel);
  }
  std::printf("SINGLE,%s,%d,%.4f,%.3e\n", tag, b.n, ms, rel);
  warn_if_tiny(ms);
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int sm = prop.multiProcessorCount;
  const double period = period_dot();

  if (!cfg.csv_only) {
    std::printf("GPU: %s  sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor, sm);
    std::printf("mode=%s n=%d block=%d runs=%d warmup=%d dtype=f32 rtol=1e-3\n", mode_name(cfg.mode),
                cfg.n, cfg.block, cfg.runs, cfg.warmup);
    std::printf("period_dot=%.0f  (host_ref = (n/256)*period)\n", period);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

  auto make_buffers = [&](int n) {
    const int grid = choose_grid(sm, n, cfg.block, cfg.grid);
    Buffers b;
    alloc_buffers(b, n, grid, cfg.block, stream, handle);
    return b;
  };

  if (cfg.mode == Mode::Sweep) {
    if (!cfg.csv_only) {
      std::printf(
          "CSV: n,scalar_ms,vec4_ms,cublas_ms,scalar/vec4,scalar/cublas,vec4/cublas,max_rel\n");
    }
    std::printf(
        "SWEEP,n,scalar_ms,vec4_ms,cublas_ms,scalar_over_vec4,scalar_over_cublas,vec4_over_cublas,"
        "max_rel\n");
    const int sizes[] = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 27, 1 << 28};
    for (int n : sizes) {
      Buffers b = make_buffers(n);
      print_row(n, run_all(b, cfg.runs, cfg.warmup, host_ref(n, period)), cfg.csv_only);
      free_buffers(b);
    }
  } else if (cfg.mode == Mode::Modes) {
    Buffers b = make_buffers(cfg.n);
    if (!cfg.csv_only) std::printf("modes n=%d\n", cfg.n);
    print_row(cfg.n, run_all(b, cfg.runs, cfg.warmup, host_ref(cfg.n, period)), cfg.csv_only);
    free_buffers(b);
  } else {
    Buffers b = make_buffers(cfg.n);
    run_single(b, cfg.mode, cfg.runs, cfg.warmup, host_ref(cfg.n, period), cfg.csv_only);
    free_buffers(b);
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
