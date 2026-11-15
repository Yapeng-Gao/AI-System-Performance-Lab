/**
 * [Module D] D-07. GELU：tanh 近似与 erf，是否掉出带宽（短章）
 *
 * 精确：y = 0.5 * x * (1 + erf(x / sqrt(2)))
 * 近似：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * x[i] = ((i & 255) / 255.f) * 8.f - 4.f   // [-4, 4]
 * n 是 256 的倍数。
 *
 *   scalar|vec4|sweep|modes|bwd   +  --formula tanh|erf  (默认 tanh)
 *
 * 有效字节：前向 2*n*4；反传 3*n*4。无 legacy cuDNN Activation（无 GELU）。
 * 网格 / block 口径同 D-03；不重讲向量化。
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

enum class Mode { Scalar = 0, Vec4 = 1, Sweep = 2, Modes = 3, Bwd = 4 };
enum class Formula { Tanh = 0, Erf = 1 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Scalar: return "scalar";
    case Mode::Vec4: return "vec4";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    case Mode::Bwd: return "bwd";
    default: return "unknown";
  }
}

static const char* formula_name(Formula f) { return f == Formula::Tanh ? "tanh" : "erf"; }

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "scalar") == 0) return Mode::Scalar;
  if (std::strcmp(s, "vec4") == 0) return Mode::Vec4;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  if (std::strcmp(s, "bwd") == 0) return Mode::Bwd;
  std::fprintf(stderr, "Invalid --mode=%s (expected scalar|vec4|sweep|modes|bwd)\n", s);
  std::exit(EXIT_FAILURE);
}

static Formula parse_formula(const char* s) {
  if (std::strcmp(s, "tanh") == 0) return Formula::Tanh;
  if (std::strcmp(s, "erf") == 0) return Formula::Erf;
  std::fprintf(stderr, "Invalid --formula=%s (expected tanh|erf)\n", s);
  std::exit(EXIT_FAILURE);
}

struct BenchConfig {
  Mode mode = Mode::Sweep;
  Formula formula = Formula::Tanh;
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
      "Usage: %s --mode <scalar|vec4|sweep|modes|bwd> [options]\n"
      "  --formula <tanh|erf>  default tanh (production approx); erf = exact\n"
      "  --n <elems>      multiple of 256 (default 1<<24)\n"
      "  --block <thr>    default 256\n"
      "  --grid <blocks>  default 0 = min(SMs*8, tiles)\n"
      "  --runs / --warmup / --device / --csv-only\n"
      "Note: sweep always times both tanh and erf vec4 (and scalar for each).\n",
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
    else if (std::strcmp(argv[i], "--formula") == 0) c.formula = parse_formula(need("--formula"));
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

static __host__ __device__ __forceinline__ float x_at(int i) {
  return (static_cast<float>(i & 255) / 255.f) * 8.f - 4.f;
}

// Exact GELU.
static __host__ __device__ __forceinline__ float gelu_erf_f(float x) {
  return 0.5f * x * (1.f + erff(x * 0.7071067811865476f));  // 1/sqrt(2)
}

// Tanh approximation (GPT / ONNX approximate="tanh").
static __host__ __device__ __forceinline__ float gelu_tanh_f(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  const float c = 0.044715f;
  const float inner = k * (x + c * x * x * x);
  return 0.5f * x * (1.f + tanhf(inner));
}

static __host__ __device__ __forceinline__ float gelu_tanh_bwd_f(float x, float dy) {
  const float k = 0.7978845608028654f;
  const float c = 0.044715f;
  const float inner = k * (x + c * x * x * x);
  const float t = tanhf(inner);
  const float dt = (1.f - t * t) * k * (1.f + 3.f * c * x * x);
  return dy * (0.5f * (1.f + t) + 0.5f * x * dt);
}

static __host__ __device__ __forceinline__ float gelu_erf_bwd_f(float x, float dy) {
  const float inv_sqrt2 = 0.7071067811865476f;
  const float cdf = 0.5f * (1.f + erff(x * inv_sqrt2));
  const float pdf = 0.3989422804014327f * expf(-0.5f * x * x);  // 1/sqrt(2*pi)
  return dy * (cdf + x * pdf);
}

__global__ void kernel_fill_x(float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x_at(i);
}

__global__ void kernel_fill_ones(float* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.f;
}

__global__ void kernel_fwd_scalar_tanh(const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    y[i] = gelu_tanh_f(x[i]);
  }
}

__global__ void kernel_fwd_vec4_tanh(const float* x, float* y, int n) {
  const int n4 = n >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* y4 = reinterpret_cast<float4*>(y);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 a = x4[i];
    y4[i] = make_float4(gelu_tanh_f(a.x), gelu_tanh_f(a.y), gelu_tanh_f(a.z), gelu_tanh_f(a.w));
  }
}

__global__ void kernel_fwd_scalar_erf(const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    y[i] = gelu_erf_f(x[i]);
  }
}

__global__ void kernel_fwd_vec4_erf(const float* x, float* y, int n) {
  const int n4 = n >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* y4 = reinterpret_cast<float4*>(y);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 a = x4[i];
    y4[i] = make_float4(gelu_erf_f(a.x), gelu_erf_f(a.y), gelu_erf_f(a.z), gelu_erf_f(a.w));
  }
}

__global__ void kernel_bwd_scalar_tanh(const float* x, const float* dy, float* dx, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    dx[i] = gelu_tanh_bwd_f(x[i], dy[i]);
  }
}

__global__ void kernel_bwd_vec4_tanh(const float* x, const float* dy, float* dx, int n) {
  const int n4 = n >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* dy4 = reinterpret_cast<const float4*>(dy);
  float4* dx4 = reinterpret_cast<float4*>(dx);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 a = x4[i];
    const float4 g = dy4[i];
    dx4[i] = make_float4(gelu_tanh_bwd_f(a.x, g.x), gelu_tanh_bwd_f(a.y, g.y),
                         gelu_tanh_bwd_f(a.z, g.z), gelu_tanh_bwd_f(a.w, g.w));
  }
}

__global__ void kernel_bwd_scalar_erf(const float* x, const float* dy, float* dx, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    dx[i] = gelu_erf_bwd_f(x[i], dy[i]);
  }
}

__global__ void kernel_bwd_vec4_erf(const float* x, const float* dy, float* dx, int n) {
  const int n4 = n >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* dy4 = reinterpret_cast<const float4*>(dy);
  float4* dx4 = reinterpret_cast<float4*>(dx);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 a = x4[i];
    const float4 g = dy4[i];
    dx4[i] = make_float4(gelu_erf_bwd_f(a.x, g.x), gelu_erf_bwd_f(a.y, g.y),
                         gelu_erf_bwd_f(a.z, g.z), gelu_erf_bwd_f(a.w, g.w));
  }
}

static int choose_grid(int sm, int n, int block, int requested) {
  if (requested > 0) return requested;
  const int cap = std::max(1, sm * 8);
  const int tiles = std::max(1, (n + block - 1) / block);
  return std::min(cap, tiles);
}

static double fwd_gbps(int n, float ms) {
  if (ms <= 0.f) return 0.0;
  return (double)n * 8.0 / (double)ms * 1e-6;
}

static double bwd_gbps(int n, float ms) {
  if (ms <= 0.f) return 0.0;
  return (double)n * 12.0 / (double)ms * 1e-6;
}

struct Buffers {
  float* x = nullptr;
  float* y = nullptr;
  float* dy = nullptr;
  float* dx = nullptr;
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
  CUDA_CHECK(cudaMalloc(&b.x, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.y, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.dy, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.dx, (size_t)n * sizeof(float)));
  const int fill_grid = (n + block - 1) / block;
  kernel_fill_x<<<fill_grid, block, 0, stream>>>(b.x, n);
  kernel_fill_ones<<<fill_grid, block, 0, stream>>>(b.dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void free_buffers(Buffers& b) {
  cudaFree(b.x);
  cudaFree(b.y);
  cudaFree(b.dy);
  cudaFree(b.dx);
  b.x = b.y = b.dy = b.dx = nullptr;
}

static double host_gelu_erf(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

static double host_gelu_tanh(double x) {
  const double k = std::sqrt(2.0 / 3.14159265358979323846);
  const double c = 0.044715;
  return 0.5 * x * (1.0 + std::tanh(k * (x + c * x * x * x)));
}

static double host_bwd_tanh(double x) {
  const double k = std::sqrt(2.0 / 3.14159265358979323846);
  const double c = 0.044715;
  const double inner = k * (x + c * x * x * x);
  const double t = std::tanh(inner);
  const double dt = (1.0 - t * t) * k * (1.0 + 3.0 * c * x * x);
  return 0.5 * (1.0 + t) + 0.5 * x * dt;
}

static double host_bwd_erf(double x) {
  const double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
  const double pdf = (1.0 / std::sqrt(2.0 * 3.14159265358979323846)) * std::exp(-0.5 * x * x);
  return cdf + x * pdf;
}

static void check_fwd(Buffers& b, Formula f, const char* tag) {
  const int probes[] = {0, 1, 128, b.n / 2, b.n - 2, b.n - 1};
  double max_abs_vs_erf = 0.0;
  double max_rel_vs_erf = 0.0;
  for (int k = 0; k < 6; ++k) {
    const int i = probes[k];
    if (i < 0 || i >= b.n) continue;
    float got = 0.f;
    CUDA_CHECK(cudaMemcpy(&got, b.y + i, sizeof(float), cudaMemcpyDeviceToHost));
    const double x = static_cast<double>(x_at(i));
    const double expect = (f == Formula::Tanh) ? host_gelu_tanh(x) : host_gelu_erf(x);
    const double gold = host_gelu_erf(x);
    const double abs_err = std::abs(got - expect);
    const double rel = abs_err / std::max(1e-12, std::abs(expect));
    const double tol = (f == Formula::Tanh) ? 2e-5 : 1e-5;
    if (rel > tol && abs_err > 1e-6) {
      std::fprintf(stderr, "VERIFY FAIL %s fwd i=%d: got %g expect %g rel=%.3e\n", tag, i, got,
                   expect, rel);
      std::exit(EXIT_FAILURE);
    }
    const double vs_gold = std::abs(static_cast<double>(got) - gold);
    max_abs_vs_erf = std::max(max_abs_vs_erf, vs_gold);
    max_rel_vs_erf = std::max(max_rel_vs_erf, vs_gold / std::max(1e-12, std::abs(gold)));
  }
  if (f == Formula::Tanh) {
    std::fprintf(stderr,
                 "INFO %s: probe tanh vs erf gold  max_abs=%.3e  max_rel=%.3e "
                 "(rel near zero can look large)\n",
                 tag, max_abs_vs_erf, max_rel_vs_erf);
  }
}

static void check_bwd(Buffers& b, Formula f, const char* tag) {
  const int probes[] = {0, 1, 128, b.n / 2, b.n - 2, b.n - 1};
  for (int k = 0; k < 6; ++k) {
    const int i = probes[k];
    if (i < 0 || i >= b.n) continue;
    float dx = 0.f;
    CUDA_CHECK(cudaMemcpy(&dx, b.dx + i, sizeof(float), cudaMemcpyDeviceToHost));
    const double x = static_cast<double>(x_at(i));
    const double expect = (f == Formula::Tanh) ? host_bwd_tanh(x) : host_bwd_erf(x);  // dy=1
    const double abs_err = std::abs(dx - expect);
    const double rel = abs_err / std::max(1e-12, std::abs(expect));
    if (rel > 2e-5 && abs_err > 1e-5) {
      std::fprintf(stderr, "VERIFY FAIL %s bwd i=%d: got %g expect %g rel=%.3e\n", tag, i, dx,
                   expect, rel);
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

static float time_fwd_scalar(Buffers& b, Formula f, int runs, int warmup) {
  const float ms = time_median(b, runs, warmup, [&] {
    if (f == Formula::Tanh) {
      kernel_fwd_scalar_tanh<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.n);
    } else {
      kernel_fwd_scalar_erf<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.n);
    }
  });
  check_fwd(b, f, f == Formula::Tanh ? "scalar_tanh" : "scalar_erf");
  return ms;
}

static float time_fwd_vec4(Buffers& b, Formula f, int runs, int warmup) {
  const float ms = time_median(b, runs, warmup, [&] {
    if (f == Formula::Tanh) {
      kernel_fwd_vec4_tanh<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.n);
    } else {
      kernel_fwd_vec4_erf<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.n);
    }
  });
  check_fwd(b, f, f == Formula::Tanh ? "vec4_tanh" : "vec4_erf");
  return ms;
}

static float time_bwd_scalar(Buffers& b, Formula f, int runs, int warmup) {
  const float ms = time_median(b, runs, warmup, [&] {
    if (f == Formula::Tanh) {
      kernel_bwd_scalar_tanh<<<b.grid, b.block, 0, b.stream>>>(b.x, b.dy, b.dx, b.n);
    } else {
      kernel_bwd_scalar_erf<<<b.grid, b.block, 0, b.stream>>>(b.x, b.dy, b.dx, b.n);
    }
  });
  check_bwd(b, f, f == Formula::Tanh ? "bwd_scalar_tanh" : "bwd_scalar_erf");
  return ms;
}

static float time_bwd_vec4(Buffers& b, Formula f, int runs, int warmup) {
  const float ms = time_median(b, runs, warmup, [&] {
    if (f == Formula::Tanh) {
      kernel_bwd_vec4_tanh<<<b.grid, b.block, 0, b.stream>>>(b.x, b.dy, b.dx, b.n);
    } else {
      kernel_bwd_vec4_erf<<<b.grid, b.block, 0, b.stream>>>(b.x, b.dy, b.dx, b.n);
    }
  });
  check_bwd(b, f, f == Formula::Tanh ? "bwd_vec4_tanh" : "bwd_vec4_erf");
  return ms;
}

static void warn_if_tiny(float ms) {
  if (ms < 0.05f) std::fprintf(stderr, "WARN: median %.4f ms < 0.05 ms; raise --n\n", ms);
}

struct FwdPair {
  float tanh_s = 0.f, tanh_v = 0.f, erf_s = 0.f, erf_v = 0.f;
};

static void print_sweep_row(int n, const FwdPair& r, bool csv_only) {
  const double sp_t = r.tanh_s / r.tanh_v;
  const double sp_e = r.erf_s / r.erf_v;
  const double te = r.tanh_v / r.erf_v;
  if (!csv_only) {
    std::printf(
        "n=%d tanh: scalar=%.4f vec4=%.4f ms (%.1f GB/s) s/v=%.3f | "
        "erf: scalar=%.4f vec4=%.4f ms (%.1f GB/s) s/v=%.3f | tanh_vec4/erf_vec4=%.3f\n",
        n, r.tanh_s, r.tanh_v, fwd_gbps(n, r.tanh_v), sp_t, r.erf_s, r.erf_v, fwd_gbps(n, r.erf_v),
        sp_e, te);
  }
  std::printf("SWEEP,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f\n", n, r.tanh_s, r.tanh_v,
              r.erf_s, r.erf_v, sp_t, sp_e, te, fwd_gbps(n, r.tanh_v), fwd_gbps(n, r.erf_v));
  warn_if_tiny(r.tanh_v);
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int sm = prop.multiProcessorCount;

  if (!cfg.csv_only) {
    std::printf("GPU: %s  sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor, sm);
    std::printf("mode=%s formula=%s n=%d block=%d runs=%d warmup=%d  fwd_bytes=2*n*4  "
                "bwd_bytes=3*n*4  (no cudnn Activation)\n",
                mode_name(cfg.mode), formula_name(cfg.formula), cfg.n, cfg.block, cfg.runs,
                cfg.warmup);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto make = [&](int n) {
    const int grid = choose_grid(sm, n, cfg.block, cfg.grid);
    Buffers b;
    alloc_buffers(b, n, grid, cfg.block, stream);
    return b;
  };

  auto run_both = [&](Buffers& b) {
    FwdPair r;
    r.tanh_s = time_fwd_scalar(b, Formula::Tanh, cfg.runs, cfg.warmup);
    r.tanh_v = time_fwd_vec4(b, Formula::Tanh, cfg.runs, cfg.warmup);
    r.erf_s = time_fwd_scalar(b, Formula::Erf, cfg.runs, cfg.warmup);
    r.erf_v = time_fwd_vec4(b, Formula::Erf, cfg.runs, cfg.warmup);
    return r;
  };

  if (cfg.mode == Mode::Sweep) {
    if (!cfg.csv_only) {
      std::printf(
          "CSV: n,tanh_scalar_ms,tanh_vec4_ms,erf_scalar_ms,erf_vec4_ms,"
          "tanh_s_over_v,erf_s_over_v,tanh_v_over_erf_v,tanh_vec4_GBps,erf_vec4_GBps\n");
    }
    std::printf(
        "SWEEP,n,tanh_scalar_ms,tanh_vec4_ms,erf_scalar_ms,erf_vec4_ms,"
        "tanh_s_over_v,erf_s_over_v,tanh_v_over_erf_v,tanh_vec4_GBps,erf_vec4_GBps\n");
    const int sizes[] = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 27, 1 << 28};
    for (int n : sizes) {
      Buffers b = make(n);
      print_sweep_row(n, run_both(b), cfg.csv_only);
      free_buffers(b);
    }
  } else if (cfg.mode == Mode::Modes) {
    Buffers b = make(cfg.n);
    if (!cfg.csv_only) std::printf("modes n=%d (both formulas + bwd default=%s)\n", cfg.n,
                                   formula_name(cfg.formula));
    print_sweep_row(cfg.n, run_both(b), cfg.csv_only);
    const float bs = time_bwd_scalar(b, cfg.formula, cfg.runs, cfg.warmup);
    const float bv = time_bwd_vec4(b, cfg.formula, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("bwd(%s) scalar=%.4f ms (%.1f GB/s) vec4=%.4f ms (%.1f GB/s) s/v=%.3f\n",
                  formula_name(cfg.formula), bs, bwd_gbps(cfg.n, bs), bv, bwd_gbps(cfg.n, bv),
                  bs / bv);
    }
    std::printf("BWD,%s,%d,%.4f,%.4f,%.4f,%.1f,%.1f\n", formula_name(cfg.formula), cfg.n, bs, bv,
                bs / bv, bwd_gbps(cfg.n, bs), bwd_gbps(cfg.n, bv));
    free_buffers(b);
  } else if (cfg.mode == Mode::Bwd) {
    Buffers b = make(cfg.n);
    const float bs = time_bwd_scalar(b, cfg.formula, cfg.runs, cfg.warmup);
    const float bv = time_bwd_vec4(b, cfg.formula, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("bwd(%s) n=%d scalar=%.4f vec4=%.4f ms  s/v=%.3f\n", formula_name(cfg.formula),
                  cfg.n, bs, bv, bs / bv);
    }
    std::printf("BWD,%s,%d,%.4f,%.4f,%.4f,%.1f,%.1f\n", formula_name(cfg.formula), cfg.n, bs, bv,
                bs / bv, bwd_gbps(cfg.n, bs), bwd_gbps(cfg.n, bv));
    warn_if_tiny(bv);
    free_buffers(b);
  } else if (cfg.mode == Mode::Scalar) {
    Buffers b = make(cfg.n);
    const float ms = time_fwd_scalar(b, cfg.formula, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("n=%d %s scalar=%.4f ms (%.1f GB/s)\n", cfg.n, formula_name(cfg.formula), ms,
                  fwd_gbps(cfg.n, ms));
    }
    std::printf("SINGLE,%s_scalar,%d,%.4f,%.1f\n", formula_name(cfg.formula), cfg.n, ms,
                fwd_gbps(cfg.n, ms));
    warn_if_tiny(ms);
    free_buffers(b);
  } else {
    Buffers b = make(cfg.n);
    const float ms = time_fwd_vec4(b, cfg.formula, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("n=%d %s vec4=%.4f ms (%.1f GB/s)\n", cfg.n, formula_name(cfg.formula), ms,
                  fwd_gbps(cfg.n, ms));
    }
    std::printf("SINGLE,%s_vec4,%d,%.4f,%.1f\n", formula_name(cfg.formula), cfg.n, ms,
                fwd_gbps(cfg.n, ms));
    warn_if_tiny(ms);
    free_buffers(b);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
