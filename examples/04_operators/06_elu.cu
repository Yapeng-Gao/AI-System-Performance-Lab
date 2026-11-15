/**
 * [Module D] D-06. ELU：负半轴带 exp，是否掉出带宽（短章）
 *
 * 前向：y = x (x>=0); alpha*(exp(x)-1) (x<0)；alpha=1
 * 反传：dx = dy (y>0); dy*(y+alpha) (y<=0)
 *
 * x[i] = ((i & 255) / 255.f) * 8.f - 4.f   // [-4, 4]
 * n 是 256 的倍数。
 *
 *   scalar / vec4 / cudnn / sweep / modes / bwd
 *
 * 有效字节：前向 2*n*4；反传 3*n*4。官方对照：cuDNN Activation（handle/desc 在计时外）。
 * 网格 / block 口径同 D-03；不重讲向量化。
 */

#include <cudnn.h>
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

#define CUDNN_CHECK(call)                                                        \
  do {                                                                           \
    cudnnStatus_t st__ = (call);                                                 \
    if (st__ != CUDNN_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuDNN Error: %s (%d) at %s:%d\n",                   \
                   cudnnGetErrorString(st__), (int)st__, __FILE__, __LINE__);   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class Mode { Scalar = 0, Vec4 = 1, Cudnn = 2, Sweep = 3, Modes = 4, Bwd = 5 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Scalar: return "scalar";
    case Mode::Vec4: return "vec4";
    case Mode::Cudnn: return "cudnn";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    case Mode::Bwd: return "bwd";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "scalar") == 0) return Mode::Scalar;
  if (std::strcmp(s, "vec4") == 0) return Mode::Vec4;
  if (std::strcmp(s, "cudnn") == 0) return Mode::Cudnn;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  if (std::strcmp(s, "bwd") == 0) return Mode::Bwd;
  std::fprintf(stderr, "Invalid --mode=%s (expected scalar|vec4|cudnn|sweep|modes|bwd)\n", s);
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
      "Usage: %s --mode <scalar|vec4|cudnn|sweep|modes|bwd> [options]\n"
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

static __host__ __device__ __forceinline__ float x_at(int i) {
  return (static_cast<float>(i & 255) / 255.f) * 8.f - 4.f;
}

static __host__ __device__ __forceinline__ float act_f(float x) {
  const float kAlpha = 1.f;
  return (x >= 0.f) ? x : (kAlpha * (expf(x) - 1.f));
}

__global__ void kernel_fill_x(float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x_at(i);
}

__global__ void kernel_fill_ones(float* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.f;
}

__global__ void kernel_fwd_scalar(const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    y[i] = act_f(x[i]);
  }
}

__global__ void kernel_fwd_vec4(const float* x, float* y, int n) {
  const int n4 = n >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* y4 = reinterpret_cast<float4*>(y);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 a = x4[i];
    y4[i] = make_float4(act_f(a.x), act_f(a.y), act_f(a.z), act_f(a.w));
  }
}

__global__ void kernel_bwd_scalar(const float* y, const float* dy, float* dx, int n) {
  const float alpha = 1.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const float yi = y[i];
    dx[i] = dy[i] * (yi > 0.f ? 1.f : (yi + alpha));
  }
}

__global__ void kernel_bwd_vec4(const float* y, const float* dy, float* dx, int n) {
  const float alpha = 1.f;
  const int n4 = n >> 2;
  const float4* y4 = reinterpret_cast<const float4*>(y);
  const float4* dy4 = reinterpret_cast<const float4*>(dy);
  float4* dx4 = reinterpret_cast<float4*>(dx);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 a = y4[i];
    const float4 g = dy4[i];
    dx4[i] = make_float4(g.x * (a.x > 0.f ? 1.f : (a.x + alpha)),
                         g.y * (a.y > 0.f ? 1.f : (a.y + alpha)),
                         g.z * (a.z > 0.f ? 1.f : (a.z + alpha)),
                         g.w * (a.w > 0.f ? 1.f : (a.w + alpha)));
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

struct CudnnAct {
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t desc = nullptr;
  cudnnActivationDescriptor_t act = nullptr;
};

static void cudnn_init(CudnnAct& c, cudaStream_t stream) {
  CUDNN_CHECK(cudnnCreate(&c.handle));
  CUDNN_CHECK(cudnnSetStream(c.handle, stream));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&c.desc));
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&c.act));
  CUDNN_CHECK(cudnnSetActivationDescriptor(c.act, CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, 1.0));
}

static void cudnn_set_n(CudnnAct& c, int n) {
  // NCHW: treat length-n vector as N=n,C=1,H=1,W=1
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(c.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, 1, 1, 1));
}

static void cudnn_destroy(CudnnAct& c) {
  if (c.act) cudnnDestroyActivationDescriptor(c.act);
  if (c.desc) cudnnDestroyTensorDescriptor(c.desc);
  if (c.handle) cudnnDestroy(c.handle);
  c.act = nullptr;
  c.desc = nullptr;
  c.handle = nullptr;
}

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

static void check_fwd(Buffers& b, const char* tag) {
  const int probes[] = {0, 1, 128, b.n / 2, b.n - 2, b.n - 1};
  for (int k = 0; k < 6; ++k) {
    const int i = probes[k];
    if (i < 0 || i >= b.n) continue;
    float got = 0.f;
    CUDA_CHECK(cudaMemcpy(&got, b.y + i, sizeof(float), cudaMemcpyDeviceToHost));
    const double x = static_cast<double>(x_at(i));
    const double expect = (x >= 0.0) ? x : (1.0 * (std::exp(x) - 1.0));
    const double rel = std::abs(got - expect) / std::max(1e-12, std::abs(expect));
    if (rel > 1e-5) {
      std::fprintf(stderr, "VERIFY FAIL %s fwd i=%d: got %g expect %g rel=%.3e\n", tag, i, got,
                   expect, rel);
      std::exit(EXIT_FAILURE);
    }
  }
}

static void check_bwd(Buffers& b, const char* tag) {
  const double alpha = 1.0;
  const int probes[] = {0, 1, 128, b.n / 2, b.n - 2, b.n - 1};
  for (int k = 0; k < 6; ++k) {
    const int i = probes[k];
    if (i < 0 || i >= b.n) continue;
    float y = 0.f, dx = 0.f;
    CUDA_CHECK(cudaMemcpy(&y, b.y + i, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&dx, b.dx + i, sizeof(float), cudaMemcpyDeviceToHost));
    const double expect = (y > 0.f) ? 1.0 : (static_cast<double>(y) + alpha);  // dy=1
    const double rel = std::abs(dx - expect) / std::max(1e-12, std::abs(expect));
    if (rel > 1e-5 && std::abs(dx - expect) > 1e-6) {
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

static float time_fwd_scalar(Buffers& b, int runs, int warmup) {
  const float ms = time_median(b, runs, warmup, [&] {
    kernel_fwd_scalar<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.n);
  });
  check_fwd(b, "scalar");
  return ms;
}

static float time_fwd_vec4(Buffers& b, int runs, int warmup) {
  const float ms = time_median(b, runs, warmup, [&] {
    kernel_fwd_vec4<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.n);
  });
  check_fwd(b, "vec4");
  return ms;
}

static float time_fwd_cudnn(Buffers& b, CudnnAct& c, int runs, int warmup) {
  cudnn_set_n(c, b.n);
  const float alpha = 1.f, beta = 0.f;
  const float ms = time_median(b, runs, warmup, [&] {
    CUDNN_CHECK(cudnnActivationForward(c.handle, c.act, &alpha, c.desc, b.x, &beta, c.desc, b.y));
  });
  check_fwd(b, "cudnn");
  return ms;
}

static void prepare_y(Buffers& b) {
  kernel_fwd_vec4<<<b.grid, b.block, 0, b.stream>>>(b.x, b.y, b.n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(b.stream));
}

static float time_bwd_scalar(Buffers& b, int runs, int warmup) {
  prepare_y(b);
  const float ms = time_median(b, runs, warmup, [&] {
    kernel_bwd_scalar<<<b.grid, b.block, 0, b.stream>>>(b.y, b.dy, b.dx, b.n);
  });
  check_bwd(b, "bwd_scalar");
  return ms;
}

static float time_bwd_vec4(Buffers& b, int runs, int warmup) {
  prepare_y(b);
  const float ms = time_median(b, runs, warmup, [&] {
    kernel_bwd_vec4<<<b.grid, b.block, 0, b.stream>>>(b.y, b.dy, b.dx, b.n);
  });
  check_bwd(b, "bwd_vec4");
  return ms;
}

static float time_bwd_cudnn(Buffers& b, CudnnAct& c, int runs, int warmup) {
  prepare_y(b);
  cudnn_set_n(c, b.n);
  const float alpha = 1.f, beta = 0.f;
  const float ms = time_median(b, runs, warmup, [&] {
    CUDNN_CHECK(cudnnActivationBackward(c.handle, c.act, &alpha, c.desc, b.y, c.desc, b.dy, c.desc,
                                        b.x, &beta, c.desc, b.dx));
  });
  check_bwd(b, "bwd_cudnn");
  return ms;
}

static void warn_if_tiny(float ms) {
  if (ms < 0.05f) std::fprintf(stderr, "WARN: median %.4f ms < 0.05 ms; raise --n\n", ms);
}

struct FwdRow {
  float scalar = 0.f, vec4 = 0.f, cudnn = 0.f;
};

static void print_fwd(int n, const FwdRow& r, bool csv_only) {
  const double sp = r.scalar / r.vec4;
  const double vc = r.vec4 / r.cudnn;
  if (!csv_only) {
    std::printf(
        "n=%d scalar=%.4f ms (%.1f GB/s) vec4=%.4f ms (%.1f GB/s) cudnn=%.4f ms (%.1f GB/s) "
        "scalar/vec4=%.3f vec4/cudnn=%.3f\n",
        n, r.scalar, fwd_gbps(n, r.scalar), r.vec4, fwd_gbps(n, r.vec4), r.cudnn, fwd_gbps(n, r.cudnn),
        sp, vc);
  }
  std::printf("SWEEP,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f\n", n, r.scalar, r.vec4, r.cudnn, sp,
              vc, fwd_gbps(n, r.scalar), fwd_gbps(n, r.vec4), fwd_gbps(n, r.cudnn));
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
    std::printf("mode=%s n=%d block=%d runs=%d warmup=%d  fwd_bytes=2*n*4  bwd_bytes=3*n*4  "
                "cudnn=elu\n",
                mode_name(cfg.mode), cfg.n, cfg.block, cfg.runs, cfg.warmup);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CudnnAct cudnn{};
  cudnn_init(cudnn, stream);

  auto make = [&](int n) {
    const int grid = choose_grid(sm, n, cfg.block, cfg.grid);
    Buffers b;
    alloc_buffers(b, n, grid, cfg.block, stream);
    return b;
  };

  auto run_fwd = [&](Buffers& b) {
    FwdRow r;
    r.scalar = time_fwd_scalar(b, cfg.runs, cfg.warmup);
    r.vec4 = time_fwd_vec4(b, cfg.runs, cfg.warmup);
    r.cudnn = time_fwd_cudnn(b, cudnn, cfg.runs, cfg.warmup);
    return r;
  };

  if (cfg.mode == Mode::Sweep) {
    if (!cfg.csv_only) {
      std::printf(
          "CSV: n,scalar_ms,vec4_ms,cudnn_ms,scalar_over_vec4,vec4_over_cudnn,scalar_GBps,vec4_GBps,"
          "cudnn_GBps\n");
    }
    std::printf(
        "SWEEP,n,scalar_ms,vec4_ms,cudnn_ms,scalar_over_vec4,vec4_over_cudnn,scalar_GBps,vec4_GBps,"
        "cudnn_GBps\n");
    const int sizes[] = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 27, 1 << 28};
    for (int n : sizes) {
      Buffers b = make(n);
      print_fwd(n, run_fwd(b), cfg.csv_only);
      free_buffers(b);
    }
  } else if (cfg.mode == Mode::Modes) {
    Buffers b = make(cfg.n);
    if (!cfg.csv_only) std::printf("modes n=%d (fwd + bwd + cudnn)\n", cfg.n);
    print_fwd(cfg.n, run_fwd(b), cfg.csv_only);
    const float bs = time_bwd_scalar(b, cfg.runs, cfg.warmup);
    const float bv = time_bwd_vec4(b, cfg.runs, cfg.warmup);
    const float bc = time_bwd_cudnn(b, cudnn, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf(
          "bwd scalar=%.4f ms (%.1f GB/s) vec4=%.4f ms (%.1f GB/s) cudnn=%.4f ms (%.1f GB/s) "
          "vec4/cudnn=%.3f\n",
          bs, bwd_gbps(cfg.n, bs), bv, bwd_gbps(cfg.n, bv), bc, bwd_gbps(cfg.n, bc), bv / bc);
    }
    std::printf("BWD,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f\n", cfg.n, bs, bv, bc, bv / bc,
                bwd_gbps(cfg.n, bs), bwd_gbps(cfg.n, bv), bwd_gbps(cfg.n, bc));
    free_buffers(b);
  } else if (cfg.mode == Mode::Bwd) {
    Buffers b = make(cfg.n);
    const float bs = time_bwd_scalar(b, cfg.runs, cfg.warmup);
    const float bv = time_bwd_vec4(b, cfg.runs, cfg.warmup);
    const float bc = time_bwd_cudnn(b, cudnn, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("bwd n=%d scalar=%.4f vec4=%.4f cudnn=%.4f ms  vec4/cudnn=%.3f\n", cfg.n, bs, bv,
                  bc, bv / bc);
    }
    std::printf("BWD,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f\n", cfg.n, bs, bv, bc, bv / bc,
                bwd_gbps(cfg.n, bs), bwd_gbps(cfg.n, bv), bwd_gbps(cfg.n, bc));
    warn_if_tiny(bv);
    free_buffers(b);
  } else if (cfg.mode == Mode::Scalar) {
    Buffers b = make(cfg.n);
    const float ms = time_fwd_scalar(b, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("n=%d scalar=%.4f ms (%.1f GB/s)\n", cfg.n, ms, fwd_gbps(cfg.n, ms));
    }
    std::printf("SINGLE,scalar,%d,%.4f,%.1f\n", cfg.n, ms, fwd_gbps(cfg.n, ms));
    warn_if_tiny(ms);
    free_buffers(b);
  } else if (cfg.mode == Mode::Vec4) {
    Buffers b = make(cfg.n);
    const float ms = time_fwd_vec4(b, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("n=%d vec4=%.4f ms (%.1f GB/s)\n", cfg.n, ms, fwd_gbps(cfg.n, ms));
    }
    std::printf("SINGLE,vec4,%d,%.4f,%.1f\n", cfg.n, ms, fwd_gbps(cfg.n, ms));
    warn_if_tiny(ms);
    free_buffers(b);
  } else {
    Buffers b = make(cfg.n);
    const float ms = time_fwd_cudnn(b, cudnn, cfg.runs, cfg.warmup);
    if (!cfg.csv_only) {
      std::printf("n=%d cudnn=%.4f ms (%.1f GB/s)\n", cfg.n, ms, fwd_gbps(cfg.n, ms));
    }
    std::printf("SINGLE,cudnn,%d,%.4f,%.1f\n", cfg.n, ms, fwd_gbps(cfg.n, ms));
    warn_if_tiny(ms);
    free_buffers(b);
  }

  cudnn_destroy(cudnn);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
