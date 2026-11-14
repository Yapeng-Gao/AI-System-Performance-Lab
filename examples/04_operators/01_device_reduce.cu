/**
 * [Module D] D-01. Device reduction：float 大数组求和
 *
 * x[i] = (float)(i & 255)。n 是 256 的倍数时，host double 真值是 n * 255 / 2。
 * 相对误差须 <= 1e-3。float 的 ulp 在大 n 上盖过「少加一个元素」，
 * 没丢元素的证据是已归档的 int32 全 1 校验，不在这次 float 运行里重证。
 *
 *   scalar   : grid-stride float 加载，block 内 shfl+SMEM，每 block 一次 atomicAdd
 *   vec4     : 同样的归约，加载改成 float4（16 字节）
 *   twopass  : float4 加载，部分和写入 partials，第二个 kernel 再收成一个数
 *   cub      : cub::DeviceReduce::Sum（临时存储在计时外）
 *   sweep    : 扫 n
 *   modes    : 定点四条路径 + 校验
 *
 * 主证据：CUDA event median。不把 ncu 附着墙钟当结论。
 * 不做：每元素一次 atomicAdd（C-03）、单 block 的 warp 课（C-01）、fp8。
 */

#include <cub/device/device_reduce.cuh>
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

enum class Mode { Scalar = 0, Vec4 = 1, TwoPass = 2, Cub = 3, Sweep = 4, Modes = 5 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Scalar: return "scalar";
    case Mode::Vec4: return "vec4";
    case Mode::TwoPass: return "twopass";
    case Mode::Cub: return "cub";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "scalar") == 0) return Mode::Scalar;
  if (std::strcmp(s, "vec4") == 0) return Mode::Vec4;
  if (std::strcmp(s, "twopass") == 0) return Mode::TwoPass;
  if (std::strcmp(s, "cub") == 0) return Mode::Cub;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected scalar|vec4|twopass|cub|sweep|modes)\n",
               s);
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
      "Usage: %s --mode <scalar|vec4|twopass|cub|sweep|modes> [options]\n"
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
    std::fprintf(stderr, "ERROR: --block must be 256 (block_sum assumes 8 warps)\n");
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

// n % 256 == 0. Sum of one period 0..255 is 32640.
static double host_ref(int n) { return (double)n * 255.0 / 2.0; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

// All 256 threads must call this. Result is valid on thread 0 only.
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

__device__ __forceinline__ float acc_scalar(const float* in, int n) {
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    acc += in[i];
  }
  return acc;
}

__device__ __forceinline__ float acc_vec4(const float* in, int n) {
  const int n4 = n >> 2;
  const float4* in4 = reinterpret_cast<const float4*>(in);
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
    const float4 v = in4[i];
    acc += (v.x + v.y) + (v.z + v.w);
  }
  return acc;
}

__global__ void kernel_scalar(const float* in, float* out, int n) {
  const float sum = block_sum(acc_scalar(in, n));
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void kernel_vec4(const float* in, float* out, int n) {
  const float sum = block_sum(acc_vec4(in, n));
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void kernel_partial(const float* in, float* partials, int n) {
  const float sum = block_sum(acc_vec4(in, n));
  if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

__global__ void kernel_finalize(const float* partials, int npartials, float* out) {
  float acc = 0.f;
  for (int i = threadIdx.x; i < npartials; i += blockDim.x) acc += partials[i];
  const float sum = block_sum(acc);
  if (threadIdx.x == 0) *out = sum;
}

__global__ void kernel_fill(float* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = static_cast<float>(i & 255);
}

static int choose_grid(int sm, int n, int block, int requested) {
  if (requested > 0) return requested;
  const int cap = std::max(1, sm * 8);
  const int tiles = std::max(1, (n + block - 1) / block);
  return std::min(cap, tiles);
}

static double useful_gbps(int n, float ms) {
  if (ms <= 0.f) return 0.0;
  return (double)n * 4.0 / (double)ms * 1e-6;
}

struct Buffers {
  float* in = nullptr;
  float* out = nullptr;
  float* partials = nullptr;
  void* cub_temp = nullptr;
  size_t cub_bytes = 0;
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
  CUDA_CHECK(cudaMalloc(&b.in, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.out, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.partials, (size_t)grid * sizeof(float)));
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, b.cub_bytes, b.in, b.out, n, stream));
  CUDA_CHECK(cudaMalloc(&b.cub_temp, b.cub_bytes));
  const int fill_grid = (n + block - 1) / block;
  kernel_fill<<<fill_grid, block, 0, stream>>>(b.in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void free_buffers(Buffers& b) {
  cudaFree(b.in);
  cudaFree(b.out);
  cudaFree(b.partials);
  cudaFree(b.cub_temp);
  b.in = b.out = b.partials = nullptr;
  b.cub_temp = nullptr;
}

// Returns relative error. Fails the process when it exceeds 1e-3.
static double check_sum(Buffers& b, const char* tag) {
  float got = 0.f;
  CUDA_CHECK(cudaMemcpy(&got, b.out, sizeof(float), cudaMemcpyDeviceToHost));
  const double ref = host_ref(b.n);
  const double rel = std::abs((double)got - ref) / std::abs(ref);
  if (rel > 1e-3) {
    std::fprintf(stderr, "VERIFY FAIL %s: got %.9g ref %.9g rel %.3e (rtol 1e-3)\n", tag, got, ref,
                 rel);
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

static float time_scalar(Buffers& b, int runs, int warmup, double* rel) {
  const float ms = time_median(b, runs, warmup, [&] {
    CUDA_CHECK(cudaMemsetAsync(b.out, 0, sizeof(float), b.stream));
    kernel_scalar<<<b.grid, b.block, 0, b.stream>>>(b.in, b.out, b.n);
  });
  *rel = check_sum(b, "scalar");
  return ms;
}

static float time_vec4(Buffers& b, int runs, int warmup, double* rel) {
  const float ms = time_median(b, runs, warmup, [&] {
    CUDA_CHECK(cudaMemsetAsync(b.out, 0, sizeof(float), b.stream));
    kernel_vec4<<<b.grid, b.block, 0, b.stream>>>(b.in, b.out, b.n);
  });
  *rel = check_sum(b, "vec4");
  return ms;
}

static float time_twopass(Buffers& b, int runs, int warmup, double* rel) {
  const float ms = time_median(b, runs, warmup, [&] {
    kernel_partial<<<b.grid, b.block, 0, b.stream>>>(b.in, b.partials, b.n);
    kernel_finalize<<<1, b.block, 0, b.stream>>>(b.partials, b.grid, b.out);
  });
  *rel = check_sum(b, "twopass");
  return ms;
}

static float time_cub(Buffers& b, int runs, int warmup, double* rel) {
  const float ms = time_median(b, runs, warmup, [&] {
    CUDA_CHECK(cub::DeviceReduce::Sum(b.cub_temp, b.cub_bytes, b.in, b.out, b.n, b.stream));
  });
  *rel = check_sum(b, "cub");
  return ms;
}

static void warn_if_tiny(float ms) {
  if (ms < 0.05f) {
    std::fprintf(stderr, "WARN: median %.4f ms < 0.05 ms; raise --n\n", ms);
  }
}

struct Row {
  float scalar = 0.f, vec4 = 0.f, twopass = 0.f, cub = 0.f;
  double max_rel = 0.0;
};

static void print_row(int n, const Row& r, bool csv_only) {
  const double v_sp = r.scalar / r.vec4;
  const double t_sp = r.scalar / r.twopass;
  const double c_sp = r.scalar / r.cub;
  const double vc = r.vec4 / r.cub;
  if (!csv_only) {
    std::printf(
        "n=%d scalar=%.4f ms (%.1f GB/s) vec4=%.4f ms (%.1f GB/s) "
        "twopass=%.4f ms cub=%.4f ms  scalar/vec4=%.3f scalar/twopass=%.3f scalar/cub=%.3f "
        "vec4/cub=%.3f max_rel=%.3e\n",
        n, r.scalar, useful_gbps(n, r.scalar), r.vec4, useful_gbps(n, r.vec4), r.twopass, r.cub, v_sp,
        t_sp, c_sp, vc, r.max_rel);
  }
  std::printf("SWEEP,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e\n", n, r.scalar, r.vec4,
              r.twopass, r.cub, v_sp, t_sp, c_sp, vc, r.max_rel);
  warn_if_tiny(r.vec4);
}

static Row run_all(Buffers& b, int runs, int warmup) {
  Row row{};
  double rel = 0.0;
  row.scalar = time_scalar(b, runs, warmup, &rel);
  row.max_rel = rel;
  row.vec4 = time_vec4(b, runs, warmup, &rel);
  row.max_rel = std::max(row.max_rel, rel);
  row.twopass = time_twopass(b, runs, warmup, &rel);
  row.max_rel = std::max(row.max_rel, rel);
  row.cub = time_cub(b, runs, warmup, &rel);
  row.max_rel = std::max(row.max_rel, rel);
  return row;
}

static void run_single(Buffers& b, Mode m, int runs, int warmup, bool csv_only) {
  double rel = 0.0;
  float ms = 0.f;
  const char* tag = mode_name(m);
  switch (m) {
    case Mode::Scalar: ms = time_scalar(b, runs, warmup, &rel); break;
    case Mode::Vec4: ms = time_vec4(b, runs, warmup, &rel); break;
    case Mode::TwoPass: ms = time_twopass(b, runs, warmup, &rel); break;
    case Mode::Cub: ms = time_cub(b, runs, warmup, &rel); break;
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

  if (!cfg.csv_only) {
    std::printf("GPU: %s  sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor, sm);
    std::printf("mode=%s n=%d block=%d runs=%d warmup=%d dtype=f32 rtol=1e-3\n", mode_name(cfg.mode),
                cfg.n, cfg.block, cfg.runs, cfg.warmup);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto make_buffers = [&](int n) {
    const int grid = choose_grid(sm, n, cfg.block, cfg.grid);
    Buffers b;
    alloc_buffers(b, n, grid, cfg.block, stream);
    return b;
  };

  if (cfg.mode == Mode::Sweep) {
    if (!cfg.csv_only) {
      std::printf(
          "CSV: n,scalar_ms,vec4_ms,twopass_ms,cub_ms,scalar/vec4,scalar/twopass,scalar/cub,"
          "vec4/cub,max_rel\n");
    }
    std::printf(
        "SWEEP,n,scalar_ms,vec4_ms,twopass_ms,cub_ms,scalar_over_vec4,scalar_over_twopass,"
        "scalar_over_cub,vec4_over_cub,max_rel\n");
    const int sizes[] = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 27, 1 << 28};
    for (int n : sizes) {
      Buffers b = make_buffers(n);
      print_row(n, run_all(b, cfg.runs, cfg.warmup), cfg.csv_only);
      free_buffers(b);
    }
  } else if (cfg.mode == Mode::Modes) {
    Buffers b = make_buffers(cfg.n);
    if (!cfg.csv_only) {
      std::printf("modes n=%d\n", cfg.n);
    }
    print_row(cfg.n, run_all(b, cfg.runs, cfg.warmup), cfg.csv_only);
    free_buffers(b);
  } else {
    Buffers b = make_buffers(cfg.n);
    run_single(b, cfg.mode, cfg.runs, cfg.warmup, cfg.csv_only);
    free_buffers(b);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
