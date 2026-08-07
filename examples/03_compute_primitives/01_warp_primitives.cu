/**
 * [Module C] C-01. Warp Primitives：寄存器级通信、mask 正确性与规约加速比
 *
 * 模式：
 *   smem    : naive SMEM tree block-reduce（float）
 *   shfl    : warp __shfl_down_sync 树 + 每 warp 一份 SMEM 部分和（float）
 *   redux   : int 路径；warp 用 __reduce_add_sync（需 sm_80+），否则跳过
 *   ballot  : __ballot_sync + elect leader 聚合计数（定点正确性+时延）
 *   sweep   : 扫 nwarps∈{1,2,4,8,16,32}，主曲线 shfl/smem 加速比 CSV
 *   modes   : 定点全表 smem/shfl/redux/ballot
 *
 * 主证据：CUDA event median 时延与加速比（sweep 主结论）
 * 硬件：不限 sm_90+；redux 需 sm_80+
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

enum class Mode {
  Smem = 0,
  Shfl = 1,
  Redux = 2,
  Ballot = 3,
  Sweep = 4,
  Modes = 5,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Smem: return "smem";
    case Mode::Shfl: return "shfl";
    case Mode::Redux: return "redux";
    case Mode::Ballot: return "ballot";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "smem") == 0) return Mode::Smem;
  if (std::strcmp(s, "shfl") == 0) return Mode::Shfl;
  if (std::strcmp(s, "redux") == 0) return Mode::Redux;
  if (std::strcmp(s, "ballot") == 0) return Mode::Ballot;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected smem|shfl|redux|ballot|sweep|modes)\n",
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

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum_shfl(float val) {
  // Full-warp participation; mask from program logic (not activemask).
  constexpr unsigned kFullMask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(kFullMask, val, offset);
  }
  return val;
}

__device__ __forceinline__ int warp_reduce_sum_shfl_int(int val) {
  constexpr unsigned kFullMask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(kFullMask, val, offset);
  }
  return val;
}

__device__ __forceinline__ int warp_reduce_sum_redux_int(int val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(0xffffffffu, val);
#else
  return warp_reduce_sum_shfl_int(val);
#endif
}

// ---------------------------------------------------------------------------
// Float block reduce: SMEM tree vs shfl+SMEM
// Each thread grid-stride accumulates, then block-reduces to d_partial[blockIdx].
// ---------------------------------------------------------------------------
__global__ void kernel_reduce_smem(const float* __restrict__ in, float* __restrict__ partial,
                                   int n) {
  extern __shared__ float sdata[];
  const int tid = threadIdx.x;
  const int gsize = blockDim.x * gridDim.x;

  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gsize) {
    sum += in[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial[blockIdx.x] = sdata[0];
  }
}

__global__ void kernel_reduce_shfl(const float* __restrict__ in, float* __restrict__ partial,
                                   int n) {
  // One float per warp in this block (max 32 warps for 1024 threads).
  __shared__ float warp_sums[32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const int gsize = blockDim.x * gridDim.x;

  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gsize) {
    sum += in[i];
  }
  sum = warp_reduce_sum_shfl(sum);
  if (lane == 0) {
    warp_sums[wid] = sum;
  }
  __syncthreads();

  // First warp reduces per-warp sums (idle lanes contribute 0).
  float block_sum = 0.f;
  if (wid == 0) {
    block_sum = (lane < nwarps) ? warp_sums[lane] : 0.f;
    block_sum = warp_reduce_sum_shfl(block_sum);
    if (lane == 0) {
      partial[blockIdx.x] = block_sum;
    }
  }
}

// ---------------------------------------------------------------------------
// Int block reduce: shfl vs hardware redux (sm_80+)
// ---------------------------------------------------------------------------
__global__ void kernel_reduce_shfl_int(const int* __restrict__ in, int* __restrict__ partial,
                                       int n) {
  __shared__ int warp_sums[32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const int gsize = blockDim.x * gridDim.x;

  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gsize) {
    sum += in[i];
  }
  sum = warp_reduce_sum_shfl_int(sum);
  if (lane == 0) {
    warp_sums[wid] = sum;
  }
  __syncthreads();

  if (wid == 0) {
    int block_sum = (lane < nwarps) ? warp_sums[lane] : 0;
    block_sum = warp_reduce_sum_shfl_int(block_sum);
    if (lane == 0) {
      partial[blockIdx.x] = block_sum;
    }
  }
}

__global__ void kernel_reduce_redux_int(const int* __restrict__ in, int* __restrict__ partial,
                                        int n) {
  __shared__ int warp_sums[32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const int gsize = blockDim.x * gridDim.x;

  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gsize) {
    sum += in[i];
  }
  sum = warp_reduce_sum_redux_int(sum);
  if (lane == 0) {
    warp_sums[wid] = sum;
  }
  __syncthreads();

  if (wid == 0) {
    int block_sum = (lane < nwarps) ? warp_sums[lane] : 0;
    block_sum = warp_reduce_sum_redux_int(block_sum);
    if (lane == 0) {
      partial[blockIdx.x] = block_sum;
    }
  }
}

// ---------------------------------------------------------------------------
// Ballot + elect: count odd elements via warp ballot, one atomic per warp
// ---------------------------------------------------------------------------
__global__ void kernel_ballot_count_odds(const int* __restrict__ in,
                                         unsigned long long* __restrict__ out, int n) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int gsize = blockDim.x * gridDim.x;

  unsigned long long local = 0;
  for (int base = blockIdx.x * blockDim.x; base < n; base += gsize) {
    const int i = base + tid;
    const int pred = (i < n && (in[i] & 1)) ? 1 : 0;
    // Program-logic mask: all currently executing lanes in this convergent region.
    // For this grid-stride loop body we require full-warp participation when i may
    // be OOB — inactive predicate still participates in ballot with pred=0.
    const unsigned mask = 0xffffffffu;
    const unsigned bits = __ballot_sync(mask, pred);
    if (lane == 0) {
      local += static_cast<unsigned long long>(__popc(bits));
    }
  }
  if (lane == 0 && local != 0) {
    atomicAdd(out, local);
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 1 << 24;       // elements
  int nwarps = 8;        // warps per block (blockDim = nwarps * 32)
  int grid = 0;          // 0 = auto from SM count
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <smem|shfl|redux|ballot|sweep|modes> [options]\n"
      "  --n <elems>           element count (default 1<<24)\n"
      "  --nwarps <k>          warps/block for fixed modes, power-of-2 in 1..32 "
      "(default 8)\n"
      "  --grid <blocks>       grid size (default 0 = auto ~SMs*8)\n"
      "  --runs <n>            timed runs (default 7)\n"
      "  --warmup <n>          warmup runs (default 2)\n"
      "  --device <id>         GPU id (default 0)\n"
      "  --csv-only            only print CSV line(s)\n",
      prog);
}

static bool is_pow2_in_range(int v, int lo, int hi) {
  if (v < lo || v > hi) return false;
  return (v & (v - 1)) == 0;
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
    } else if (std::strcmp(argv[i], "--nwarps") == 0) {
      c.nwarps = std::atoi(need("--nwarps"));
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
  if (!is_pow2_in_range(c.nwarps, 1, 32)) {
    std::fprintf(stderr, "ERROR: --nwarps must be power-of-2 in [1,32]\n");
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

static int auto_grid(int sm_count) {
  return std::max(1, sm_count * 8);
}

struct FloatBuf {
  float* d_in = nullptr;
  float* d_partial = nullptr;
  int n = 0;
  int grid = 0;
};

struct IntBuf {
  int* d_in = nullptr;
  int* d_partial = nullptr;
  unsigned long long* d_count = nullptr;
  int n = 0;
  int grid = 0;
};

static void alloc_float(FloatBuf* b, int n, int grid) {
  b->n = n;
  b->grid = grid;
  CUDA_CHECK(cudaMalloc(&b->d_in, size_t(n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b->d_partial, size_t(grid) * sizeof(float)));
}

static void free_float(FloatBuf* b) {
  if (b->d_in) CUDA_CHECK(cudaFree(b->d_in));
  if (b->d_partial) CUDA_CHECK(cudaFree(b->d_partial));
}

static void alloc_int(IntBuf* b, int n, int grid) {
  b->n = n;
  b->grid = grid;
  CUDA_CHECK(cudaMalloc(&b->d_in, size_t(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b->d_partial, size_t(grid) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b->d_count, sizeof(unsigned long long)));
}

static void free_int(IntBuf* b) {
  if (b->d_in) CUDA_CHECK(cudaFree(b->d_in));
  if (b->d_partial) CUDA_CHECK(cudaFree(b->d_partial));
  if (b->d_count) CUDA_CHECK(cudaFree(b->d_count));
}

static void init_float_host_pattern(std::vector<float>* h, int n) {
  h->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    // Small ints-as-float keep exact sums for correctness.
    (*h)[static_cast<size_t>(i)] = float((i % 16) + 1);
  }
}

static void init_int_host_pattern(std::vector<int>* h, int n) {
  h->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*h)[static_cast<size_t>(i)] = (i % 16) + 1;
  }
}

static double host_sum_float(const std::vector<float>& h) {
  long double s = 0.0L;
  for (float v : h) s += static_cast<long double>(v);
  return static_cast<double>(s);
}

static long long host_sum_int(const std::vector<int>& h) {
  long long s = 0;
  for (int v : h) s += v;
  return s;
}

static long long host_count_odds(const std::vector<int>& h) {
  long long c = 0;
  for (int v : h) {
    if (v & 1) ++c;
  }
  return c;
}

static void verify_float_reduce(FloatBuf& buf, double expect, bool quiet) {
  std::vector<float> h_partial(static_cast<size_t>(buf.grid));
  CUDA_CHECK(cudaMemcpy(h_partial.data(), buf.d_partial,
                        size_t(buf.grid) * sizeof(float), cudaMemcpyDeviceToHost));
  long double got = 0.0L;
  for (float v : h_partial) got += static_cast<long double>(v);
  const double g = static_cast<double>(got);
  const double tol = 1.0e-3 * (1.0 + std::fabs(expect));
  if (std::fabs(g - expect) > tol) {
    std::fprintf(stderr,
                 "ERROR: float reduce mismatch: got=%.6f expect=%.6f (tol=%.6f)\n", g,
                 expect, tol);
    std::exit(EXIT_FAILURE);
  }
  if (!quiet) {
    std::printf("verify float-reduce OK (sum=%.0f)\n", expect);
  }
}

static void verify_int_reduce(IntBuf& buf, long long expect, bool quiet) {
  std::vector<int> h_partial(static_cast<size_t>(buf.grid));
  CUDA_CHECK(cudaMemcpy(h_partial.data(), buf.d_partial,
                        size_t(buf.grid) * sizeof(int), cudaMemcpyDeviceToHost));
  long long got = 0;
  for (int v : h_partial) got += v;
  if (got != expect) {
    std::fprintf(stderr, "ERROR: int reduce mismatch: got=%lld expect=%lld\n",
                 (long long)got, expect);
    std::exit(EXIT_FAILURE);
  }
  if (!quiet) {
    std::printf("verify int-reduce OK (sum=%lld)\n", expect);
  }
}

static void verify_ballot(IntBuf& buf, long long expect, bool quiet) {
  unsigned long long got = 0;
  CUDA_CHECK(
      cudaMemcpy(&got, buf.d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  if (static_cast<long long>(got) != expect) {
    std::fprintf(stderr, "ERROR: ballot count mismatch: got=%llu expect=%lld\n",
                 (unsigned long long)got, expect);
    std::exit(EXIT_FAILURE);
  }
  if (!quiet) {
    std::printf("verify ballot OK (odd_count=%lld)\n", expect);
  }
}

static float run_float_mode(Mode mode, FloatBuf& buf, int block, int warmup, int runs,
                            double expect_sum, bool verify, bool quiet,
                            std::vector<float>* samples) {
  const size_t smem_bytes =
      (mode == Mode::Smem) ? size_t(block) * sizeof(float) : size_t(0);

  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(buf.d_partial, 0, size_t(buf.grid) * sizeof(float)));
    if (mode == Mode::Smem) {
      kernel_reduce_smem<<<buf.grid, block, smem_bytes>>>(buf.d_in, buf.d_partial,
                                                          buf.n);
    } else {
      kernel_reduce_shfl<<<buf.grid, block>>>(buf.d_in, buf.d_partial, buf.n);
    }
    CUDA_CHECK(cudaGetLastError());
  };

  if (verify) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    verify_float_reduce(buf, expect_sum, quiet);
  }
  return time_launch_ms(launch, warmup, runs, samples);
}

static float run_int_reduce_mode(bool use_redux, IntBuf& buf, int block, int warmup,
                                 int runs, long long expect_sum, bool verify, bool quiet,
                                 std::vector<float>* samples) {
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(buf.d_partial, 0, size_t(buf.grid) * sizeof(int)));
    if (use_redux) {
      kernel_reduce_redux_int<<<buf.grid, block>>>(buf.d_in, buf.d_partial, buf.n);
    } else {
      kernel_reduce_shfl_int<<<buf.grid, block>>>(buf.d_in, buf.d_partial, buf.n);
    }
    CUDA_CHECK(cudaGetLastError());
  };

  if (verify) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    verify_int_reduce(buf, expect_sum, quiet);
  }
  return time_launch_ms(launch, warmup, runs, samples);
}

static float run_ballot(IntBuf& buf, int block, int warmup, int runs, long long expect,
                        bool verify, bool quiet, std::vector<float>* samples) {
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(buf.d_count, 0, sizeof(unsigned long long)));
    kernel_ballot_count_odds<<<buf.grid, block>>>(buf.d_in, buf.d_count, buf.n);
    CUDA_CHECK(cudaGetLastError());
  };

  if (verify) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    verify_ballot(buf, expect, quiet);
  }
  return time_launch_ms(launch, warmup, runs, samples);
}

static void print_float_row(const char* tag, int nwarps, int block, int grid, int n,
                            float med_ms, const std::vector<float>& samples, bool csv_only) {
  float p10 = percentile_of(samples, 10.0f);
  float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", tag, nwarps, block, grid, n, med_ms, p10,
                p90);
  } else {
    std::printf("%-6s nwarps=%2d block=%4d grid=%4d n=%d | median=%.4f ms "
                "(p10=%.4f p90=%.4f)\n",
                tag, nwarps, block, grid, n, med_ms, p10, p90);
  }
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int sm = prop.major * 10 + prop.minor;
  const int grid = (cfg.grid > 0) ? cfg.grid : auto_grid(prop.multiProcessorCount);
  const bool quiet = cfg.csv_only;

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("mode=%s n=%d nwarps=%d grid=%d runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.n, cfg.nwarps, grid, cfg.runs, cfg.warmup);
  }

  std::vector<float> h_f;
  init_float_host_pattern(&h_f, cfg.n);
  const double expect_f = host_sum_float(h_f);

  std::vector<int> h_i;
  init_int_host_pattern(&h_i, cfg.n);
  const long long expect_i = host_sum_int(h_i);
  const long long expect_odds = host_count_odds(h_i);

  FloatBuf fbuf{};
  alloc_float(&fbuf, cfg.n, grid);
  CUDA_CHECK(cudaMemcpy(fbuf.d_in, h_f.data(), size_t(cfg.n) * sizeof(float),
                        cudaMemcpyHostToDevice));

  IntBuf ibuf{};
  alloc_int(&ibuf, cfg.n, grid);
  CUDA_CHECK(cudaMemcpy(ibuf.d_in, h_i.data(), size_t(cfg.n) * sizeof(int),
                        cudaMemcpyHostToDevice));

  std::vector<float> samples;

  auto run_pair_at_nwarps = [&](int nwarps, bool verify) {
    const int block = nwarps * 32;
    float ms_smem =
        run_float_mode(Mode::Smem, fbuf, block, cfg.warmup, cfg.runs, expect_f, verify,
                       quiet, &samples);
    print_float_row("smem", nwarps, block, grid, cfg.n, ms_smem, samples, cfg.csv_only);

    float ms_shfl =
        run_float_mode(Mode::Shfl, fbuf, block, cfg.warmup, cfg.runs, expect_f, verify,
                       quiet, &samples);
    print_float_row("shfl", nwarps, block, grid, cfg.n, ms_shfl, samples, cfg.csv_only);

    const float speedup = (ms_shfl > 0.f) ? (ms_smem / ms_shfl) : 0.f;
    if (cfg.csv_only) {
      std::printf("speedup,%d,%d,%d,%d,%.6f,,\n", nwarps, block, grid, cfg.n, speedup);
    } else {
      std::printf("speedup shfl/smem @ nwarps=%d: %.3fx\n", nwarps, speedup);
    }
    return speedup;
  };

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) {
      std::printf("tag,nwarps,block,grid,n,median_ms,p10_ms,p90_ms\n");
    } else {
      std::printf("\n== sweep nwarps (main curve: shfl/smem) ==\n");
    }
    const int kNwarpsList[] = {1, 2, 4, 8, 16, 32};
    bool first = true;
    for (int nw : kNwarpsList) {
      run_pair_at_nwarps(nw, /*verify=*/first);
      first = false;
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) {
      std::printf("tag,nwarps,block,grid,n,median_ms,p10_ms,p90_ms\n");
    } else {
      std::printf("\n== modes (fixed-point table) ==\n");
    }
    const int block = cfg.nwarps * 32;
    run_pair_at_nwarps(cfg.nwarps, /*verify=*/true);

    // redux定点：int shfl vs redux
    if (sm < 80) {
      if (!quiet) {
        std::printf("redux SKIPPED (need sm_80+, got sm_%d%d)\n", prop.major,
                    prop.minor);
      } else {
        std::printf("redux_skip,%d,%d,%d,%d,0,,\n", cfg.nwarps, block, grid, cfg.n);
      }
    } else {
      float ms_shfl_i =
          run_int_reduce_mode(/*use_redux=*/false, ibuf, block, cfg.warmup, cfg.runs,
                              expect_i, /*verify=*/true, quiet, &samples);
      print_float_row("shfl_i", cfg.nwarps, block, grid, cfg.n, ms_shfl_i, samples,
                      cfg.csv_only);
      float ms_redux =
          run_int_reduce_mode(/*use_redux=*/true, ibuf, block, cfg.warmup, cfg.runs,
                              expect_i, /*verify=*/true, quiet, &samples);
      print_float_row("redux", cfg.nwarps, block, grid, cfg.n, ms_redux, samples,
                      cfg.csv_only);
      const float sp = (ms_redux > 0.f) ? (ms_shfl_i / ms_redux) : 0.f;
      if (cfg.csv_only) {
        std::printf("redux_speedup,%d,%d,%d,%d,%.6f,,\n", cfg.nwarps, block, grid, cfg.n,
                    sp);
      } else {
        std::printf("speedup redux/shfl_i @ nwarps=%d: %.3fx\n", cfg.nwarps, sp);
      }
    }

    float ms_ballot = run_ballot(ibuf, block, cfg.warmup, cfg.runs, expect_odds,
                                 /*verify=*/true, quiet, &samples);
    print_float_row("ballot", cfg.nwarps, block, grid, cfg.n, ms_ballot, samples,
                    cfg.csv_only);
  } else if (cfg.mode == Mode::Smem || cfg.mode == Mode::Shfl) {
    const int block = cfg.nwarps * 32;
    if (cfg.csv_only) {
      std::printf("tag,nwarps,block,grid,n,median_ms,p10_ms,p90_ms\n");
    }
    float ms = run_float_mode(cfg.mode, fbuf, block, cfg.warmup, cfg.runs, expect_f,
                              /*verify=*/true, quiet, &samples);
    print_float_row(mode_name(cfg.mode), cfg.nwarps, block, grid, cfg.n, ms, samples,
                    cfg.csv_only);
  } else if (cfg.mode == Mode::Redux) {
    const int block = cfg.nwarps * 32;
    if (sm < 80) {
      std::fprintf(stderr, "ERROR: --mode redux requires sm_80+ (got sm_%d%d)\n",
                   prop.major, prop.minor);
      free_float(&fbuf);
      free_int(&ibuf);
      return EXIT_FAILURE;
    }
    if (cfg.csv_only) {
      std::printf("tag,nwarps,block,grid,n,median_ms,p10_ms,p90_ms\n");
    }
    float ms_shfl_i =
        run_int_reduce_mode(false, ibuf, block, cfg.warmup, cfg.runs, expect_i, true,
                            quiet, &samples);
    print_float_row("shfl_i", cfg.nwarps, block, grid, cfg.n, ms_shfl_i, samples,
                    cfg.csv_only);
    float ms_redux =
        run_int_reduce_mode(true, ibuf, block, cfg.warmup, cfg.runs, expect_i, true,
                            quiet, &samples);
    print_float_row("redux", cfg.nwarps, block, grid, cfg.n, ms_redux, samples,
                    cfg.csv_only);
  } else if (cfg.mode == Mode::Ballot) {
    const int block = cfg.nwarps * 32;
    if (cfg.csv_only) {
      std::printf("tag,nwarps,block,grid,n,median_ms,p10_ms,p90_ms\n");
    }
    float ms = run_ballot(ibuf, block, cfg.warmup, cfg.runs, expect_odds, true, quiet,
                          &samples);
    print_float_row("ballot", cfg.nwarps, block, grid, cfg.n, ms, samples, cfg.csv_only);
  }

  free_float(&fbuf);
  free_int(&ibuf);
  return EXIT_SUCCESS;
}
