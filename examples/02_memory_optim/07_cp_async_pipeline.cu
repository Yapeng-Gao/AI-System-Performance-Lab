/**
 * [Module B] B-07. Async Copy / Pipeline：GMEM→SMEM 何时能藏延迟
 *
 * 模式：
 *   sync     : gmem→reg→smem，同步基线
 *   async1   : memcpy_async + 立刻 wait（换指令、无流水 overlap）
 *   pipe2    : 2-stage thread-local pipeline
 *   pipe4    : 4-stage thread-local pipeline
 *   pipe2_blk: 2-stage block-shared pipeline（对照 shared 开销）
 *   sweep    : 扫 fma-iters，输出 sync/pipe2/pipe4 加速比 CSV
 *
 * 需要：sm_80+（Ampere 及以上），CUDA 11.1+ / 推荐 12+
 */

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <algorithm>
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
  Sync = 0,
  Async1 = 1,
  Pipe2 = 2,
  Pipe4 = 3,
  Pipe2Blk = 4,
  Sweep = 5,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Sync: return "sync";
    case Mode::Async1: return "async1";
    case Mode::Pipe2: return "pipe2";
    case Mode::Pipe4: return "pipe4";
    case Mode::Pipe2Blk: return "pipe2_blk";
    case Mode::Sweep: return "sweep";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "sync") == 0) return Mode::Sync;
  if (std::strcmp(s, "async1") == 0) return Mode::Async1;
  if (std::strcmp(s, "pipe2") == 0) return Mode::Pipe2;
  if (std::strcmp(s, "pipe4") == 0) return Mode::Pipe4;
  if (std::strcmp(s, "pipe2_blk") == 0) return Mode::Pipe2Blk;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  std::fprintf(stderr,
               "Invalid --mode=%s "
               "(expected sync|async1|pipe2|pipe4|pipe2_blk|sweep)\n",
               s);
  std::exit(EXIT_FAILURE);
}

static float percentile_of(std::vector<float> v, float p) {
  if (v.empty()) return 0.0f;
  p = std::max(0.0f, std::min(100.0f, p));
  std::sort(v.begin(), v.end());
  const float pos = (p / 100.0f) * float(v.size() - 1);
  const size_t lo = (size_t)pos;
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const float t = pos - float(lo);
  return v[lo] * (1.0f - t) + v[hi] * t;
}

static float median_of(const std::vector<float>& v) { return percentile_of(v, 50.0f); }

static float mean_of(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  return std::accumulate(v.begin(), v.end(), 0.0f) / float(v.size());
}

__device__ __forceinline__ float burn_fma(float x, int fma_iters) {
  float acc = 0.f;
#pragma unroll 1
  for (int k = 0; k < fma_iters; ++k) {
    acc = fmaf(x, acc, 1.0f);
  }
  return acc;
}

// ---------------------------------------------------------------------------
// A) sync: gmem -> reg -> smem -> compute
// ---------------------------------------------------------------------------
__global__ void kernel_sync(const float* __restrict__ in, float* __restrict__ out,
                            int tiles_per_block, int fma_iters) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int tile_stride = gridDim.x * blockDim.x;
  const int base0 = blockIdx.x * blockDim.x + tid;
  float acc = 0.f;

  for (int t = 0; t < tiles_per_block; ++t) {
    const int gidx = base0 + t * tile_stride;
    // 每线程只读自己写入的 smem[tid]，不加 __syncthreads，
    // 避免把「block 同步税」算进 sync 基线，导致 pipeline 虚高。
    smem[tid] = in[gidx];
    const float x = smem[tid];
    acc += burn_fma(x, fma_iters);
  }
  out[base0] = acc;
}

// ---------------------------------------------------------------------------
// B) async1: memcpy_async + immediate wait (no multi-stage overlap)
// ---------------------------------------------------------------------------
__global__ void kernel_async1(const float* __restrict__ in, float* __restrict__ out,
                              int tiles_per_block, int fma_iters) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int tile_stride = gridDim.x * blockDim.x;
  const int base0 = blockIdx.x * blockDim.x + tid;
  float acc = 0.f;
  auto pipe = cuda::make_pipeline();

  for (int t = 0; t < tiles_per_block; ++t) {
    const int gidx = base0 + t * tile_stride;
    pipe.producer_acquire();
    cuda::memcpy_async(smem + tid, in + gidx, sizeof(float), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    const float x = smem[tid];
    acc += burn_fma(x, fma_iters);
    pipe.consumer_release();
  }
  out[base0] = acc;
}

// ---------------------------------------------------------------------------
// C/D) STAGES-stage thread-local software pipeline
// ---------------------------------------------------------------------------
template <int STAGES>
__global__ void kernel_pipe_tl(const float* __restrict__ in, float* __restrict__ out,
                               int tiles_per_block, int fma_iters) {
  extern __shared__ float smem[];  // STAGES * blockDim.x
  const int tid = threadIdx.x;
  const int tile_stride = gridDim.x * blockDim.x;
  const int base0 = blockIdx.x * blockDim.x + tid;
  float acc = 0.f;
  auto pipe = cuda::make_pipeline();

  auto issue = [&](int t) {
    pipe.producer_acquire();
    if (t < tiles_per_block) {
      const int gidx = base0 + t * tile_stride;
      const int slot = t % STAGES;
      cuda::memcpy_async(smem + slot * blockDim.x + tid, in + gidx, sizeof(float),
                         pipe);
    }
    pipe.producer_commit();
  };

#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    issue(s);
  }

  for (int t = 0; t < tiles_per_block; ++t) {
    pipe.consumer_wait();
    const float x = smem[(t % STAGES) * blockDim.x + tid];
    acc += burn_fma(x, fma_iters);
    pipe.consumer_release();
    if (t + STAGES < tiles_per_block) {
      issue(t + STAGES);
    }
  }
  out[base0] = acc;
}

// ---------------------------------------------------------------------------
// F) 2-stage block-shared pipeline (extra barrier overhead path)
// ---------------------------------------------------------------------------
__global__ void kernel_pipe2_blk(const float* __restrict__ in, float* __restrict__ out,
                                 int tiles_per_block, int fma_iters) {
  extern __shared__ float smem[];
  constexpr int STAGES = 2;
  // CUDA 13+: function-scope __shared__ 对象禁止动态初始化；抑制该诊断即可（与 CCCL 示例一致）
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pipe_state;
#pragma nv_diag_default static_var_with_dynamic_init

  const int tid = threadIdx.x;
  const int tile_stride = gridDim.x * blockDim.x;
  const int base0 = blockIdx.x * blockDim.x + tid;
  float acc = 0.f;

  auto block = cooperative_groups::this_thread_block();
  auto pipe = cuda::make_pipeline(block, &pipe_state);

  auto issue = [&](int t) {
    pipe.producer_acquire();
    if (t < tiles_per_block) {
      const int gidx = base0 + t * tile_stride;
      const int slot = t % STAGES;
      cuda::memcpy_async(block, smem + slot * blockDim.x + tid, in + gidx,
                         sizeof(float), pipe);
    }
    pipe.producer_commit();
  };

  for (int s = 0; s < STAGES; ++s) {
    issue(s);
  }

  for (int t = 0; t < tiles_per_block; ++t) {
    pipe.consumer_wait();
    const float x = smem[(t % STAGES) * blockDim.x + tid];
    acc += burn_fma(x, fma_iters);
    pipe.consumer_release();
    if (t + STAGES < tiles_per_block) {
      issue(t + STAGES);
    }
  }
  out[base0] = acc;
}

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <sync|async1|pipe2|pipe4|pipe2_blk|sweep> [options]\n"
      "  --n <elems>          total floats (default 1<<22)\n"
      "  --tiles <n>          tiles per block (default 64)\n"
      "  --block <threads>    block size (default 256)\n"
      "  --fma-iters <n>      FMAs per element after load (default 8)\n"
      "  --runs <n>           timed runs (default 7)\n"
      "  --warmup <n>         warmup runs (default 2)\n"
      "  --device <id>        GPU id (default 0)\n"
      "  --csv-only           only print CSV line(s)\n",
      prog);
}

struct BenchConfig {
  Mode mode = Mode::Pipe2;
  int n = 1 << 22;
  int tiles = 64;
  int block = 256;
  int fma_iters = 8;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

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
    } else if (std::strcmp(argv[i], "--tiles") == 0) {
      c.tiles = std::atoi(need("--tiles"));
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
    } else if (std::strcmp(argv[i], "--fma-iters") == 0) {
      c.fma_iters = std::atoi(need("--fma-iters"));
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
  if (c.n <= 0 || c.tiles <= 0 || c.block <= 0 || c.fma_iters < 0 || c.runs <= 0) {
    std::fprintf(stderr, "Invalid numeric args\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

struct LaunchPlan {
  int grid = 0;
  size_t smem_bytes = 0;
  int stages = 1;
};

static LaunchPlan make_plan(Mode mode, int n, int block, int tiles) {
  LaunchPlan p;
  p.grid = n / (block * tiles);
  if (p.grid < 1) {
    std::fprintf(stderr, "n too small for block*tiles; need n >= block*tiles\n");
    std::exit(EXIT_FAILURE);
  }
  if (n != p.grid * block * tiles) {
    std::fprintf(stderr, "n must be divisible by block*tiles (%d)\n", block * tiles);
    std::exit(EXIT_FAILURE);
  }
  switch (mode) {
    case Mode::Sync:
    case Mode::Async1:
      p.stages = 1;
      p.smem_bytes = size_t(block) * sizeof(float);
      break;
    case Mode::Pipe2:
    case Mode::Pipe2Blk:
    case Mode::Sweep:
      p.stages = 2;
      p.smem_bytes = size_t(2) * block * sizeof(float);
      break;
    case Mode::Pipe4:
      p.stages = 4;
      p.smem_bytes = size_t(4) * block * sizeof(float);
      break;
  }
  return p;
}

using KernelFn = void (*)(const float*, float*, int, int);

static KernelFn pick_kernel(Mode mode) {
  switch (mode) {
    case Mode::Sync: return kernel_sync;
    case Mode::Async1: return kernel_async1;
    case Mode::Pipe2: return kernel_pipe_tl<2>;
    case Mode::Pipe4: return kernel_pipe_tl<4>;
    case Mode::Pipe2Blk: return kernel_pipe2_blk;
    default: return nullptr;
  }
}

static float time_kernel_ms(KernelFn kn, float* d_in, float* d_out, int grid, int block,
                            size_t smem, int tiles, int fma_iters, int warmup, int runs,
                            std::vector<float>* samples) {
  for (int i = 0; i < warmup; ++i) {
    kn<<<grid, block, smem>>>(d_in, d_out, tiles, fma_iters);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  samples->clear();
  samples->reserve(runs);

  for (int i = 0; i < runs; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    kn<<<grid, block, smem>>>(d_in, d_out, tiles, fma_iters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples->push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return median_of(*samples);
}

static void run_one(const BenchConfig& c, Mode mode, float* d_in, float* d_out) {
  LaunchPlan plan = make_plan(mode, c.n, c.block, c.tiles);
  KernelFn kn = pick_kernel(mode);
  std::vector<float> samples;
  const float med =
      time_kernel_ms(kn, d_in, d_out, plan.grid, c.block, plan.smem_bytes, c.tiles,
                     c.fma_iters, c.warmup, c.runs, &samples);
  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean = mean_of(samples);
  const double bytes = double(c.n) * sizeof(float);
  const double gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (double(med) / 1000.0);

  if (c.csv_only) {
    std::printf("CSV,mode,%s,fma_iters,%d,n,%d,tiles,%d,median_ms,%.6f,gbps,%.3f\n",
                mode_name(mode), c.fma_iters, c.n, c.tiles, med, gbps);
    return;
  }

  std::printf(
      "mode=%-9s fma_iters=%-5d stages=%d  first=%.4f  median=%.4f  p95=%.4f  "
      "mean=%.4f ms  ~%.2f GB/s (payload read)\n",
      mode_name(mode), c.fma_iters, plan.stages, first, med, p95, mean, gbps);
}

static void run_sweep(const BenchConfig& c, float* d_in, float* d_out) {
  const int iters[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  if (!c.csv_only) {
    std::printf("\n=== intensity sweep (speedup = sync_median / pipe_median) ===\n");
    std::printf("fma_iters,sync_ms,pipe2_ms,pipe4_ms,speedup_pipe2,speedup_pipe4\n");
  }
  for (int fi : iters) {
    LaunchPlan p_sync = make_plan(Mode::Sync, c.n, c.block, c.tiles);
    LaunchPlan p2 = make_plan(Mode::Pipe2, c.n, c.block, c.tiles);
    LaunchPlan p4 = make_plan(Mode::Pipe4, c.n, c.block, c.tiles);
    std::vector<float> s0, s2, s4;
    const float m_sync =
        time_kernel_ms(kernel_sync, d_in, d_out, p_sync.grid, c.block, p_sync.smem_bytes,
                       c.tiles, fi, c.warmup, c.runs, &s0);
    const float m_p2 =
        time_kernel_ms(kernel_pipe_tl<2>, d_in, d_out, p2.grid, c.block, p2.smem_bytes,
                       c.tiles, fi, c.warmup, c.runs, &s2);
    const float m_p4 =
        time_kernel_ms(kernel_pipe_tl<4>, d_in, d_out, p4.grid, c.block, p4.smem_bytes,
                       c.tiles, fi, c.warmup, c.runs, &s4);
    const float sp2 = m_p2 > 0.f ? m_sync / m_p2 : 0.f;
    const float sp4 = m_p4 > 0.f ? m_sync / m_p4 : 0.f;
    std::printf("%d,%.6f,%.6f,%.6f,%.4f,%.4f\n", fi, m_sync, m_p2, m_p4, sp2, sp4);
  }
}

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  if (prop.major < 8) {
    std::fprintf(stderr,
                 "ERROR: need compute capability >= 8.0 for cp.async; got sm_%d%d\n",
                 prop.major, prop.minor);
    return EXIT_FAILURE;
  }

  if (!cfg.csv_only) {
    std::printf("GPU: %s | sm_%d%d | sharedMemPerBlock=%zu KB\n", prop.name, prop.major,
                prop.minor, prop.sharedMemPerBlock / 1024);
    std::printf("n=%d tiles/block=%d block=%d fma_iters=%d runs=%d warmup=%d\n", cfg.n,
                cfg.tiles, cfg.block, cfg.fma_iters, cfg.runs, cfg.warmup);
  }

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, size_t(cfg.n) * sizeof(float)));

  if (cfg.mode == Mode::Sweep) {
    run_sweep(cfg, d_in, d_out);
  } else {
    run_one(cfg, cfg.mode, d_in, d_out);
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
