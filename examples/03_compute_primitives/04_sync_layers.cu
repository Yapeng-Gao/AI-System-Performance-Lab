/**
 * [Module C] C-04. 同步分层：warp / block / grid
 *
 * 主命题：三层同步相对代价 + cooperative launch 门槛 + 选型（非 launch 拆解）。
 *
 * 模式：
 *   warp         : __syncwarp 空循环 × iters
 *   block        : __syncthreads 空循环 × iters
 *   grid         : this_grid().sync 空循环 × iters（cudaLaunchCooperativeKernel）
 *   sweep        : 扫 nwarps∈{1,2,4,8,16,32}：warp vs block（主曲线）
 *   sweep_grid   : 扫 nblocks（coop occupancy 夹紧）：grid sync 时延形状
 *   correctness  : SMEM 交接 + __syncthreads 正确性（缺 sync 不冒充成功断言）
 *   phases       : 同载荷两阶段：单核 grid.sync vs 两 kernel（定点；不拆 launch）
 *   modes        : 定点全表（默认 nwarps=8, nblocks=coop 上限夹紧后的中档）
 *
 * 主证据：CUDA event median。硬件：不限 sm_90+；grid* 需 CooperativeLaunch。
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",             \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__);   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class Mode {
  Warp = 0,
  Block = 1,
  Grid = 2,
  Sweep = 3,
  SweepGrid = 4,
  Correctness = 5,
  Phases = 6,
  Modes = 7,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Warp: return "warp";
    case Mode::Block: return "block";
    case Mode::Grid: return "grid";
    case Mode::Sweep: return "sweep";
    case Mode::SweepGrid: return "sweep_grid";
    case Mode::Correctness: return "correctness";
    case Mode::Phases: return "phases";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "warp") == 0) return Mode::Warp;
  if (std::strcmp(s, "block") == 0) return Mode::Block;
  if (std::strcmp(s, "grid") == 0) return Mode::Grid;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "sweep_grid") == 0) return Mode::SweepGrid;
  if (std::strcmp(s, "correctness") == 0) return Mode::Correctness;
  if (std::strcmp(s, "phases") == 0) return Mode::Phases;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected "
               "warp|block|grid|sweep|sweep_grid|correctness|phases|modes)\n",
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
// Device: empty sync microbench (anti-DCE via clock64 accumulation)
// ---------------------------------------------------------------------------

__global__ void kernel_sync_warp(int iters, unsigned long long* __restrict__ out) {
  unsigned long long acc = clock64();
  for (int i = 0; i < iters; ++i) {
    __syncwarp();
    acc += static_cast<unsigned long long>(i) + static_cast<unsigned long long>(threadIdx.x);
  }
  if (threadIdx.x == 0) {
    atomicAdd(out, acc);
  }
}

__global__ void kernel_sync_block(int iters, unsigned long long* __restrict__ out) {
  unsigned long long acc = clock64();
  for (int i = 0; i < iters; ++i) {
    __syncthreads();
    acc += static_cast<unsigned long long>(i) + static_cast<unsigned long long>(threadIdx.x);
  }
  if (threadIdx.x == 0) {
    atomicAdd(out, acc);
  }
}

__global__ void kernel_sync_grid(int iters, unsigned long long* __restrict__ out) {
  cg::grid_group grid = cg::this_grid();
  unsigned long long acc = clock64();
  for (int i = 0; i < iters; ++i) {
    grid.sync();
    acc += static_cast<unsigned long long>(i) + static_cast<unsigned long long>(threadIdx.x);
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    atomicAdd(out, acc);
  }
}

// SMEM handoff: each thread writes slot, sync, reads neighbor.
__global__ void kernel_smem_handoff(int* __restrict__ out, int use_sync) {
  extern __shared__ int buf[];
  const int tid = threadIdx.x;
  buf[tid] = tid + 1;
  if (use_sync) {
    __syncthreads();
  }
  // Without sync: UB for cross-thread SMEM visibility — do not assert as stable fail.
  const int neighbor = (tid + 1) % blockDim.x;
  out[blockIdx.x * blockDim.x + tid] = buf[neighbor];
}

// Two-phase payload: write global[i]=i, sync, read global[(i+1)%n] into local sum.
__global__ void kernel_phases_grid(int* __restrict__ data, int n,
                                   unsigned long long* __restrict__ out) {
  cg::grid_group grid = cg::this_grid();
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    data[i] = i;
  }
  grid.sync();
  unsigned long long local = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    const int nxt = (i + 1) % n;
    local += static_cast<unsigned long long>(data[nxt]);
  }
  // Block reduce to one atomic (cheap; not the sync story).
  __shared__ unsigned long long block_sum;
  if (threadIdx.x == 0) block_sum = 0;
  __syncthreads();
  atomicAdd(&block_sum, local);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(out, block_sum);
}

__global__ void kernel_phases_write(int* __restrict__ data, int n) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    data[i] = i;
  }
}

__global__ void kernel_phases_read(const int* __restrict__ data, int n,
                                   unsigned long long* __restrict__ out) {
  const int gsize = blockDim.x * gridDim.x;
  unsigned long long local = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    const int nxt = (i + 1) % n;
    local += static_cast<unsigned long long>(data[nxt]);
  }
  __shared__ unsigned long long block_sum;
  if (threadIdx.x == 0) block_sum = 0;
  __syncthreads();
  atomicAdd(&block_sum, local);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(out, block_sum);
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

struct BenchConfig {
  Mode mode = Mode::Sweep;
  int iters = 256;
  int nwarps = 8;     // blockDim = nwarps * 32
  int nblocks = 0;    // 0 = auto (coop max or SMs)
  int n = 1 << 20;    // phases payload elems
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <warp|block|grid|sweep|sweep_grid|correctness|phases|modes> "
      "[options]\n"
      "  --iters <n>         empty-sync loop count (default 256)\n"
      "  --nwarps <n>        warps/block for fixed modes (default 8; blockDim=n*32)\n"
      "  --nblocks <n>       grid size (default 0 = auto)\n"
      "  --n <elems>         phases payload size (default 1<<20)\n"
      "  --runs <n>          timed runs (default 7)\n"
      "  --warmup <n>        warmup runs (default 2)\n"
      "  --device <id>       GPU id (default 0)\n"
      "  --csv-only          only print CSV line(s)\n",
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
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      c.iters = std::atoi(need("--iters"));
    } else if (std::strcmp(argv[i], "--nwarps") == 0) {
      c.nwarps = std::atoi(need("--nwarps"));
    } else if (std::strcmp(argv[i], "--nblocks") == 0) {
      c.nblocks = std::atoi(need("--nblocks"));
    } else if (std::strcmp(argv[i], "--n") == 0) {
      c.n = std::atoi(need("--n"));
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
  if (c.iters <= 0) {
    std::fprintf(stderr, "ERROR: --iters must be positive\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.nwarps < 1 || c.nwarps > 32) {
    std::fprintf(stderr, "ERROR: --nwarps must be in [1,32]\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.n <= 0) {
    std::fprintf(stderr, "ERROR: --n must be positive\n");
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

struct CoopInfo {
  bool supported = false;
  int max_blocks_per_sm = 0;
  int max_grid = 0;
};

static CoopInfo query_coop(void* kernel, int block_dim, int sm_count, int device) {
  CoopInfo info;
  int attr = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrCooperativeLaunch, device));
  info.supported = (attr != 0);
  if (!info.supported) return info;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info.max_blocks_per_sm,
                                                           kernel, block_dim, 0));
  info.max_grid = std::max(1, info.max_blocks_per_sm * sm_count);
  return info;
}

static void launch_coop(const void* kernel, int grid, int block, void** args) {
  CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(block), args, 0, 0));
}

static void print_row(const char* tag, int nwarps, int nblocks, int iters, float med_ms,
                      const std::vector<float>& samples, bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%.6f,%.6f,%.6f\n", tag, nwarps, nblocks, iters, med_ms, p10,
                p90);
  } else {
    std::printf("%-12s nwarps=%2d nblocks=%4d iters=%d | median=%.4f ms "
                "(p10=%.4f p90=%.4f)\n",
                tag, nwarps, nblocks, iters, med_ms, p10, p90);
  }
}

static const char* kCsvHeader = "tag,nwarps,nblocks,iters,median_ms,p10_ms,p90_ms";

static float run_empty_sync(Mode layer, int nwarps, int nblocks, int iters, int warmup,
                            int runs, unsigned long long* d_out, bool quiet,
                            std::vector<float>* samples) {
  const int block = nwarps * 32;
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    if (layer == Mode::Warp) {
      kernel_sync_warp<<<nblocks, block>>>(iters, d_out);
      CUDA_CHECK(cudaGetLastError());
    } else if (layer == Mode::Block) {
      kernel_sync_block<<<nblocks, block>>>(iters, d_out);
      CUDA_CHECK(cudaGetLastError());
    } else if (layer == Mode::Grid) {
      void* args[] = {&iters, &d_out};
      launch_coop((void*)kernel_sync_grid, nblocks, block, args);
    } else {
      std::fprintf(stderr, "ERROR: run_empty_sync bad layer\n");
      std::exit(EXIT_FAILURE);
    }
  };
  (void)quiet;
  return time_launch_ms(launch, warmup, runs, samples);
}

static void run_correctness(int nwarps, int nblocks, bool quiet) {
  const int block = nwarps * 32;
  const int n = nblocks * block;
  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, size_t(n) * sizeof(int)));

  // With sync: must match neighbor = ((tid+1)%block)+1
  kernel_smem_handoff<<<nblocks, block, size_t(block) * sizeof(int)>>>(d_out, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, size_t(n) * sizeof(int), cudaMemcpyDeviceToHost));
  for (int b = 0; b < nblocks; ++b) {
    for (int t = 0; t < block; ++t) {
      const int expect = ((t + 1) % block) + 1;
      const int got = h[static_cast<size_t>(b * block + t)];
      if (got != expect) {
        std::fprintf(stderr,
                     "ERROR: correctness with-sync mismatch block=%d tid=%d got=%d "
                     "expect=%d\n",
                     b, t, got, expect);
        std::exit(EXIT_FAILURE);
      }
    }
  }
  if (!quiet) {
    std::printf("verify correctness(with __syncthreads) OK (nblocks=%d block=%d)\n",
                nblocks, block);
    std::printf("note: without-sync SMEM handoff is UB — not asserted as stable fail; "
                "use compute-sanitizer --tool synccheck if needed.\n");
  }

  // Fire without-sync once so the binary exercises the path (no verify).
  kernel_smem_handoff<<<nblocks, block, size_t(block) * sizeof(int)>>>(d_out, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_out));
}

static unsigned long long host_phases_expect(int n) {
  // sum_{i=0}^{n-1} data[(i+1)%n] with data[j]=j → sum of 0..n-1
  return static_cast<unsigned long long>(n) * static_cast<unsigned long long>(n - 1) / 2ULL;
}

static float run_phases_grid(int nwarps, int nblocks, int n, int warmup, int runs,
                             int* d_data, unsigned long long* d_out, bool verify,
                             bool quiet, std::vector<float>* samples) {
  const int block = nwarps * 32;
  const unsigned long long expect = host_phases_expect(n);
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    void* args[] = {&d_data, &n, &d_out};
    launch_coop((void*)kernel_phases_grid, nblocks, block, args);
  };
  if (verify) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long got = 0;
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost));
    if (got != expect) {
      std::fprintf(stderr, "ERROR: phases_grid sum mismatch got=%llu expect=%llu\n",
                   (unsigned long long)got, (unsigned long long)expect);
      std::exit(EXIT_FAILURE);
    }
    if (!quiet) std::printf("verify phases_grid OK (sum=%llu)\n", (unsigned long long)expect);
  }
  return time_launch_ms(launch, warmup, runs, samples);
}

static float run_phases_two_kernel(int nwarps, int nblocks, int n, int warmup, int runs,
                                   int* d_data, unsigned long long* d_out, bool verify,
                                   bool quiet, std::vector<float>* samples) {
  const int block = nwarps * 32;
  const unsigned long long expect = host_phases_expect(n);
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    kernel_phases_write<<<nblocks, block>>>(d_data, n);
    kernel_phases_read<<<nblocks, block>>>(d_data, n, d_out);
    CUDA_CHECK(cudaGetLastError());
  };
  if (verify) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long got = 0;
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost));
    if (got != expect) {
      std::fprintf(stderr, "ERROR: phases_two_kernel sum mismatch got=%llu expect=%llu\n",
                   (unsigned long long)got, (unsigned long long)expect);
      std::exit(EXIT_FAILURE);
    }
    if (!quiet)
      std::printf("verify phases_two_kernel OK (sum=%llu)\n", (unsigned long long)expect);
  }
  return time_launch_ms(launch, warmup, runs, samples);
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const bool quiet = cfg.csv_only;
  const int block_dim = cfg.nwarps * 32;

  CoopInfo coop =
      query_coop((void*)kernel_sync_grid, block_dim, prop.multiProcessorCount, cfg.device);
  // Re-query occupancy for phases kernel too (usually similar); use min for safety.
  CoopInfo coop_phases =
      query_coop((void*)kernel_phases_grid, block_dim, prop.multiProcessorCount, cfg.device);
  const int coop_max =
      coop.supported ? std::min(coop.max_grid, coop_phases.max_grid) : 0;

  int nblocks = cfg.nblocks;
  if (nblocks <= 0) {
    if (cfg.mode == Mode::Grid || cfg.mode == Mode::SweepGrid || cfg.mode == Mode::Phases ||
        cfg.mode == Mode::Modes) {
      nblocks = coop.supported ? std::max(1, std::min(coop_max, prop.multiProcessorCount))
                               : prop.multiProcessorCount;
    } else {
      nblocks = prop.multiProcessorCount;
    }
  }
  if (coop.supported && nblocks > coop_max &&
      (cfg.mode == Mode::Grid || cfg.mode == Mode::SweepGrid || cfg.mode == Mode::Phases ||
       cfg.mode == Mode::Modes)) {
    if (!quiet) {
      std::printf("note: clamp nblocks %d -> coop_max %d\n", nblocks, coop_max);
    }
    nblocks = coop_max;
  }

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("CooperativeLaunch=%s | coop_max_grid≈%d (blockDim=%d)\n",
                coop.supported ? "yes" : "no", coop_max, block_dim);
    std::printf("mode=%s iters=%d nwarps=%d nblocks=%d n=%d runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.iters, cfg.nwarps, nblocks, cfg.n, cfg.runs,
                cfg.warmup);
  }

  unsigned long long* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
  std::vector<float> samples;

  auto need_coop_or_exit = [&](const char* what) {
    if (!coop.supported) {
      std::fprintf(stderr, "SKIP/ERROR: %s requires CooperativeLaunch (not supported)\n",
                   what);
      std::exit(EXIT_FAILURE);
    }
  };

  if (cfg.mode == Mode::Correctness) {
    run_correctness(cfg.nwarps, std::max(1, std::min(nblocks, 4)), quiet);
    CUDA_CHECK(cudaFree(d_out));
    return EXIT_SUCCESS;
  }

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) {
      std::printf("%s\n", kCsvHeader);
    } else {
      std::printf("\n== sweep nwarps (main curve: block/warp) ==\n");
    }
    const int kWarps[] = {1, 2, 4, 8, 16, 32};
    // Keep grid modest for empty sync; use SMs blocks (not coop).
    const int grid_for_sweep = prop.multiProcessorCount;
    for (int nw : kWarps) {
      const float ms_w =
          run_empty_sync(Mode::Warp, nw, grid_for_sweep, cfg.iters, cfg.warmup, cfg.runs,
                         d_out, quiet, &samples);
      print_row("warp", nw, grid_for_sweep, cfg.iters, ms_w, samples, cfg.csv_only);
      const float ms_b =
          run_empty_sync(Mode::Block, nw, grid_for_sweep, cfg.iters, cfg.warmup, cfg.runs,
                         d_out, quiet, &samples);
      print_row("block", nw, grid_for_sweep, cfg.iters, ms_b, samples, cfg.csv_only);
      const float ratio = (ms_w > 0.f) ? (ms_b / ms_w) : 0.f;
      if (cfg.csv_only) {
        std::printf("ratio_block_warp,%d,%d,%d,%.6f,,\n", nw, grid_for_sweep, cfg.iters,
                    ratio);
      } else {
        std::printf("  ratio block/warp @nwarps=%d: %.3fx\n", nw, ratio);
      }
    }
    CUDA_CHECK(cudaFree(d_out));
    return EXIT_SUCCESS;
  }

  if (cfg.mode == Mode::SweepGrid) {
    need_coop_or_exit("sweep_grid");
    if (cfg.csv_only) {
      std::printf("%s\n", kCsvHeader);
    } else {
      std::printf("\n== sweep_grid nblocks (main curve: grid sync vs scale) ==\n");
    }
    // Points: 1, SMs, 2*SMs, coop_max (unique, ascending, clamped)
    std::vector<int> points;
    auto add = [&](int v) {
      v = std::max(1, std::min(v, coop_max));
      if (std::find(points.begin(), points.end(), v) == points.end()) points.push_back(v);
    };
    add(1);
    add(prop.multiProcessorCount);
    add(prop.multiProcessorCount * 2);
    add(coop_max);
    std::sort(points.begin(), points.end());

    for (int nb : points) {
      const float ms =
          run_empty_sync(Mode::Grid, cfg.nwarps, nb, cfg.iters, cfg.warmup, cfg.runs, d_out,
                         quiet, &samples);
      print_row("grid", cfg.nwarps, nb, cfg.iters, ms, samples, cfg.csv_only);
    }
    CUDA_CHECK(cudaFree(d_out));
    return EXIT_SUCCESS;
  }

  if (cfg.mode == Mode::Phases) {
    need_coop_or_exit("phases");
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, size_t(cfg.n) * sizeof(int)));
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);

    const float ms_g =
        run_phases_grid(cfg.nwarps, nblocks, cfg.n, cfg.warmup, cfg.runs, d_data, d_out,
                        true, quiet, &samples);
    print_row("phases_grid", cfg.nwarps, nblocks, cfg.iters, ms_g, samples, cfg.csv_only);

    const float ms_2 =
        run_phases_two_kernel(cfg.nwarps, nblocks, cfg.n, cfg.warmup, cfg.runs, d_data,
                              d_out, true, quiet, &samples);
    print_row("phases_two_k", cfg.nwarps, nblocks, cfg.iters, ms_2, samples, cfg.csv_only);

    if (!cfg.csv_only) {
      std::printf("phases ratio grid/two_kernel=%.3fx  (≈1 or >1: grid not auto-faster; "
                  "launch decompose → C-06)\n",
                  (ms_2 > 0.f) ? (ms_g / ms_2) : 0.f);
    } else {
      std::printf("ratio_phases_grid_two,%d,%d,%d,%.6f,,\n", cfg.nwarps, nblocks, cfg.iters,
                  (ms_2 > 0.f) ? (ms_g / ms_2) : 0.f);
    }
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    return EXIT_SUCCESS;
  }

  if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else std::printf("\n== modes (fixed-point) ==\n");

    run_correctness(cfg.nwarps, 2, quiet);

    const int grid_local = prop.multiProcessorCount;
    const float ms_w =
        run_empty_sync(Mode::Warp, cfg.nwarps, grid_local, cfg.iters, cfg.warmup, cfg.runs,
                       d_out, quiet, &samples);
    print_row("warp", cfg.nwarps, grid_local, cfg.iters, ms_w, samples, cfg.csv_only);
    const float ms_b =
        run_empty_sync(Mode::Block, cfg.nwarps, grid_local, cfg.iters, cfg.warmup, cfg.runs,
                       d_out, quiet, &samples);
    print_row("block", cfg.nwarps, grid_local, cfg.iters, ms_b, samples, cfg.csv_only);

    if (coop.supported) {
      const int nb = std::max(1, std::min(nblocks, coop_max));
      const float ms_g =
          run_empty_sync(Mode::Grid, cfg.nwarps, nb, cfg.iters, cfg.warmup, cfg.runs, d_out,
                         quiet, &samples);
      print_row("grid", cfg.nwarps, nb, cfg.iters, ms_g, samples, cfg.csv_only);
      if (!cfg.csv_only) {
        std::printf("ratio block/warp=%.3fx  grid/block=%.3fx\n",
                    (ms_w > 0.f) ? (ms_b / ms_w) : 0.f,
                    (ms_b > 0.f) ? (ms_g / ms_b) : 0.f);
      }

      int* d_data = nullptr;
      CUDA_CHECK(cudaMalloc(&d_data, size_t(cfg.n) * sizeof(int)));
      const float ms_pg =
          run_phases_grid(cfg.nwarps, nb, cfg.n, cfg.warmup, cfg.runs, d_data, d_out, true,
                          quiet, &samples);
      print_row("phases_grid", cfg.nwarps, nb, cfg.iters, ms_pg, samples, cfg.csv_only);
      const float ms_p2 =
          run_phases_two_kernel(cfg.nwarps, nb, cfg.n, cfg.warmup, cfg.runs, d_data, d_out,
                                true, quiet, &samples);
      print_row("phases_two_k", cfg.nwarps, nb, cfg.iters, ms_p2, samples, cfg.csv_only);
      if (!cfg.csv_only) {
        std::printf("phases ratio grid/two_kernel=%.3fx\n",
                    (ms_p2 > 0.f) ? (ms_pg / ms_p2) : 0.f);
      }
      CUDA_CHECK(cudaFree(d_data));
    } else if (!quiet) {
      std::printf("SKIP grid/phases: CooperativeLaunch not supported\n");
      std::printf("ratio block/warp=%.3fx\n", (ms_w > 0.f) ? (ms_b / ms_w) : 0.f);
    }

    CUDA_CHECK(cudaFree(d_out));
    return EXIT_SUCCESS;
  }

  // Single layer: warp | block | grid
  if (cfg.mode == Mode::Grid) need_coop_or_exit("grid");
  const int grid =
      (cfg.mode == Mode::Grid) ? nblocks : prop.multiProcessorCount;
  const float ms =
      run_empty_sync(cfg.mode, cfg.nwarps, grid, cfg.iters, cfg.warmup, cfg.runs, d_out,
                     quiet, &samples);
  if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
  print_row(mode_name(cfg.mode), cfg.nwarps, grid, cfg.iters, ms, samples, cfg.csv_only);

  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
