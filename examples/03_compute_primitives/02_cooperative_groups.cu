/**
 * [Module C] C-02. Cooperative Groups：安全分组、tile 集体与 coalesced 聚合
 *
 * 模式：
 *   intrinsic : C-01 风格手写 __shfl_down_sync warp reduce（基线；tile=32）
 *   tile32    : thread_block_tile<32> + g.shfl_down 手写树（CG API 同构对照）
 *   cg_reduce : thread_block_tile<32> + cg::reduce（tile=32）
 *   coalesced : cg::coalesced_threads() 聚合计数（atomicAggInc 形态；定点正确性）
 *   sweep     : 扫 tile∈{8,16,32,64,128}，cg::reduce 时延——看 >32 悬崖（主曲线）
 *   cluster   : sm_90+ DSMEM 邻块读写 + cluster.sync（可选支线；不进主结论）
 *   modes     : 定点全表 intrinsic/tile32/cg_reduce/coalesced(+cluster)
 *
 * 主证据：CUDA event median 时延（sweep 主结论）。
 * 说明：为把「分组原语代价」从 GMEM 墙里剥出来，reduce 阶段带 --reps 放大因子
 *       （只放大规约本身，一次 GMEM 载入；正确性 verify 走 reps=1 精确比对）。
 * 硬件：主路径不限 sm_90+；cluster 需 sm_90+（否则清晰跳过）。
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// Reduce strategy (compile-time).
enum { STRAT_INTRINSIC = 0, STRAT_TILE_SHFL = 1, STRAT_CG_REDUCE = 2 };

enum class Mode {
  Intrinsic = 0,
  Tile32 = 1,
  CgReduce = 2,
  Coalesced = 3,
  Sweep = 4,
  Cluster = 5,
  Modes = 6,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Intrinsic: return "intrinsic";
    case Mode::Tile32: return "tile32";
    case Mode::CgReduce: return "cg_reduce";
    case Mode::Coalesced: return "coalesced";
    case Mode::Sweep: return "sweep";
    case Mode::Cluster: return "cluster";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "intrinsic") == 0) return Mode::Intrinsic;
  if (std::strcmp(s, "tile32") == 0) return Mode::Tile32;
  if (std::strcmp(s, "cg_reduce") == 0) return Mode::CgReduce;
  if (std::strcmp(s, "coalesced") == 0) return Mode::Coalesced;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "cluster") == 0) return Mode::Cluster;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected "
               "intrinsic|tile32|cg_reduce|coalesced|sweep|cluster|modes)\n",
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
// Device: tile reduce helpers
// ---------------------------------------------------------------------------
// Manual shfl tree over a tile (TILE<=32, single warp). Broadcast full sum to
// all ranks so the reps-perturbation creates a real dependency (no DCE).
template <int TILE>
__device__ __forceinline__ float tile_reduce_shfl(const cg::thread_block_tile<TILE>& g,
                                                   float val) {
#pragma unroll
  for (int offset = TILE / 2; offset > 0; offset >>= 1) {
    val += g.shfl_down(val, offset);
  }
  return g.shfl(val, 0);  // broadcast rank0 result to all
}

__device__ __forceinline__ float warp_reduce_intrinsic(float val) {
  constexpr unsigned kFullMask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(kFullMask, val, offset);
  }
  return __shfl_sync(kFullMask, val, 0);  // broadcast full sum to all lanes
}

// One kernel covers intrinsic / tile-shfl / cg::reduce via compile-time STRAT.
// Each thread grid-stride loads once, then reps rounds of tile-reduce isolate
// the primitive cost. Tile leaders atomicAdd their tile sum into partial[block].
template <int TILE, int STRAT>
__global__ void kernel_tile_reduce(const float* __restrict__ in,
                                   float* __restrict__ partial, int n, int reps) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TILE> tile = cg::tiled_partition<TILE>(block);

  const int gsize = blockDim.x * gridDim.x;
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    acc += in[i];
  }

  float val = acc;
  float red = 0.f;
  for (int r = 0; r < reps; ++r) {
    if constexpr (STRAT == STRAT_INTRINSIC) {
      red = warp_reduce_intrinsic(val);
    } else if constexpr (STRAT == STRAT_TILE_SHFL) {
      red = tile_reduce_shfl<TILE>(tile, val);
    } else {
      red = cg::reduce(tile, val, cg::plus<float>());
    }
    val += red * 1.0e-30f;  // dependency across rounds; negligible drift
  }

  if (tile.thread_rank() == 0) {
    atomicAdd(&partial[blockIdx.x], red);
  }
}

// ---------------------------------------------------------------------------
// Device: coalesced aggregation (atomicAggInc form)
// ---------------------------------------------------------------------------
// Count odd elements. Threads whose element is odd form a coalesced group;
// the group leader performs one atomicAdd of the group size. This is the safe,
// architecture-portable warp-aggregated atomic pattern from the CG blog.
__global__ void kernel_coalesced_count_odds(const int* __restrict__ in,
                                            unsigned long long* __restrict__ out, int n) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    if (in[i] & 1) {
      cg::coalesced_group active = cg::coalesced_threads();
      if (active.thread_rank() == 0) {
        atomicAdd(out, static_cast<unsigned long long>(active.size()));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Device: cluster DSMEM neighbor read (sm_90+)
// ---------------------------------------------------------------------------
// Each block writes a known value into its shared memory, then reads the next
// block's shared memory in the cluster via map_shared_rank. out[block] receives
// the peer's value. Host verifies against the expected cluster-local mapping.
__global__ void kernel_cluster_dsmem(int* __restrict__ out, int clusize) {
#if __CUDA_ARCH__ >= 900
  __shared__ int smem[1];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned brank = cluster.block_rank();

  smem[0] = static_cast<int>(brank) + 1;  // known value per cluster-local rank
  cluster.sync();

  const unsigned nblocks = cluster.num_blocks();
  const unsigned peer = (brank + 1u) % nblocks;
  int* peer_smem = cluster.map_shared_rank(smem, peer);
  const int got = peer_smem[0];
  cluster.sync();

  if (threadIdx.x == 0) {
    out[blockIdx.x] = got;
  }
#else
  (void)out;
  (void)clusize;
#endif
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 1 << 24;
  int block = 128;   // threads/block for reduce modes (must be >= max tile)
  int tile = 32;     // fixed tile size for cg_reduce standalone
  int reps = 50;     // reduce-amplification factor
  int grid = 0;      // 0 = auto ~SMs*8
  int clusize = 2;   // cluster dimension (blocks) for cluster mode
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode "
      "<intrinsic|tile32|cg_reduce|coalesced|sweep|cluster|modes> [options]\n"
      "  --n <elems>       element count (default 1<<24)\n"
      "  --block <thr>     threads/block, multiple of 128 recommended (default 128)\n"
      "  --tile <N>        tile size for cg_reduce standalone (default 32)\n"
      "  --reps <k>        reduce-amplification rounds (default 50)\n"
      "  --grid <blocks>   grid size (default 0 = auto ~SMs*8)\n"
      "  --clusize <c>     cluster blocks for cluster mode (default 2)\n"
      "  --runs <n>        timed runs (default 7)\n"
      "  --warmup <n>      warmup runs (default 2)\n"
      "  --device <id>     GPU id (default 0)\n"
      "  --csv-only        only print CSV line(s)\n",
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
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
    } else if (std::strcmp(argv[i], "--tile") == 0) {
      c.tile = std::atoi(need("--tile"));
    } else if (std::strcmp(argv[i], "--reps") == 0) {
      c.reps = std::atoi(need("--reps"));
    } else if (std::strcmp(argv[i], "--grid") == 0) {
      c.grid = std::atoi(need("--grid"));
    } else if (std::strcmp(argv[i], "--clusize") == 0) {
      c.clusize = std::atoi(need("--clusize"));
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
  if (c.block < 128 || c.block > 1024 || (c.block % 32) != 0) {
    std::fprintf(stderr, "ERROR: --block must be in [128,1024], multiple of 32\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.reps <= 0) {
    std::fprintf(stderr, "ERROR: --reps must be positive\n");
    std::exit(EXIT_FAILURE);
  }
  if (!is_pow2_in_range(c.tile, 8, 128)) {
    std::fprintf(stderr, "ERROR: --tile must be power-of-2 in [8,128]\n");
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

static int auto_grid(int sm_count) { return std::max(1, sm_count * 8); }

// Dispatch the templated reduce kernel by runtime tile/strategy.
static void launch_reduce(int strat, int tile, int grid, int block, const float* d_in,
                          float* d_partial, int n, int reps) {
#define ASPL_LAUNCH(T, S) \
  kernel_tile_reduce<T, S><<<grid, block>>>(d_in, d_partial, n, reps)
  if (strat == STRAT_INTRINSIC) {
    ASPL_LAUNCH(32, STRAT_INTRINSIC);
  } else if (strat == STRAT_TILE_SHFL) {
    ASPL_LAUNCH(32, STRAT_TILE_SHFL);
  } else {  // STRAT_CG_REDUCE
    switch (tile) {
      case 8: ASPL_LAUNCH(8, STRAT_CG_REDUCE); break;
      case 16: ASPL_LAUNCH(16, STRAT_CG_REDUCE); break;
      case 32: ASPL_LAUNCH(32, STRAT_CG_REDUCE); break;
      case 64: ASPL_LAUNCH(64, STRAT_CG_REDUCE); break;
      case 128: ASPL_LAUNCH(128, STRAT_CG_REDUCE); break;
      default:
        std::fprintf(stderr, "ERROR: unsupported tile=%d\n", tile);
        std::exit(EXIT_FAILURE);
    }
  }
#undef ASPL_LAUNCH
  CUDA_CHECK(cudaGetLastError());
}

static void init_float_host_pattern(std::vector<float>* h, int n) {
  h->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*h)[static_cast<size_t>(i)] = float((i % 16) + 1);
  }
}

static double host_sum_float(const std::vector<float>& h) {
  long double s = 0.0L;
  for (float v : h) s += static_cast<long double>(v);
  return static_cast<double>(s);
}

static long long host_count_odds(const std::vector<int>& h) {
  long long c = 0;
  for (int v : h) {
    if (v & 1) ++c;
  }
  return c;
}

static void verify_float(float* d_partial, int grid, double expect, bool quiet,
                         const char* tag) {
  std::vector<float> h_partial(static_cast<size_t>(grid));
  CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, size_t(grid) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  long double got = 0.0L;
  for (float v : h_partial) got += static_cast<long double>(v);
  const double g = static_cast<double>(got);
  const double tol = 1.0e-3 * (1.0 + std::fabs(expect));
  if (std::fabs(g - expect) > tol) {
    std::fprintf(stderr, "ERROR: %s reduce mismatch: got=%.6f expect=%.6f (tol=%.6f)\n",
                 tag, g, expect, tol);
    std::exit(EXIT_FAILURE);
  }
  if (!quiet) {
    std::printf("verify %s OK (sum=%.0f)\n", tag, expect);
  }
}

// Run one reduce config: verify (reps=1, exact) then time (cfg.reps).
static float run_reduce(int strat, int tile, int grid, int block, float* d_in,
                        float* d_partial, int n, int reps, int warmup, int runs,
                        double expect, bool verify, bool quiet, const char* tag,
                        std::vector<float>* samples) {
  if (verify) {
    CUDA_CHECK(cudaMemset(d_partial, 0, size_t(grid) * sizeof(float)));
    launch_reduce(strat, tile, grid, block, d_in, d_partial, n, /*reps=*/1);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify_float(d_partial, grid, expect, quiet, tag);
  }
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(d_partial, 0, size_t(grid) * sizeof(float)));
    launch_reduce(strat, tile, grid, block, d_in, d_partial, n, reps);
  };
  return time_launch_ms(launch, warmup, runs, samples);
}

static void print_row(const char* tag, int tile, int block, int grid, int reps, int n,
                      float med_ms, const std::vector<float>& samples, bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", tag, tile, block, grid, reps, n,
                med_ms, p10, p90);
  } else {
    std::printf("%-9s tile=%3d block=%4d grid=%4d reps=%d | median=%.4f ms "
                "(p10=%.4f p90=%.4f)\n",
                tag, tile, block, grid, reps, med_ms, p10, p90);
  }
}

static const char* kCsvHeader = "tag,tile,block,grid,reps,n,median_ms,p10_ms,p90_ms";

// coalesced aggregation: verify + time
static float run_coalesced(int grid, int block, int* d_in, unsigned long long* d_count,
                           int n, long long expect, int warmup, int runs, bool quiet,
                           std::vector<float>* samples) {
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    kernel_coalesced_count_odds<<<grid, block>>>(d_in, d_count, n);
    CUDA_CHECK(cudaGetLastError());
  };
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long got = 0;
  CUDA_CHECK(cudaMemcpy(&got, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  if (static_cast<long long>(got) != expect) {
    std::fprintf(stderr, "ERROR: coalesced count mismatch: got=%llu expect=%lld\n", got,
                 expect);
    std::exit(EXIT_FAILURE);
  }
  if (!quiet) {
    std::printf("verify coalesced OK (odd_count=%lld)\n", expect);
  }
  return time_launch_ms(launch, warmup, runs, samples);
}

// cluster DSMEM: verify neighbor mapping. Returns median ms; sets *ok=false if
// hardware/launch unsupported.
static float run_cluster(int grid, int block, int clusize, int warmup, int runs,
                         bool quiet, std::vector<float>* samples, bool* ok) {
  *ok = false;
  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, size_t(grid) * sizeof(int)));

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid, 1, 1);
  config.blockDim = dim3(block, 1, 1);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = clusize;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  config.attrs = attr;
  config.numAttrs = 1;

  cudaError_t err =
      cudaLaunchKernelEx(&config, kernel_cluster_dsmem, d_out, clusize);
  if (err != cudaSuccess) {
    if (!quiet) {
      std::printf("cluster SKIPPED (launch failed: %s)\n", cudaGetErrorString(err));
    }
    (void)cudaGetLastError();
    CUDA_CHECK(cudaFree(d_out));
    return 0.f;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h_out(static_cast<size_t>(grid));
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_t(grid) * sizeof(int),
                        cudaMemcpyDeviceToHost));
  for (int b = 0; b < grid; ++b) {
    const int local = b % clusize;
    const int peer = (local + 1) % clusize;
    const int expect = peer + 1;
    if (h_out[static_cast<size_t>(b)] != expect) {
      std::fprintf(stderr, "ERROR: cluster dsmem mismatch @block=%d: got=%d expect=%d\n",
                   b, h_out[static_cast<size_t>(b)], expect);
      CUDA_CHECK(cudaFree(d_out));
      std::exit(EXIT_FAILURE);
    }
  }
  if (!quiet) {
    std::printf("verify cluster OK (clusize=%d, DSMEM neighbor read)\n", clusize);
  }

  auto launch = [&]() {
    cudaError_t e = cudaLaunchKernelEx(&config, kernel_cluster_dsmem, d_out, clusize);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "cluster launch error: %s\n", cudaGetErrorString(e));
      std::exit(EXIT_FAILURE);
    }
  };
  const float ms = time_launch_ms(launch, warmup, runs, samples);
  CUDA_CHECK(cudaFree(d_out));
  *ok = true;
  return ms;
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int sm = prop.major * 10 + prop.minor;
  int grid = (cfg.grid > 0) ? cfg.grid : auto_grid(prop.multiProcessorCount);
  const bool quiet = cfg.csv_only;

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("mode=%s n=%d block=%d tile=%d reps=%d grid=%d runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.n, cfg.block, cfg.tile, cfg.reps, grid, cfg.runs,
                cfg.warmup);
  }

  // Shared float input for reduce modes.
  std::vector<float> h_f;
  init_float_host_pattern(&h_f, cfg.n);
  const double expect_f = host_sum_float(h_f);

  float* d_in = nullptr;
  float* d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partial, size_t(grid) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_f.data(), size_t(cfg.n) * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<float> samples;

  auto run_reduce_tagged = [&](int strat, int tile, const char* tag, bool verify) {
    const float ms = run_reduce(strat, tile, grid, cfg.block, d_in, d_partial, cfg.n,
                                cfg.reps, cfg.warmup, cfg.runs, expect_f, verify, quiet,
                                tag, &samples);
    print_row(tag, tile, cfg.block, grid, cfg.reps, cfg.n, ms, samples, cfg.csv_only);
    return ms;
  };

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) {
      std::printf("%s\n", kCsvHeader);
    } else {
      std::printf("\n== sweep tile (main curve: cg::reduce vs tile size; >32 cliff) ==\n");
    }
    const int kTiles[] = {8, 16, 32, 64, 128};
    float base32 = 0.f;
    bool first = true;
    for (int t : kTiles) {
      if (t > cfg.block) continue;  // tile must fit block
      const float ms = run_reduce_tagged(STRAT_CG_REDUCE, t, "cg_reduce", first);
      first = false;
      if (t == 32) base32 = ms;
    }
    if (base32 > 0.f && !cfg.csv_only) {
      std::printf("(read: normalize each point to tile=32 = %.4f ms to see >32 cost)\n",
                  base32);
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) {
      std::printf("%s\n", kCsvHeader);
    } else {
      std::printf("\n== modes (fixed-point: abstraction tax @ tile=32) ==\n");
    }
    const float ms_intr = run_reduce_tagged(STRAT_INTRINSIC, 32, "intrinsic", true);
    const float ms_t32 = run_reduce_tagged(STRAT_TILE_SHFL, 32, "tile32", true);
    const float ms_cg = run_reduce_tagged(STRAT_CG_REDUCE, 32, "cg_reduce", true);
    if (!cfg.csv_only) {
      std::printf("tax tile32/intrinsic = %.3fx ; cg_reduce/intrinsic = %.3fx\n",
                  (ms_intr > 0 ? ms_t32 / ms_intr : 0.f),
                  (ms_intr > 0 ? ms_cg / ms_intr : 0.f));
    }

    // coalesced aggregation (int)
    std::vector<int> h_i(static_cast<size_t>(cfg.n));
    for (int i = 0; i < cfg.n; ++i) h_i[static_cast<size_t>(i)] = (i % 16) + 1;
    const long long expect_odds = host_count_odds(h_i);
    int* d_i = nullptr;
    unsigned long long* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_i, size_t(cfg.n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_i, h_i.data(), size_t(cfg.n) * sizeof(int),
                          cudaMemcpyHostToDevice));
    const float ms_coal = run_coalesced(grid, cfg.block, d_i, d_count, cfg.n, expect_odds,
                                        cfg.warmup, cfg.runs, quiet, &samples);
    print_row("coalesced", 0, cfg.block, grid, 1, cfg.n, ms_coal, samples, cfg.csv_only);
    CUDA_CHECK(cudaFree(d_i));
    CUDA_CHECK(cudaFree(d_count));

    // cluster (optional, sm_90+)
    if (sm < 90) {
      if (!quiet) {
        std::printf("cluster SKIPPED (need sm_90+, got sm_%d%d)\n", prop.major,
                    prop.minor);
      }
    } else {
      const int cgrid = (grid % cfg.clusize == 0) ? grid : (grid / cfg.clusize) * cfg.clusize;
      bool ok = false;
      const float ms_clu = run_cluster(std::max(cfg.clusize, cgrid), cfg.block,
                                       cfg.clusize, cfg.warmup, cfg.runs, quiet, &samples,
                                       &ok);
      if (ok) {
        print_row("cluster", 0, cfg.block, cgrid, 1, cfg.n, ms_clu, samples,
                  cfg.csv_only);
      }
    }
  } else if (cfg.mode == Mode::Intrinsic) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    run_reduce_tagged(STRAT_INTRINSIC, 32, "intrinsic", true);
  } else if (cfg.mode == Mode::Tile32) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    run_reduce_tagged(STRAT_TILE_SHFL, 32, "tile32", true);
  } else if (cfg.mode == Mode::CgReduce) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    run_reduce_tagged(STRAT_CG_REDUCE, cfg.tile, "cg_reduce", true);
  } else if (cfg.mode == Mode::Coalesced) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    std::vector<int> h_i(static_cast<size_t>(cfg.n));
    for (int i = 0; i < cfg.n; ++i) h_i[static_cast<size_t>(i)] = (i % 16) + 1;
    const long long expect_odds = host_count_odds(h_i);
    int* d_i = nullptr;
    unsigned long long* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_i, size_t(cfg.n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_i, h_i.data(), size_t(cfg.n) * sizeof(int),
                          cudaMemcpyHostToDevice));
    const float ms = run_coalesced(grid, cfg.block, d_i, d_count, cfg.n, expect_odds,
                                   cfg.warmup, cfg.runs, quiet, &samples);
    print_row("coalesced", 0, cfg.block, grid, 1, cfg.n, ms, samples, cfg.csv_only);
    CUDA_CHECK(cudaFree(d_i));
    CUDA_CHECK(cudaFree(d_count));
  } else if (cfg.mode == Mode::Cluster) {
    if (sm < 90) {
      std::fprintf(stderr, "ERROR: --mode cluster requires sm_90+ (got sm_%d%d)\n",
                   prop.major, prop.minor);
      CUDA_CHECK(cudaFree(d_in));
      CUDA_CHECK(cudaFree(d_partial));
      return EXIT_FAILURE;
    }
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const int cgrid = (grid % cfg.clusize == 0) ? grid : (grid / cfg.clusize) * cfg.clusize;
    bool ok = false;
    const float ms = run_cluster(std::max(cfg.clusize, cgrid), cfg.block, cfg.clusize,
                                 cfg.warmup, cfg.runs, quiet, &samples, &ok);
    if (ok) {
      print_row("cluster", 0, cfg.block, cgrid, 1, cfg.n, ms, samples, cfg.csv_only);
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_partial));
  return EXIT_SUCCESS;
}
