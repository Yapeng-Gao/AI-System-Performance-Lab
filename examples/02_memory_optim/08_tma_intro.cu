/**
 * [Module B] B-08. Hopper TMA：GMEM→SMEM bulk / tensor-map micro-bench
 *
 * 模式：
 *   sync     : 协作 sync load 整 tile → compute（公平基线，非 per-thread 小拷贝）
 *   bulk1d   : 1D cp.async.bulk / memcpy_async_tx + mbarrier，立刻 wait
 *   tensor2d : 2D cuTensorMapEncodeTiled + cp.async.bulk.tensor，立刻 wait
 *   pipe2    : 2-stage 1D TMA prefetch ∥ compute（parity mbarrier）
 *   sweep    : 扫 fma-iters，输出 sync/bulk1d/tensor2d/pipe2 加速比 CSV
 *
 * 需要：sm_90+（Hopper / Blackwell），CUDA 12+ / 推荐 12.4+
 * 对齐：GMEM/SMEM 16B；tensor 目标 SMEM 128B；拷贝字节数为 16 的倍数
 */

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
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

#define CU_CHECK(call)                                                           \
  do {                                                                           \
    CUresult res__ = (call);                                                     \
    if (res__ != CUDA_SUCCESS) {                                                 \
      const char* name__ = nullptr;                                              \
      cuGetErrorName(res__, &name__);                                            \
      std::fprintf(stderr, "CUDA Driver Error: %s (%d) at %s:%d\n",             \
                   name__ ? name__ : "unknown", (int)res__, __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Tile: 1024 floats = 4 KiB（16B 对齐友好；2D 为 32×32）
static constexpr int kTileElems = 1024;
static constexpr int kTileW = 32;
static constexpr int kTileH = 32;
static_assert(kTileW * kTileH == kTileElems, "2D tile must match 1D tile elems");

using block_barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

enum class Mode {
  Sync = 0,
  Bulk1d = 1,
  Tensor2d = 2,
  Pipe2 = 3,
  Sweep = 4,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Sync: return "sync";
    case Mode::Bulk1d: return "bulk1d";
    case Mode::Tensor2d: return "tensor2d";
    case Mode::Pipe2: return "pipe2";
    case Mode::Sweep: return "sweep";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "sync") == 0) return Mode::Sync;
  if (std::strcmp(s, "bulk1d") == 0) return Mode::Bulk1d;
  if (std::strcmp(s, "tensor2d") == 0) return Mode::Tensor2d;
  if (std::strcmp(s, "pipe2") == 0) return Mode::Pipe2;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  std::fprintf(stderr,
               "Invalid --mode=%s "
               "(expected sync|bulk1d|tensor2d|pipe2|sweep)\n",
               s);
  std::exit(EXIT_FAILURE);
}

__device__ inline bool is_elected() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const unsigned int tid = threadIdx.x;
  const unsigned int warp_id = tid / 32;
  const unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFFu, warp_id, 0);
  return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFFu));
#else
  return threadIdx.x == 0;
#endif
}

__device__ __forceinline__ float burn_fma(float x, int fma_iters) {
  float acc = 0.f;
#pragma unroll 1
  for (int k = 0; k < fma_iters; ++k) {
    acc = fmaf(x, acc, 1.0f);
  }
  return acc;
}

__device__ __forceinline__ float consume_tile(const float* smem, int fma_iters) {
  float acc = 0.f;
  for (int i = static_cast<int>(threadIdx.x); i < kTileElems;
       i += static_cast<int>(blockDim.x)) {
    acc += burn_fma(smem[i], fma_iters);
  }
  return acc;
}

__device__ __forceinline__ int tiles_for_block(int tile0, int tiles_per_block,
                                              int num_tiles) {
  const int remain = num_tiles - tile0;
  return remain < tiles_per_block ? remain : tiles_per_block;
}

// ---------------------------------------------------------------------------
// A) sync cooperative load（整 tile，公平基线）
// ---------------------------------------------------------------------------
__global__ void kernel_sync(const float* __restrict__ in, float* __restrict__ out,
                            int tiles_per_block, int num_tiles, int fma_iters) {
  __shared__ alignas(16) float smem[kTileElems];
  const int tid = static_cast<int>(threadIdx.x);
  const int tile0 = static_cast<int>(blockIdx.x) * tiles_per_block;
  const int ntiles = tiles_for_block(tile0, tiles_per_block, num_tiles);
  float acc = 0.f;

  for (int t = 0; t < ntiles; ++t) {
    const int tile = tile0 + t;
    const int base = tile * kTileElems;
    for (int i = tid; i < kTileElems; i += static_cast<int>(blockDim.x)) {
      smem[i] = in[base + i];
    }
    __syncthreads();
    acc += consume_tile(smem, fma_iters);
    __syncthreads();
  }
  // 每线程写回，避免非 0 号线程的 FMA 被 DCE
  out[static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid] = acc;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// ---------------------------------------------------------------------------
// B) 1D TMA bulk + mbarrier（立刻 wait，无多 stage overlap）
// ---------------------------------------------------------------------------
__global__ void kernel_bulk1d(const float* __restrict__ in, float* __restrict__ out,
                              int tiles_per_block, int num_tiles, int fma_iters) {
  __shared__ alignas(16) float smem[kTileElems];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier bar;
#pragma nv_diag_default static_var_with_dynamic_init

  const int tid = static_cast<int>(threadIdx.x);
  const int tile0 = static_cast<int>(blockIdx.x) * tiles_per_block;
  const int ntiles = tiles_for_block(tile0, tiles_per_block, num_tiles);
  float acc = 0.f;

  if (tid == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  for (int t = 0; t < ntiles; ++t) {
    const int tile = tile0 + t;
    const float* gptr = in + tile * kTileElems;
    constexpr size_t nbytes = sizeof(float) * kTileElems;

    block_barrier::arrival_token token;
    if (is_elected()) {
      cuda::device::memcpy_async_tx(smem, gptr, cuda::aligned_size_t<16>(nbytes), bar);
      token = cuda::device::barrier_arrive_tx(bar, 1, nbytes);
    } else {
      token = bar.arrive();
    }
    bar.wait(std::move(token));

    acc += consume_tile(smem, fma_iters);
    __syncthreads();
  }
  out[static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid] = acc;
}

// ---------------------------------------------------------------------------
// C) 2D tensor-map TMA（立刻 wait）
// ---------------------------------------------------------------------------
__global__ void kernel_tensor2d(const __grid_constant__ CUtensorMap tensor_map,
                                float* __restrict__ out, int tiles_per_block,
                                int num_tiles, int fma_iters, int tiles_x) {
  // multi-D bulk tensor：目标 SMEM 需 128B 对齐
  __shared__ alignas(128) float smem[kTileH][kTileW];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier bar;
#pragma nv_diag_default static_var_with_dynamic_init

  const int tid = static_cast<int>(threadIdx.x);
  const int tile0 = static_cast<int>(blockIdx.x) * tiles_per_block;
  const int ntiles = tiles_for_block(tile0, tiles_per_block, num_tiles);
  float acc = 0.f;

  if (tid == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  for (int t = 0; t < ntiles; ++t) {
    const int tile = tile0 + t;
    const int tile_x = tile % tiles_x;
    const int tile_y = tile / tiles_x;
    const int32_t coords[2] = {tile_x * kTileW, tile_y * kTileH};

    block_barrier::arrival_token token;
    if (is_elected()) {
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &smem[0][0],
                                &tensor_map, coords,
                                cuda::device::barrier_native_handle(bar));
      token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
    } else {
      token = bar.arrive();
    }
    bar.wait(std::move(token));

    acc += consume_tile(&smem[0][0], fma_iters);
    __syncthreads();
  }
  out[static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid] = acc;
}

// ---------------------------------------------------------------------------
// D) 2-stage 1D TMA prefetch ∥ compute
// ---------------------------------------------------------------------------
__global__ void kernel_pipe2(const float* __restrict__ in, float* __restrict__ out,
                             int tiles_per_block, int num_tiles, int fma_iters) {
  constexpr int kStages = 2;
  __shared__ alignas(16) float smem[kStages][kTileElems];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier bar[kStages];
#pragma nv_diag_default static_var_with_dynamic_init

  const int tid = static_cast<int>(threadIdx.x);
  const int tile0 = static_cast<int>(blockIdx.x) * tiles_per_block;
  const int ntiles = tiles_for_block(tile0, tiles_per_block, num_tiles);
  float acc = 0.f;
  constexpr size_t nbytes = sizeof(float) * kTileElems;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      init(&bar[s], 1);
    }
  }
  __syncthreads();

  auto issue = [&](int tile_rel, int stage) {
    if (tile_rel >= ntiles) return;
    const float* gptr = in + (tile0 + tile_rel) * kTileElems;
    cuda::device::memcpy_async_tx(smem[stage], gptr, cuda::aligned_size_t<16>(nbytes),
                                  bar[stage]);
    (void)cuda::device::barrier_arrive_tx(bar[stage], 1, nbytes);
  };

  if (is_elected()) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      issue(s, s);
    }
  }

  int stage = 0;
  uint32_t parity = 0;
  for (int t = 0; t < ntiles; ++t) {
    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                          cuda::device::barrier_native_handle(bar[stage]),
                                          parity)) {
    }

    acc += consume_tile(smem[stage], fma_iters);
    __syncthreads();

    if (is_elected()) {
      issue(t + kStages, stage);
    }

    stage++;
    if (stage == kStages) {
      stage = 0;
      parity ^= 1u;
    }
  }

  out[static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid] = acc;
}

#else  // !sm_90+

__global__ void kernel_bulk1d(const float*, float*, int, int, int) {}
__global__ void kernel_tensor2d(const __grid_constant__ CUtensorMap, float*, int, int, int,
                                int) {}
__global__ void kernel_pipe2(const float*, float*, int, int, int) {}

#endif

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static float percentile_of(std::vector<float> v, float p) {
  if (v.empty()) return 0.0f;
  p = std::max(0.0f, std::min(100.0f, p));
  std::sort(v.begin(), v.end());
  const float pos = (p / 100.0f) * float(v.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const float t = pos - float(lo);
  return v[lo] * (1.0f - t) + v[hi] * t;
}

static PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  cudaDriverEntryPointQueryResult driver_status;
  void* ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000,
                                              cudaEnableDefault, &driver_status));
  if (driver_status != cudaDriverEntryPointSuccess || ptr == nullptr) {
    std::fprintf(stderr, "ERROR: cuTensorMapEncodeTiled entry point unavailable\n");
    std::exit(EXIT_FAILURE);
  }
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
}

static CUtensorMap make_tensor_map_2d(float* d_in, int g_width, int g_height) {
  CUtensorMap tensor_map{};
  constexpr cuuint32_t rank = 2;
  const cuuint64_t size[rank] = {static_cast<cuuint64_t>(g_width),
                                 static_cast<cuuint64_t>(g_height)};
  const cuuint64_t stride[rank - 1] = {static_cast<cuuint64_t>(g_width) * sizeof(float)};
  const cuuint32_t box_size[rank] = {static_cast<cuuint32_t>(kTileW),
                                     static_cast<cuuint32_t>(kTileH)};
  const cuuint32_t elem_stride[rank] = {1, 1};

  auto encode = get_cuTensorMapEncodeTiled();
  CU_CHECK(encode(
      &tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, rank, d_in, size,
      stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensor_map;
}

struct BenchConfig {
  Mode mode = Mode::Bulk1d;
  int n = 1 << 22;  // floats；须能整除 tile
  int tiles = 64;   // tiles per block
  int block = 256;
  int fma_iters = 8;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

struct LaunchPlan {
  int grid = 0;
  int tiles_per_block = 0;
  int num_tiles = 0;
  int tiles_x = 0;
  int g_width = 0;
  int g_height = 0;
};

static LaunchPlan make_plan(const BenchConfig& c) {
  LaunchPlan p;
  if (c.n % kTileElems != 0) {
    std::fprintf(stderr, "ERROR: --n=%d must be multiple of tile elems (%d)\n", c.n,
                 kTileElems);
    std::exit(EXIT_FAILURE);
  }
  p.num_tiles = c.n / kTileElems;
  p.tiles_per_block = std::max(1, c.tiles);
  p.grid = (p.num_tiles + p.tiles_per_block - 1) / p.tiles_per_block;
  // 2D：宽取能整除的 tile 列数；尽量方阵倾向
  p.tiles_x = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(p.num_tiles))));
  while (p.tiles_x > 1 && (p.num_tiles % p.tiles_x) != 0) {
    --p.tiles_x;
  }
  p.g_width = p.tiles_x * kTileW;
  p.g_height = (p.num_tiles / p.tiles_x) * kTileH;
  if (p.g_width * p.g_height != c.n) {
    std::fprintf(stderr, "ERROR: failed to factor n into 2D grid\n");
    std::exit(EXIT_FAILURE);
  }
  return p;
}

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <sync|bulk1d|tensor2d|pipe2|sweep> [options]\n"
      "  --n <elems>          total floats, multiple of %d (default 1<<22)\n"
      "  --tiles <n>          tiles per block (default 64)\n"
      "  --block <threads>    block size (default 256)\n"
      "  --fma-iters <n>      FMAs per smem element (default 8)\n"
      "  --runs <n>           timed runs (default 7)\n"
      "  --warmup <n>         warmup runs (default 2)\n"
      "  --device <id>        GPU id (default 0)\n"
      "  --csv-only           only print CSV line(s)\n",
      prog, kTileElems);
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
  return percentile_of(*samples, 50.f);
}

static void run_one(const BenchConfig& c, Mode mode, float* d_in, float* d_out,
                    const CUtensorMap* tensor_map, const LaunchPlan& plan) {
  std::vector<float> samples;
  const float med = time_launch_ms(
      [&]() {
        switch (mode) {
          case Mode::Sync:
            kernel_sync<<<plan.grid, c.block>>>(d_in, d_out, plan.tiles_per_block,
                                                plan.num_tiles, c.fma_iters);
            break;
          case Mode::Bulk1d:
            kernel_bulk1d<<<plan.grid, c.block>>>(d_in, d_out, plan.tiles_per_block,
                                                  plan.num_tiles, c.fma_iters);
            break;
          case Mode::Tensor2d:
            kernel_tensor2d<<<plan.grid, c.block>>>(
                *tensor_map, d_out, plan.tiles_per_block, plan.num_tiles, c.fma_iters,
                plan.tiles_x);
            break;
          case Mode::Pipe2:
            kernel_pipe2<<<plan.grid, c.block>>>(d_in, d_out, plan.tiles_per_block,
                                                 plan.num_tiles, c.fma_iters);
            break;
          default:
            break;
        }
        CUDA_CHECK(cudaGetLastError());
      },
      c.warmup, c.runs, &samples);

  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean =
      samples.empty()
          ? 0.f
          : std::accumulate(samples.begin(), samples.end(), 0.f) / float(samples.size());
  const double bytes = double(c.n) * sizeof(float);
  const double gbps = (med > 0.f) ? (bytes / 1.0e9) / (double(med) / 1.0e3) : 0.0;

  if (c.csv_only) {
    std::printf("%s,%.6f\n", mode_name(mode), med);
  } else {
    std::printf(
        "mode=%-9s fma_iters=%-5d tiles/block=%d  first=%.4f  median=%.4f  p95=%.4f  "
        "mean=%.4f ms  ~%.2f GB/s (payload read)\n",
        mode_name(mode), c.fma_iters, plan.tiles_per_block, first, med, p95, mean, gbps);
  }
}

static float median_mode(const BenchConfig& c, Mode mode, float* d_in, float* d_out,
                         const CUtensorMap* tensor_map, const LaunchPlan& plan) {
  BenchConfig tmp = c;
  tmp.csv_only = true;
  std::vector<float> samples;
  return time_launch_ms(
      [&]() {
        switch (mode) {
          case Mode::Sync:
            kernel_sync<<<plan.grid, tmp.block>>>(d_in, d_out, plan.tiles_per_block,
                                                  plan.num_tiles, tmp.fma_iters);
            break;
          case Mode::Bulk1d:
            kernel_bulk1d<<<plan.grid, tmp.block>>>(d_in, d_out, plan.tiles_per_block,
                                                    plan.num_tiles, tmp.fma_iters);
            break;
          case Mode::Tensor2d:
            kernel_tensor2d<<<plan.grid, tmp.block>>>(
                *tensor_map, d_out, plan.tiles_per_block, plan.num_tiles, tmp.fma_iters,
                plan.tiles_x);
            break;
          case Mode::Pipe2:
            kernel_pipe2<<<plan.grid, tmp.block>>>(d_in, d_out, plan.tiles_per_block,
                                                   plan.num_tiles, tmp.fma_iters);
            break;
          default:
            break;
        }
        CUDA_CHECK(cudaGetLastError());
      },
      tmp.warmup, tmp.runs, &samples);
}

static void run_sweep(const BenchConfig& c, float* d_in, float* d_out,
                      const CUtensorMap& tensor_map, const LaunchPlan& plan) {
  const int iters[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  if (!c.csv_only) {
    std::printf("\n=== intensity sweep (speedup = sync_median / mode_median) ===\n");
    std::printf(
        "fma_iters,sync_ms,bulk1d_ms,tensor2d_ms,pipe2_ms,"
        "speedup_bulk1d,speedup_tensor2d,speedup_pipe2\n");
  }
  for (int fi : iters) {
    BenchConfig cfg = c;
    cfg.fma_iters = fi;
    const float m_sync = median_mode(cfg, Mode::Sync, d_in, d_out, &tensor_map, plan);
    const float m_b1 = median_mode(cfg, Mode::Bulk1d, d_in, d_out, &tensor_map, plan);
    const float m_t2 = median_mode(cfg, Mode::Tensor2d, d_in, d_out, &tensor_map, plan);
    const float m_p2 = median_mode(cfg, Mode::Pipe2, d_in, d_out, &tensor_map, plan);
    const float sb = m_b1 > 0.f ? m_sync / m_b1 : 0.f;
    const float st = m_t2 > 0.f ? m_sync / m_t2 : 0.f;
    const float sp = m_p2 > 0.f ? m_sync / m_p2 : 0.f;
    std::printf("%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n", fi, m_sync, m_b1, m_t2, m_p2,
                sb, st, sp);
  }
}

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  if (prop.major < 9) {
    std::fprintf(stderr,
                 "ERROR: need compute capability >= 9.0 for TMA; got sm_%d%d\n",
                 prop.major, prop.minor);
    return EXIT_FAILURE;
  }

  const LaunchPlan plan = make_plan(cfg);

  if (!cfg.csv_only) {
    std::printf("GPU: %s | sm_%d%d | sharedMemPerBlock=%zu KB\n", prop.name, prop.major,
                prop.minor, prop.sharedMemPerBlock / 1024);
    std::printf(
        "n=%d (%d tiles) tiles/block=%d grid=%d block=%d fma_iters=%d | 2D=%dx%d "
        "(tiles_x=%d)\n",
        cfg.n, plan.num_tiles, plan.tiles_per_block, plan.grid, cfg.block, cfg.fma_iters,
        plan.g_width, plan.g_height, plan.tiles_x);
  }

  float *d_in = nullptr, *d_out = nullptr;
  const size_t out_elems = size_t(plan.grid) * size_t(cfg.block);
  CUDA_CHECK(cudaMalloc(&d_in, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, size_t(cfg.n) * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(float)));

  const CUtensorMap tensor_map = make_tensor_map_2d(d_in, plan.g_width, plan.g_height);

  if (cfg.mode == Mode::Sweep) {
    run_sweep(cfg, d_in, d_out, tensor_map, plan);
  } else {
    run_one(cfg, cfg.mode, d_in, d_out, &tensor_map, plan);
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
