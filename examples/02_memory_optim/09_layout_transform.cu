/**
 * [Module B] B-09. 数据布局（AoS/SoA/Transpose）：一次布局调整带来的事务变化
 *
 * 模式：
 *   aos              : 宽 struct（8×float），按 touch_fields 读写前 k 个字段
 *   soa              : 同逻辑 SoA（8 个 float 数组）
 *   copy             : 方阵 GMEM→GMEM copy（transpose 带宽上限）
 *   transpose_naive  : 直接跨步写（无 SMEM）
 *   transpose_tiled  : SMEM tile 重排，读合写合
 *   transpose_pad    : 同上 + tile[T][T+1] 消 bank conflict（挂钩 B-02）
 *   sweep            : 扫 touch_fields∈{1,2,4,8}，输出 aos/soa 时延与加速比 CSV
 *   modes            : 一次跑齐 layout(默认 touch=1) + transpose 全表
 *
 * 主证据：CUDA event median → useful-payload GB/s（仅计触达字段字节）
 * 硬件：不限 sm_90+（全架构合并问题）
 * 矩阵 mode 计时前做轻量正确性检查（角点 + 步长抽样）
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

static constexpr int kNumFields = 8;
static constexpr int kTileDim = 32;
static constexpr int kBlockRows = 8;  // Harris-style: 32×8 threads cover 32×32 tile

struct ParticleAoS {
  float f[kNumFields];
};

enum class Mode {
  Aos = 0,
  Soa = 1,
  Copy = 2,
  TransposeNaive = 3,
  TransposeTiled = 4,
  TransposePad = 5,
  Sweep = 6,
  Modes = 7,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Aos: return "aos";
    case Mode::Soa: return "soa";
    case Mode::Copy: return "copy";
    case Mode::TransposeNaive: return "transpose_naive";
    case Mode::TransposeTiled: return "transpose_tiled";
    case Mode::TransposePad: return "transpose_pad";
    case Mode::Sweep: return "sweep";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "aos") == 0) return Mode::Aos;
  if (std::strcmp(s, "soa") == 0) return Mode::Soa;
  if (std::strcmp(s, "copy") == 0) return Mode::Copy;
  if (std::strcmp(s, "transpose_naive") == 0) return Mode::TransposeNaive;
  if (std::strcmp(s, "transpose_tiled") == 0) return Mode::TransposeTiled;
  if (std::strcmp(s, "transpose_pad") == 0) return Mode::TransposePad;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected aos|soa|copy|transpose_naive|"
               "transpose_tiled|transpose_pad|sweep|modes)\n",
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
// AoS / SoA：每线程处理一个粒子；读写前 touch_fields 个字段（+1.0f 写回防 DCE）
// ---------------------------------------------------------------------------
__global__ void kernel_aos(ParticleAoS* __restrict__ p, int n, int touch_fields) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // 按字段读写：禁止整 struct 赋值，否则编译器会一次搬满 32B，掩盖 stride 故事
  if (touch_fields >= 1) p[i].f[0] = p[i].f[0] + 1.0f;
  if (touch_fields >= 2) p[i].f[1] = p[i].f[1] + 1.0f;
  if (touch_fields >= 3) p[i].f[2] = p[i].f[2] + 1.0f;
  if (touch_fields >= 4) p[i].f[3] = p[i].f[3] + 1.0f;
  if (touch_fields >= 5) p[i].f[4] = p[i].f[4] + 1.0f;
  if (touch_fields >= 6) p[i].f[5] = p[i].f[5] + 1.0f;
  if (touch_fields >= 7) p[i].f[6] = p[i].f[6] + 1.0f;
  if (touch_fields >= 8) p[i].f[7] = p[i].f[7] + 1.0f;
}

__device__ __forceinline__ void soa_touch(float* __restrict__ col, int i) {
  col[i] = col[i] + 1.0f;
}

__global__ void kernel_soa(float* __restrict__ f0, float* __restrict__ f1,
                           float* __restrict__ f2, float* __restrict__ f3,
                           float* __restrict__ f4, float* __restrict__ f5,
                           float* __restrict__ f6, float* __restrict__ f7, int n,
                           int touch_fields) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // 显式分支：保证只触达前 k 列，且每列相邻线程地址连续
  if (touch_fields >= 1) soa_touch(f0, i);
  if (touch_fields >= 2) soa_touch(f1, i);
  if (touch_fields >= 3) soa_touch(f2, i);
  if (touch_fields >= 4) soa_touch(f3, i);
  if (touch_fields >= 5) soa_touch(f4, i);
  if (touch_fields >= 6) soa_touch(f5, i);
  if (touch_fields >= 7) soa_touch(f6, i);
  if (touch_fields >= 8) soa_touch(f7, i);
}

// ---------------------------------------------------------------------------
// Matrix: copy / transpose family（方阵 dim×dim）
// ---------------------------------------------------------------------------
// Host/device 同公式（校验用）
static __host__ __device__ __forceinline__ float matrix_pattern(int row, int col) {
  return float(row) * 0.5f + float(col) * 1.0e-3f;
}

__global__ void kernel_fill_matrix(float* __restrict__ out, int dim) {
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = y + j;
    if (x < dim && yy < dim) {
      out[yy * dim + x] = matrix_pattern(yy, x);
    }
  }
}

__global__ void kernel_copy(const float* __restrict__ in, float* __restrict__ out,
                            int dim) {
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = y + j;
    if (x < dim && yy < dim) {
      out[yy * dim + x] = in[yy * dim + x];
    }
  }
}

__global__ void kernel_transpose_naive(const float* __restrict__ in,
                                       float* __restrict__ out, int dim) {
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = y + j;
    if (x < dim && yy < dim) {
      // 读合并，写跨步
      out[x * dim + yy] = in[yy * dim + x];
    }
  }
}

__global__ void kernel_transpose_tiled(const float* __restrict__ in,
                                       float* __restrict__ out, int dim) {
  __shared__ float tile[kTileDim][kTileDim];
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = y + j;
    if (x < dim && yy < dim) {
      tile[threadIdx.y + j][threadIdx.x] = in[yy * dim + x];
    }
  }
  __syncthreads();

  const int xt = blockIdx.y * kTileDim + threadIdx.x;
  const int yt = blockIdx.x * kTileDim + threadIdx.y;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = yt + j;
    if (xt < dim && yy < dim) {
      out[yy * dim + xt] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void kernel_transpose_pad(const float* __restrict__ in,
                                     float* __restrict__ out, int dim) {
  __shared__ float tile[kTileDim][kTileDim + 1];
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = y + j;
    if (x < dim && yy < dim) {
      tile[threadIdx.y + j][threadIdx.x] = in[yy * dim + x];
    }
  }
  __syncthreads();

  const int xt = blockIdx.y * kTileDim + threadIdx.x;
  const int yt = blockIdx.x * kTileDim + threadIdx.y;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = yt + j;
    if (xt < dim && yy < dim) {
      out[yy * dim + xt] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 1 << 22;          // particles for AoS/SoA
  int dim = 4096;           // matrix edge for transpose/copy
  int touch_fields = 1;     // 1..8
  int block = 256;
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <aos|soa|copy|transpose_naive|transpose_tiled|"
      "transpose_pad|sweep|modes> [options]\n"
      "  --n <particles>       AoS/SoA particle count (default 1<<22)\n"
      "  --dim <edge>          square matrix edge, multiple of 32 (default 4096)\n"
      "  --touch-fields <k>    fields read/written in AoS/SoA, 1..8 (default 1)\n"
      "  --block <threads>     1D block for AoS/SoA (default 256)\n"
      "  --runs <n>            timed runs (default 7)\n"
      "  --warmup <n>          warmup runs (default 2)\n"
      "  --device <id>         GPU id (default 0)\n"
      "  --csv-only            only print CSV line(s)\n",
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
    } else if (std::strcmp(argv[i], "--dim") == 0) {
      c.dim = std::atoi(need("--dim"));
    } else if (std::strcmp(argv[i], "--touch-fields") == 0) {
      c.touch_fields = std::atoi(need("--touch-fields"));
    } else if (std::strcmp(argv[i], "--block") == 0) {
      c.block = std::atoi(need("--block"));
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
  if (c.touch_fields < 1 || c.touch_fields > kNumFields) {
    std::fprintf(stderr, "ERROR: --touch-fields must be in [1,%d]\n", kNumFields);
    std::exit(EXIT_FAILURE);
  }
  if (c.dim <= 0 || (c.dim % kTileDim) != 0) {
    std::fprintf(stderr, "ERROR: --dim must be positive multiple of %d\n", kTileDim);
    std::exit(EXIT_FAILURE);
  }
  if (c.n <= 0 || c.block <= 0) {
    std::fprintf(stderr, "ERROR: --n and --block must be positive\n");
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

static double useful_bytes_layout(int n, int touch_fields) {
  // read + write touched fields
  return double(n) * double(touch_fields) * sizeof(float) * 2.0;
}

static double useful_bytes_matrix(int dim) {
  // read + write full matrix
  return double(dim) * double(dim) * sizeof(float) * 2.0;
}

static double gbps_from_ms(double bytes, float med_ms) {
  if (med_ms <= 0.f) return 0.0;
  return (bytes / 1.0e9) / (double(med_ms) / 1.0e3);
}

struct LayoutBuf {
  ParticleAoS* d_aos = nullptr;
  float* d_soa[kNumFields] = {};
  int n = 0;
};

struct MatrixBuf {
  float* d_in = nullptr;
  float* d_out = nullptr;
  int dim = 0;
};

static void alloc_layout(LayoutBuf* b, int n) {
  b->n = n;
  CUDA_CHECK(cudaMalloc(&b->d_aos, size_t(n) * sizeof(ParticleAoS)));
  CUDA_CHECK(cudaMemset(b->d_aos, 0, size_t(n) * sizeof(ParticleAoS)));
  for (int f = 0; f < kNumFields; ++f) {
    CUDA_CHECK(cudaMalloc(&b->d_soa[f], size_t(n) * sizeof(float)));
    CUDA_CHECK(cudaMemset(b->d_soa[f], 0, size_t(n) * sizeof(float)));
  }
}

static void free_layout(LayoutBuf* b) {
  if (b->d_aos) CUDA_CHECK(cudaFree(b->d_aos));
  for (int f = 0; f < kNumFields; ++f) {
    if (b->d_soa[f]) CUDA_CHECK(cudaFree(b->d_soa[f]));
  }
}

static void alloc_matrix(MatrixBuf* b, int dim) {
  b->dim = dim;
  const size_t bytes = size_t(dim) * size_t(dim) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&b->d_in, bytes));
  CUDA_CHECK(cudaMalloc(&b->d_out, bytes));
  CUDA_CHECK(cudaMemset(b->d_in, 0, bytes));
  CUDA_CHECK(cudaMemset(b->d_out, 0, bytes));
}

static void free_matrix(MatrixBuf* b) {
  if (b->d_in) CUDA_CHECK(cudaFree(b->d_in));
  if (b->d_out) CUDA_CHECK(cudaFree(b->d_out));
}

static dim3 matrix_grid(int dim) {
  return dim3((dim + kTileDim - 1) / kTileDim, (dim + kTileDim - 1) / kTileDim);
}

static dim3 matrix_block() { return dim3(kTileDim, kBlockRows); }

static float run_layout(const BenchConfig& c, Mode mode, LayoutBuf& buf,
                        int touch_fields, std::vector<float>* samples) {
  const int grid = (c.n + c.block - 1) / c.block;
  return time_launch_ms(
      [&]() {
        if (mode == Mode::Aos) {
          kernel_aos<<<grid, c.block>>>(buf.d_aos, c.n, touch_fields);
        } else {
          kernel_soa<<<grid, c.block>>>(buf.d_soa[0], buf.d_soa[1], buf.d_soa[2],
                                        buf.d_soa[3], buf.d_soa[4], buf.d_soa[5],
                                        buf.d_soa[6], buf.d_soa[7], c.n, touch_fields);
        }
        CUDA_CHECK(cudaGetLastError());
      },
      c.warmup, c.runs, samples);
}

static void launch_matrix_kernel(Mode mode, MatrixBuf& buf) {
  const dim3 grid = matrix_grid(buf.dim);
  const dim3 block = matrix_block();
  switch (mode) {
    case Mode::Copy:
      kernel_copy<<<grid, block>>>(buf.d_in, buf.d_out, buf.dim);
      break;
    case Mode::TransposeNaive:
      kernel_transpose_naive<<<grid, block>>>(buf.d_in, buf.d_out, buf.dim);
      break;
    case Mode::TransposeTiled:
      kernel_transpose_tiled<<<grid, block>>>(buf.d_in, buf.d_out, buf.dim);
      break;
    case Mode::TransposePad:
      kernel_transpose_pad<<<grid, block>>>(buf.d_in, buf.d_out, buf.dim);
      break;
    default:
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

// 轻量正确性：填已知图案 → 跑一次 → 抽样+角点校验（全量对 4096^2 也便宜，这里用步长抽样）
static void verify_matrix_mode(Mode mode, MatrixBuf& buf, bool quiet) {
  const int dim = buf.dim;
  const dim3 grid = matrix_grid(dim);
  const dim3 block = matrix_block();
  kernel_fill_matrix<<<grid, block>>>(buf.d_in, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(buf.d_out, 0, size_t(dim) * size_t(dim) * sizeof(float)));
  launch_matrix_kernel(mode, buf);
  CUDA_CHECK(cudaDeviceSynchronize());

  const size_t n = size_t(dim) * size_t(dim);
  std::vector<float> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), buf.d_out, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const bool is_copy = (mode == Mode::Copy);
  auto expect = [&](int row, int col) -> float {
    // copy: out[r,c]=in[r,c]；transpose: out[r,c]=in[c,r]
    if (is_copy) return matrix_pattern(row, col);
    return matrix_pattern(col, row);
  };
  auto nearly_eq = [](float a, float b) {
    return fabsf(a - b) <= 1.0e-5f * (1.0f + fabsf(b));
  };

  int mismatches = 0;
  const int corners[4][2] = {{0, 0}, {0, dim - 1}, {dim - 1, 0}, {dim - 1, dim - 1}};
  for (const auto& rc : corners) {
    const int r = rc[0], c = rc[1];
    const float got = h_out[size_t(r) * size_t(dim) + size_t(c)];
    if (!nearly_eq(got, expect(r, c))) ++mismatches;
  }
  const int step = std::max(1, dim / 64);
  for (int r = 0; r < dim; r += step) {
    for (int c = 0; c < dim; c += step) {
      const float got = h_out[size_t(r) * size_t(dim) + size_t(c)];
      if (!nearly_eq(got, expect(r, c))) ++mismatches;
    }
  }
  if (mismatches != 0) {
    std::fprintf(stderr,
                 "ERROR: correctness check failed for mode=%s (mismatches=%d, dim=%d)\n",
                 mode_name(mode), mismatches, dim);
    std::exit(EXIT_FAILURE);
  }
  if (!quiet) {
    std::printf("verify %-16s OK (corners + stride-%d samples)\n", mode_name(mode),
                step);
  }
}

static float run_matrix(const BenchConfig& c, Mode mode, MatrixBuf& buf,
                        std::vector<float>* samples) {
  verify_matrix_mode(mode, buf, c.csv_only);
  // 计时前重新填输入并清空输出，避免校验图案干扰（虽对带宽影响小）
  {
    const dim3 grid = matrix_grid(buf.dim);
    const dim3 block = matrix_block();
    kernel_fill_matrix<<<grid, block>>>(buf.d_in, buf.dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(buf.d_out, 0, size_t(buf.dim) * size_t(buf.dim) * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  return time_launch_ms([&]() { launch_matrix_kernel(mode, buf); }, c.warmup, c.runs,
                        samples);
}

static void print_layout_row(const BenchConfig& c, Mode mode, float med,
                             const std::vector<float>& samples, int touch_fields) {
  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean =
      samples.empty()
          ? 0.f
          : std::accumulate(samples.begin(), samples.end(), 0.f) / float(samples.size());
  const double gbps = gbps_from_ms(useful_bytes_layout(c.n, touch_fields), med);
  if (c.csv_only) {
    std::printf("%s,%d,%.6f,%.4f\n", mode_name(mode), touch_fields, med, gbps);
  } else {
    std::printf(
        "mode=%-16s touch=%d  first=%.4f  median=%.4f  p95=%.4f  mean=%.4f ms  "
        "~%.2f GB/s (useful R+W)\n",
        mode_name(mode), touch_fields, first, med, p95, mean, gbps);
  }
}

static void print_matrix_row(const BenchConfig& c, Mode mode, float med,
                             const std::vector<float>& samples) {
  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean =
      samples.empty()
          ? 0.f
          : std::accumulate(samples.begin(), samples.end(), 0.f) / float(samples.size());
  const double gbps = gbps_from_ms(useful_bytes_matrix(c.dim), med);
  if (c.csv_only) {
    std::printf("%s,%d,%.6f,%.4f\n", mode_name(mode), c.dim, med, gbps);
  } else {
    std::printf(
        "mode=%-16s dim=%d  first=%.4f  median=%.4f  p95=%.4f  mean=%.4f ms  "
        "~%.2f GB/s (useful R+W)\n",
        mode_name(mode), c.dim, first, med, p95, mean, gbps);
  }
}

static void run_sweep(const BenchConfig& c, LayoutBuf& buf) {
  const int touches[] = {1, 2, 4, 8};
  if (!c.csv_only) {
    std::printf("\n=== touch_fields sweep (speedup = aos_median / soa_median) ===\n");
    std::printf(
        "touch_fields,aos_ms,soa_ms,aos_gbps,soa_gbps,speedup_soa\n");
  } else {
    std::printf(
        "touch_fields,aos_ms,soa_ms,aos_gbps,soa_gbps,speedup_soa\n");
  }
  std::vector<float> samples;
  for (int tf : touches) {
    const float m_aos = run_layout(c, Mode::Aos, buf, tf, &samples);
    const double g_aos = gbps_from_ms(useful_bytes_layout(c.n, tf), m_aos);
    const float m_soa = run_layout(c, Mode::Soa, buf, tf, &samples);
    const double g_soa = gbps_from_ms(useful_bytes_layout(c.n, tf), m_soa);
    const float sp = m_soa > 0.f ? m_aos / m_soa : 0.f;
    std::printf("%d,%.6f,%.6f,%.4f,%.4f,%.4f\n", tf, m_aos, m_soa, g_aos, g_soa, sp);
  }
}

static void run_modes(const BenchConfig& c, LayoutBuf& buf, MatrixBuf& mat) {
  std::vector<float> samples;
  struct Row {
    const char* mode;
    float ms;
    double gbps;
  };
  std::vector<Row> rows;
  rows.reserve(6);

  if (!c.csv_only) {
    std::printf("\n=== layout (touch_fields=%d) ===\n", c.touch_fields);
  }
  {
    const float m = run_layout(c, Mode::Aos, buf, c.touch_fields, &samples);
    print_layout_row(c, Mode::Aos, m, samples, c.touch_fields);
    rows.push_back({mode_name(Mode::Aos), m,
                    gbps_from_ms(useful_bytes_layout(c.n, c.touch_fields), m)});
  }
  {
    const float m = run_layout(c, Mode::Soa, buf, c.touch_fields, &samples);
    print_layout_row(c, Mode::Soa, m, samples, c.touch_fields);
    rows.push_back({mode_name(Mode::Soa), m,
                    gbps_from_ms(useful_bytes_layout(c.n, c.touch_fields), m)});
  }

  if (!c.csv_only) {
    std::printf("\n=== transpose / copy (dim=%d) ===\n", c.dim);
  }
  const Mode mats[] = {Mode::Copy, Mode::TransposeNaive, Mode::TransposeTiled,
                       Mode::TransposePad};
  for (Mode m : mats) {
    const float med = run_matrix(c, m, mat, &samples);
    print_matrix_row(c, m, med, samples);
    rows.push_back(
        {mode_name(m), med, gbps_from_ms(useful_bytes_matrix(c.dim), med)});
  }

  // Machine-friendly block for docs/results/B-09_modes.csv
  std::printf("\nmode,median_ms,useful_gbps\n");
  for (const Row& r : rows) {
    std::printf("%s,%.6f,%.4f\n", r.mode, r.ms, r.gbps);
  }
}

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

  if (!cfg.csv_only) {
    std::printf("GPU: %s | sm_%d%d\n", prop.name, prop.major, prop.minor);
    std::printf("n=%d particles | dim=%d | touch_fields=%d | block=%d | runs=%d "
                "warmup=%d\n",
                cfg.n, cfg.dim, cfg.touch_fields, cfg.block, cfg.runs, cfg.warmup);
    std::printf("AoS sizeof(Particle)=%zu B | useful-GB/s counts touched fields R+W\n",
                sizeof(ParticleAoS));
  }

  const bool need_layout =
      cfg.mode == Mode::Aos || cfg.mode == Mode::Soa || cfg.mode == Mode::Sweep ||
      cfg.mode == Mode::Modes;
  const bool need_matrix =
      cfg.mode == Mode::Copy || cfg.mode == Mode::TransposeNaive ||
      cfg.mode == Mode::TransposeTiled || cfg.mode == Mode::TransposePad ||
      cfg.mode == Mode::Modes;

  LayoutBuf layout{};
  MatrixBuf matrix{};
  if (need_layout) alloc_layout(&layout, cfg.n);
  if (need_matrix) alloc_matrix(&matrix, cfg.dim);

  std::vector<float> samples;
  switch (cfg.mode) {
    case Mode::Sweep:
      run_sweep(cfg, layout);
      break;
    case Mode::Modes:
      run_modes(cfg, layout, matrix);
      break;
    case Mode::Aos:
    case Mode::Soa: {
      const float med =
          run_layout(cfg, cfg.mode, layout, cfg.touch_fields, &samples);
      print_layout_row(cfg, cfg.mode, med, samples, cfg.touch_fields);
      break;
    }
    case Mode::Copy:
    case Mode::TransposeNaive:
    case Mode::TransposeTiled:
    case Mode::TransposePad: {
      const float med = run_matrix(cfg, cfg.mode, matrix, &samples);
      print_matrix_row(cfg, cfg.mode, med, samples);
      break;
    }
    default:
      break;
  }

  if (need_layout) free_layout(&layout);
  if (need_matrix) free_matrix(&matrix);
  return 0;
}
