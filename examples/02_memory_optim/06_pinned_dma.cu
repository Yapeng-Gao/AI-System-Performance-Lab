/**
 * [Module B] B-06. Pinned Memory 与 DMA：H2D/D2H 吞吐与 Overlap
 *
 * 模式：
 *   pageable : malloc + cudaMemcpyAsync H2D（伪异步对照）
 *   pinned   : cudaMallocHost + Async H2D（单向 DMA 上限）
 *   serial   : pinned + 单 stream 切块 H2D→Kernel（无跨流重叠）
 *   overlap  : pinned + 多 stream 切块 H2D→Kernel（跨 chunk 重叠）
 *   bidir    : pinned + H2D ∥ D2H
 *   mapped   : cudaHostAllocMapped，kernel 直读 host（Zero-Copy）
 *
 * 输出：first / median / p95 / mean（ms）与 GB/s，以及 CSV 行。
 *
 * 注意：
 * - overlap 的“真重叠”发生在不同 stream 的不同 chunk 之间：
 *   Stream0: [H2D0][K0][H2D2][K2]...
 *   Stream1:      [H2D1][K1][H2D3][K3]...
 *   同一 stream 内 H2D→Kernel 仍保序。
 * - Windows WDDM 消费级驱动可能抑制 copy∥compute；以 NSYS 为准。
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
  Pageable = 0,
  Pinned = 1,
  Serial = 2,
  Overlap = 3,
  Bidir = 4,
  Mapped = 5,
};

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Pageable: return "pageable";
    case Mode::Pinned:   return "pinned";
    case Mode::Serial:   return "serial";
    case Mode::Overlap:  return "overlap";
    case Mode::Bidir:    return "bidir";
    case Mode::Mapped:   return "mapped";
    default:             return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "pageable") == 0) return Mode::Pageable;
  if (std::strcmp(s, "pinned") == 0) return Mode::Pinned;
  if (std::strcmp(s, "serial") == 0) return Mode::Serial;
  if (std::strcmp(s, "overlap") == 0) return Mode::Overlap;
  if (std::strcmp(s, "bidir") == 0) return Mode::Bidir;
  if (std::strcmp(s, "mapped") == 0) return Mode::Mapped;
  std::fprintf(stderr,
               "Invalid --mode=%s "
               "(expected pageable|pinned|serial|overlap|bidir|mapped)\n",
               s);
  std::exit(EXIT_FAILURE);
}

static float mean_of(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  return std::accumulate(v.begin(), v.end(), 0.0f) / float(v.size());
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

static double gbps(size_t bytes, float ms) {
  if (ms <= 0.0f) return 0.0;
  return (double(bytes) / (1024.0 * 1024.0 * 1024.0)) / (double(ms) / 1000.0);
}

static void print_usage(const char* prog) {
  std::printf(
      "Usage:\n"
      "  %s [--mode pageable|pinned|serial|overlap|bidir|mapped] [--mb M]\n"
      "     [--chunk-mb C] [--streams S] [--kernel-iters K]\n"
      "     [--runs R] [--warmup W] [--device D] [--csv-only]\n"
      "\n"
      "Args:\n"
      "  --mode         Transfer strategy (default: pinned)\n"
      "  --mb           Total payload size in MiB (default: 256)\n"
      "  --chunk-mb     Chunk size for serial/overlap in MiB (default: 16)\n"
      "  --streams      Stream count for overlap mode (default: 4)\n"
      "  --kernel-iters Extra compute loops in serial/overlap kernel (default: 8)\n"
      "  --runs         Measured runs (default: 5)\n"
      "  --warmup       Warmup runs (default: 1)\n"
      "  --device       CUDA device id (default: 0)\n"
      "  --csv-only     Print only CSV line\n",
      prog);
}

// serial/overlap：可调计算强度，避免 kernel 过短导致“看不见 overlap”
__global__ void light_touch_kernel(float* __restrict__ data, int n, int iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    float x = data[i];
    for (int r = 0; r < iters; ++r) {
      x = x * 1.000001f + 0.000001f;
    }
    data[i] = x;
  }
}

// Zero-Copy：单遍顺序读 host-mapped 内存
__global__ void mapped_sum_kernel(const float* __restrict__ src,
                                  float* __restrict__ partial,
                                  int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float acc = 0.0f;
  for (int i = tid; i < n; i += stride) {
    acc += src[i];
  }
  // 每线程写自己的 slot，避免 atomic 干扰计时结论
  if (tid < n && tid < 1024) {
    partial[tid] = acc;
  } else if (tid < 1024) {
    partial[tid] = 0.0f;
  }
}

template <class Fn>
static float time_cuda_ms(Fn&& fn, cudaStream_t stream) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s, stream));
  fn(stream);
  CUDA_CHECK(cudaEventRecord(e, stream));
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

// 跨多 stream 的端到端计时：host wall clock + DeviceSynchronize
// （CUDA event 绑在单一 stream 上无法覆盖“多 stream 并发”语义）
template <class Fn>
static float time_wall_ms(Fn&& fn) {
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

static void fill_host(float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    p[i] = float((i * 131) & 1023) * 0.001f;
  }
}

static size_t align_down_floats(size_t bytes) {
  return (bytes / sizeof(float)) * sizeof(float);
}

struct RunConfig {
  Mode mode = Mode::Pinned;
  size_t bytes = 256ull * 1024ull * 1024ull;
  size_t chunk_bytes = 16ull * 1024ull * 1024ull;
  int streams = 4;
  int kernel_iters = 8;
  int runs = 5;
  int warmup = 1;
  int device = 0;
  bool csv_only = false;
};

static void run_chunked_pipeline(float* h,
                                 float* d,
                                 size_t n,
                                 size_t chunk_n,
                                 const std::vector<cudaStream_t>& streams,
                                 int kernel_iters) {
  const int n_streams = (int)streams.size();
  const int num_chunks = (int)((n + chunk_n - 1) / chunk_n);
  const dim3 block(256);

  for (int c = 0; c < num_chunks; ++c) {
    const size_t offset = (size_t)c * chunk_n;
    const size_t this_n = std::min(chunk_n, n - offset);
    const size_t this_bytes = this_n * sizeof(float);
    const int sid = c % n_streams;
    CUDA_CHECK(cudaMemcpyAsync(d + offset, h + offset, this_bytes,
                               cudaMemcpyHostToDevice, streams[(size_t)sid]));
    const dim3 grid((unsigned)std::min(65535, (int)((this_n + 255) / 256)));
    light_touch_kernel<<<grid, block, 0, streams[(size_t)sid]>>>(
        d + offset, (int)this_n, kernel_iters);
    CUDA_CHECK(cudaGetLastError());
  }
}

int main(int argc, char** argv) {
  RunConfig cfg;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      cfg.mode = parse_mode(argv[++i]);
    } else if (std::strcmp(argv[i], "--mb") == 0 && i + 1 < argc) {
      cfg.bytes = (size_t)std::atoll(argv[++i]) * 1024ull * 1024ull;
    } else if (std::strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
      cfg.chunk_bytes = (size_t)std::atoll(argv[++i]) * 1024ull * 1024ull;
    } else if (std::strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
      cfg.streams = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--kernel-iters") == 0 && i + 1 < argc) {
      cfg.kernel_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
      cfg.runs = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg.warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      cfg.device = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--csv-only") == 0) {
      cfg.csv_only = true;
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  cfg.bytes = align_down_floats(std::max(cfg.bytes, sizeof(float)));
  cfg.chunk_bytes = align_down_floats(std::max(cfg.chunk_bytes, sizeof(float)));
  cfg.streams = std::max(1, cfg.streams);
  cfg.kernel_iters = std::max(1, cfg.kernel_iters);
  cfg.runs = std::max(1, cfg.runs);
  cfg.warmup = std::max(0, cfg.warmup);
  if (cfg.chunk_bytes > cfg.bytes) cfg.chunk_bytes = cfg.bytes;

  CUDA_CHECK(cudaSetDevice(cfg.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

  const size_t n = cfg.bytes / sizeof(float);
  std::vector<float> times;
  times.reserve((size_t)cfg.runs);
  size_t bytes_counted = cfg.bytes;
  const char* bw_note = "memcpy_payload";

  if (!cfg.csv_only && prop.asyncEngineCount < 1 &&
      (cfg.mode == Mode::Overlap || cfg.mode == Mode::Bidir)) {
    std::printf("Warning: asyncEngineCount=%d; copy∥compute / bidir may not overlap.\n",
                prop.asyncEngineCount);
  }

  if (cfg.mode == Mode::Pageable || cfg.mode == Mode::Pinned) {
    const bool pinned = (cfg.mode == Mode::Pinned);
    float* h = nullptr;
    float* d = nullptr;
    if (pinned) {
      CUDA_CHECK(cudaMallocHost(&h, cfg.bytes));
    } else {
      h = (float*)std::malloc(cfg.bytes);
      if (!h) {
        std::fprintf(stderr, "malloc failed for %zu bytes\n", cfg.bytes);
        return 1;
      }
    }
    CUDA_CHECK(cudaMalloc(&d, cfg.bytes));
    fill_host(h, n);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto once = [&](cudaStream_t s) {
      CUDA_CHECK(cudaMemcpyAsync(d, h, cfg.bytes, cudaMemcpyHostToDevice, s));
    };

    for (int i = 0; i < cfg.warmup; ++i) (void)time_cuda_ms(once, stream);
    for (int i = 0; i < cfg.runs; ++i) times.push_back(time_cuda_ms(once, stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d));
    if (pinned) {
      CUDA_CHECK(cudaFreeHost(h));
    } else {
      std::free(h);
    }

  } else if (cfg.mode == Mode::Serial || cfg.mode == Mode::Overlap) {
    float* h = nullptr;
    float* d = nullptr;
    CUDA_CHECK(cudaMallocHost(&h, cfg.bytes));
    CUDA_CHECK(cudaMalloc(&d, cfg.bytes));
    fill_host(h, n);

    const int n_streams = (cfg.mode == Mode::Serial) ? 1 : cfg.streams;
    std::vector<cudaStream_t> streams((size_t)n_streams);
    for (int i = 0; i < n_streams; ++i) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[(size_t)i], cudaStreamNonBlocking));
    }

    const size_t chunk_n = cfg.chunk_bytes / sizeof(float);
    auto once = [&]() {
      run_chunked_pipeline(h, d, n, chunk_n, streams, cfg.kernel_iters);
    };

    for (int i = 0; i < cfg.warmup; ++i) (void)time_wall_ms(once);
    for (int i = 0; i < cfg.runs; ++i) times.push_back(time_wall_ms(once));
    bw_note = "h2d_payload_e2e_with_kernel";

    for (int i = 0; i < n_streams; ++i) {
      CUDA_CHECK(cudaStreamDestroy(streams[(size_t)i]));
    }
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFreeHost(h));

  } else if (cfg.mode == Mode::Bidir) {
    float* h_src = nullptr;
    float* h_dst = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_src, cfg.bytes));
    CUDA_CHECK(cudaMallocHost(&h_dst, cfg.bytes));
    CUDA_CHECK(cudaMalloc(&d_a, cfg.bytes));
    CUDA_CHECK(cudaMalloc(&d_b, cfg.bytes));
    fill_host(h_src, n);
    fill_host(h_dst, n);
    CUDA_CHECK(cudaMemcpy(d_b, h_dst, cfg.bytes, cudaMemcpyHostToDevice));

    cudaStream_t s_h2d, s_d2h;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_h2d, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_d2h, cudaStreamNonBlocking));

    auto once = [&]() {
      CUDA_CHECK(cudaMemcpyAsync(d_a, h_src, cfg.bytes, cudaMemcpyHostToDevice, s_h2d));
      CUDA_CHECK(cudaMemcpyAsync(h_dst, d_b, cfg.bytes, cudaMemcpyDeviceToHost, s_d2h));
    };

    for (int i = 0; i < cfg.warmup; ++i) (void)time_wall_ms(once);
    for (int i = 0; i < cfg.runs; ++i) times.push_back(time_wall_ms(once));
    bytes_counted = cfg.bytes * 2;
    bw_note = "h2d_plus_d2h_payload";

    CUDA_CHECK(cudaStreamDestroy(s_h2d));
    CUDA_CHECK(cudaStreamDestroy(s_d2h));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_dst));

  } else {  // mapped
    if (!prop.canMapHostMemory) {
      std::fprintf(stderr, "Device does not support mapped host memory\n");
      return 1;
    }
    float* h = nullptr;
    float* d_partial = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h, cfg.bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_partial, 1024 * sizeof(float)));
    fill_host(h, n);

    float* d_h = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_h, h, 0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const dim3 block(256);
    const dim3 grid((unsigned)std::min(1024, (int)((n + 255) / 256)));

    auto once = [&](cudaStream_t s) {
      mapped_sum_kernel<<<grid, block, 0, s>>>(d_h, d_partial, (int)n);
      CUDA_CHECK(cudaGetLastError());
    };

    for (int i = 0; i < cfg.warmup; ++i) (void)time_cuda_ms(once, stream);
    for (int i = 0; i < cfg.runs; ++i) times.push_back(time_cuda_ms(once, stream));
    bw_note = "mapped_host_read_effective";

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFreeHost(h));
  }

  const float first = times.front();
  const float median = median_of(times);
  const float p95 = percentile_of(times, 95.0f);
  const float mean = mean_of(times);
  const double med_gbps = gbps(bytes_counted, median);

  if (!cfg.csv_only) {
    std::printf("=== [Module B] B-06 Pinned Memory / DMA / Overlap ===\n");
    std::printf("GPU: %s (device=%d)\n", prop.name, cfg.device);
    std::printf("asyncEngineCount=%d, canMapHostMemory=%d\n",
                prop.asyncEngineCount, prop.canMapHostMemory);
    std::printf(
        "mode=%s, bytes=%.2f MiB, chunk=%.2f MiB, streams=%d, "
        "kernel_iters=%d, warmup=%d, runs=%d\n",
        mode_name(cfg.mode),
        cfg.bytes / (1024.0 * 1024.0),
        cfg.chunk_bytes / (1024.0 * 1024.0),
        (cfg.mode == Mode::Serial) ? 1 : cfg.streams,
        cfg.kernel_iters,
        cfg.warmup,
        cfg.runs);
    std::printf("Result: first=%.4f ms, median=%.4f ms, p95=%.4f ms, mean=%.4f ms\n",
                first, median, p95, mean);
    std::printf("Throughput (median): %.2f GB/s  [%s] (bytes_counted=%.2f MiB)\n",
                med_gbps, bw_note, bytes_counted / (1024.0 * 1024.0));
    if (cfg.mode == Mode::Overlap) {
      std::printf(
          "Tip: compare with --mode serial (same mb/chunk/kernel-iters). "
          "Confirm Copy∩Kernel in NSYS; WDDM may hide overlap.\n");
    }
    if (cfg.mode == Mode::Mapped) {
      std::printf(
          "Note: mapped GB/s is effective host-read bandwidth during kernel, "
          "not cudaMemcpy H2D bandwidth.\n");
    }
  }

  std::printf(
      "CSV,mode=%s,bytes=%zu,chunk_bytes=%zu,streams=%d,kernel_iters=%d,"
      "warmup=%d,runs=%d,first_ms=%.4f,median_ms=%.4f,p95_ms=%.4f,mean_ms=%.4f,"
      "median_GBps=%.3f,asyncEngineCount=%d,bw_note=%s\n",
      mode_name(cfg.mode),
      cfg.bytes,
      cfg.chunk_bytes,
      (cfg.mode == Mode::Serial) ? 1 : cfg.streams,
      cfg.kernel_iters,
      cfg.warmup,
      cfg.runs,
      first,
      median,
      p95,
      mean,
      med_gbps,
      prop.asyncEngineCount,
      bw_note);

  return 0;
}
