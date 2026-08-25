/**
 * [Module B] B-04. L2：Persisting / Access Policy / hitRatio
 *
 *   streaming : 扫过大数组，policy off
 *   hot       : 小工作集反复读，policy off
 *   mixed     : 热/冷混合，policy off（persist 的公平基线）
 *   persist   : 同一 mixed + set-aside + window + hitRatio
 *   thrash    : 同一窗，hitRatio=1.0
 *   modes     : 五档一次跑齐 + 相对 mixed 加速比 CSV
 *
 * 主证据：CUDA event warmup + 多次 run → median
 * 刻意不做：TMA / cp.async / UM / bank（→ B-08 / B-07 / B-05 / B-02）
 * 硬件：sm_80+ 且 persistingL2CacheMaxSize > 0
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",           \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

enum class Mode { Streaming = 0, Hot = 1, Mixed = 2, Persist = 3, Thrash = 4, Modes = 5 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Streaming: return "streaming";
    case Mode::Hot: return "hot";
    case Mode::Mixed: return "mixed";
    case Mode::Persist: return "persist";
    case Mode::Thrash: return "thrash";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "streaming") == 0) return Mode::Streaming;
  if (std::strcmp(s, "hot") == 0) return Mode::Hot;
  if (std::strcmp(s, "mixed") == 0) return Mode::Mixed;
  if (std::strcmp(s, "persist") == 0) return Mode::Persist;
  if (std::strcmp(s, "thrash") == 0) return Mode::Thrash;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected streaming|hot|mixed|persist|thrash|modes)\n",
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

static float mean_of(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  return std::accumulate(v.begin(), v.end(), 0.0f) / float(v.size());
}

static size_t mb_to_bytes(double mb) {
  if (mb <= 0.0) return 0;
  return (size_t)(mb * 1024.0 * 1024.0);
}

static int next_pow2(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

__global__ void streaming_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 int n, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (int r = 0; r < iters; ++r) {
    int i = tid;
    for (; i + 3 * stride < n; i += 4 * stride) {
      acc0 = fmaf(in[i], 1.000001f, acc0);
      acc1 = fmaf(in[i + stride], 1.000001f, acc1);
      acc2 = fmaf(in[i + 2 * stride], 1.000001f, acc2);
      acc3 = fmaf(in[i + 3 * stride], 1.000001f, acc3);
    }
    for (; i < n; i += stride) acc0 = fmaf(in[i], 1.000001f, acc0);
  }
  if (tid < n) out[tid] = (acc0 + acc1) + (acc2 + acc3);
}

__global__ void hot_reuse_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 int n, int iters, int hot_mask) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  const int base = (int)((unsigned)tid * 1315423911u) & hot_mask;
  float acc = 0.0f;
#pragma unroll 4
  for (int r = 0; r < iters; ++r) {
    const int idx = (base + (r * 17)) & hot_mask;
    acc = fmaf(in[idx], 1.000003f, acc);
  }
  out[tid] = acc;
}

__global__ void mixed_window_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int n, int iters, int window_n, int hot_n,
                                    int cold_stride, unsigned seed_base) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  unsigned seed = ((unsigned)tid ^ seed_base) * 1664525u + 1013904223u;
  float acc = 0.0f;
  const int hn = (hot_n > 0) ? hot_n : 1;
  const int wn = (window_n > 0) ? window_n : 1;
  for (int r = 0; r < iters; ++r) {
    seed = seed * 1664525u + 1013904223u;
    int idx = 0;
    if ((seed & 3u) != 0u) {
      idx = (int)(seed % (unsigned)hn);
    } else {
      const int j = (int)((seed + (unsigned)r * 97u) % (unsigned)wn);
      idx = (j * cold_stride) % wn;
    }
    acc = fmaf(in[idx], 1.000007f, acc);
  }
  out[tid] = acc;
}

static void set_persisting_l2_bytes(size_t bytes, size_t max_bytes) {
  if (bytes > max_bytes) bytes = max_bytes;
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes));
}

static void set_access_policy_window(cudaStream_t stream, const void* base_ptr,
                                     size_t num_bytes, float hit_ratio) {
  cudaAccessPolicyWindow w{};
  w.base_ptr = const_cast<void*>(base_ptr);
  w.num_bytes = num_bytes;
  w.hitRatio = hit_ratio;
  w.hitProp = cudaAccessPropertyPersisting;
  w.missProp = cudaAccessPropertyStreaming;
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = w;
  CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void disable_access_policy_window(cudaStream_t stream) {
  cudaAccessPolicyWindow w{};
  w.base_ptr = nullptr;
  w.num_bytes = 0;
  w.hitRatio = 0.0f;
  w.hitProp = cudaAccessPropertyNormal;
  w.missProp = cudaAccessPropertyNormal;
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = w;
  CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void reset_persisting_l2() { CUDA_CHECK(cudaCtxResetPersistingL2Cache()); }

struct BenchConfig {
  Mode mode = Mode::Modes;
  double data_mb = 64.0;
  int iters = 2048;
  double set_aside_mb = 8.0;
  double window_mb = 32.0;
  float hit_ratio = 0.25f;
  float hot_ratio = 0.25f;
  int runs = 7;
  int warmup = 2;
  unsigned seed = 12345u;
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
    } else if (std::strcmp(argv[i], "--data-mb") == 0) {
      c.data_mb = std::atof(need("--data-mb"));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      c.iters = std::atoi(need("--iters"));
    } else if (std::strcmp(argv[i], "--set-aside-mb") == 0) {
      c.set_aside_mb = std::atof(need("--set-aside-mb"));
    } else if (std::strcmp(argv[i], "--window-mb") == 0) {
      c.window_mb = std::atof(need("--window-mb"));
    } else if (std::strcmp(argv[i], "--hit-ratio") == 0) {
      c.hit_ratio = (float)std::atof(need("--hit-ratio"));
    } else if (std::strcmp(argv[i], "--hot-ratio") == 0) {
      c.hot_ratio = (float)std::atof(need("--hot-ratio"));
    } else if (std::strcmp(argv[i], "--runs") == 0) {
      c.runs = std::atoi(need("--runs"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      c.warmup = std::atoi(need("--warmup"));
    } else if (std::strcmp(argv[i], "--seed") == 0) {
      c.seed = (unsigned)std::strtoul(need("--seed"), nullptr, 10);
    } else if (std::strcmp(argv[i], "--device") == 0) {
      c.device = std::atoi(need("--device"));
    } else if (std::strcmp(argv[i], "--csv-only") == 0) {
      c.csv_only = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf(
          "Usage: %s --mode <streaming|hot|mixed|persist|thrash|modes> [options]\n"
          "  --data-mb --iters --set-aside-mb --window-mb --hit-ratio --hot-ratio\n"
          "  --runs --warmup --seed --device --csv-only\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  c.hit_ratio = std::min(1.0f, std::max(0.0f, c.hit_ratio));
  c.hot_ratio = std::min(1.0f, std::max(0.0f, c.hot_ratio));
  if (c.data_mb <= 0 || c.iters <= 0 || c.runs <= 0 || c.warmup < 0) {
    std::fprintf(stderr, "Invalid numeric args\n");
    std::exit(EXIT_FAILURE);
  }
  return c;
}

template <typename Launch>
static float time_launch_ms(Launch&& launch, cudaStream_t stream, int warmup, int runs,
                            std::vector<float>* samples) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  samples->clear();
  samples->reserve(runs);
  for (int i = 0; i < runs; ++i) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples->push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return median_of(*samples);
}

struct Workload {
  float* d_in = nullptr;
  float* d_out = nullptr;
  int n = 0;
  int iters = 0;
  int grid = 0;
  int block = 256;
  int hot_mask = 0;
  int window_n = 0;
  int hot_subset_n = 0;
  size_t set_aside_bytes = 0;
  size_t persist_max_bytes = 0;
  size_t window_bytes = 0;
  float hit_ratio = 0.25f;
  unsigned seed = 12345u;
  cudaStream_t stream = nullptr;
};

static void apply_policy(const Workload& w, bool on, float hit) {
  if (!on) {
    disable_access_policy_window(w.stream);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
    reset_persisting_l2();
    return;
  }
  set_persisting_l2_bytes(w.set_aside_bytes, w.persist_max_bytes);
  set_access_policy_window(w.stream, w.d_in, w.window_bytes, hit);
  reset_persisting_l2();
}

static float run_mode(Mode mode, const Workload& w, std::vector<float>* samples,
                      int warmup, int runs) {
  const bool persist_on = (mode == Mode::Persist || mode == Mode::Thrash);
  const float hit = (mode == Mode::Thrash) ? 1.0f : w.hit_ratio;
  apply_policy(w, persist_on, hit);
  const int cold = 17;

  auto launch = [&]() {
    switch (mode) {
      case Mode::Streaming:
        streaming_kernel<<<w.grid, w.block, 0, w.stream>>>(w.d_in, w.d_out, w.n, w.iters);
        break;
      case Mode::Hot:
        hot_reuse_kernel<<<w.grid, w.block, 0, w.stream>>>(w.d_in, w.d_out, w.n, w.iters,
                                                           w.hot_mask);
        break;
      case Mode::Mixed:
      case Mode::Persist:
      case Mode::Thrash:
        mixed_window_kernel<<<w.grid, w.block, 0, w.stream>>>(
            w.d_in, w.d_out, w.n, w.iters, w.window_n, w.hot_subset_n, cold, w.seed);
        break;
      default:
        break;
    }
  };
  return time_launch_ms(launch, w.stream, warmup, runs, samples);
}

static void print_row(bool csv_only, Mode mode, float med, const std::vector<float>& samples,
                      float mixed_ms) {
  const float first = samples.empty() ? 0.f : samples.front();
  const float p95 = percentile_of(samples, 95.f);
  const float mean = mean_of(samples);
  const float speedup = (med > 0.f && mixed_ms > 0.f) ? (mixed_ms / med) : 0.f;
  if (csv_only) {
    std::printf("%s,%.6f,%.4f\n", mode_name(mode), med, speedup);
    return;
  }
  std::printf(
      "mode=%-10s  first=%.4f  median=%.4f  p95=%.4f  mean=%.4f ms  vs_mixed=%.3fx\n",
      mode_name(mode), first, med, p95, mean, speedup);
}

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  if (prop.major < 8 || prop.persistingL2CacheMaxSize == 0) {
    std::fprintf(stderr,
                 "B-04 needs sm_80+ persisting L2 (got sm_%d%d, persistingL2CacheMaxSize=%zu)\n",
                 prop.major, prop.minor, (size_t)prop.persistingL2CacheMaxSize);
    return 2;
  }

  const size_t bytes = mb_to_bytes(cfg.data_mb);
  const int n = (int)(bytes / sizeof(float));
  if (n <= 0) {
    std::fprintf(stderr, "Invalid --data-mb\n");
    return 1;
  }

  size_t set_aside_bytes =
      std::min(mb_to_bytes(cfg.set_aside_mb), (size_t)prop.persistingL2CacheMaxSize);
  size_t window_bytes = std::min(mb_to_bytes(cfg.window_mb), bytes);
  window_bytes = std::min(window_bytes, (size_t)prop.accessPolicyMaxWindowSize);

  int hot_n = (int)(std::min(cfg.data_mb, prop.l2CacheSize / (1024.0 * 1024.0)) * 1024.0 *
                    1024.0 / sizeof(float));
  hot_n = std::max(4096, std::min(hot_n, n));
  int hot_pow2 = next_pow2(hot_n);
  while (hot_pow2 > n) hot_pow2 >>= 1;
  const int hot_mask = hot_pow2 - 1;
  const int window_n = std::max(1, (int)(window_bytes / sizeof(float)));
  const int hot_subset_n = std::max(1, (int)((double)window_n * (double)cfg.hot_ratio));

  if (!cfg.csv_only) {
    std::printf("GPU: %s | sm_%d%d\n", prop.name, prop.major, prop.minor);
    std::printf("L2=%.1f MB  persistingMax=%.1f MB  policyWindowMax=%.1f MB\n",
                prop.l2CacheSize / (1024.0 * 1024.0),
                prop.persistingL2CacheMaxSize / (1024.0 * 1024.0),
                prop.accessPolicyMaxWindowSize / (1024.0 * 1024.0));
    std::printf("data_mb=%.2f n=%d iters=%d | set-aside=%.2f MB window=%.2f MB hit=%.3f hot=%.3f\n",
                cfg.data_mb, n, cfg.iters, set_aside_bytes / (1024.0 * 1024.0),
                window_bytes / (1024.0 * 1024.0), cfg.hit_ratio, cfg.hot_ratio);
    std::printf("runs=%d warmup=%d seed=%u | speedup = mixed_median / mode_median\n",
                cfg.runs, cfg.warmup, cfg.seed);
  }

  Workload w;
  w.n = n;
  w.iters = cfg.iters;
  w.block = 256;
  w.grid = std::min(65535, (n + w.block - 1) / w.block);
  w.hot_mask = hot_mask;
  w.window_n = window_n;
  w.hot_subset_n = hot_subset_n;
  w.set_aside_bytes = set_aside_bytes;
  w.persist_max_bytes = (size_t)prop.persistingL2CacheMaxSize;
  w.window_bytes = window_bytes;
  w.hit_ratio = cfg.hit_ratio;
  w.seed = cfg.seed;
  CUDA_CHECK(cudaMalloc(&w.d_in, bytes));
  CUDA_CHECK(cudaMalloc(&w.d_out, bytes));
  std::vector<float> h_in(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) h_in[static_cast<size_t>(i)] = float((i * 131) % 1024) * 0.001f;
  CUDA_CHECK(cudaMemcpy(w.d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(w.d_out, 0, bytes));
  CUDA_CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));

  auto emit_csv = [&](Mode m, float med, float mixed_ms) {
    const float sp = (med > 0.f && mixed_ms > 0.f) ? (mixed_ms / med) : 0.f;
    std::printf("%s,%.6f,%.4f\n", mode_name(m), med, (m == Mode::Mixed) ? 1.0f : sp);
  };

  if (cfg.mode == Mode::Modes) {
    std::vector<float> s0, s1, s2, s3, s4;
    const float t0 = run_mode(Mode::Streaming, w, &s0, cfg.warmup, cfg.runs);
    const float t1 = run_mode(Mode::Hot, w, &s1, cfg.warmup, cfg.runs);
    const float t2 = run_mode(Mode::Mixed, w, &s2, cfg.warmup, cfg.runs);
    const float t3 = run_mode(Mode::Persist, w, &s3, cfg.warmup, cfg.runs);
    const float t4 = run_mode(Mode::Thrash, w, &s4, cfg.warmup, cfg.runs);
    if (!cfg.csv_only) {
      std::printf("\n=== modes (speedup = mixed_median / mode_median) ===\n");
    }
    print_row(cfg.csv_only, Mode::Streaming, t0, s0, t2);
    print_row(cfg.csv_only, Mode::Hot, t1, s1, t2);
    print_row(cfg.csv_only, Mode::Mixed, t2, s2, t2);
    print_row(cfg.csv_only, Mode::Persist, t3, s3, t2);
    print_row(cfg.csv_only, Mode::Thrash, t4, s4, t2);
    std::printf("\nmode,median_ms,speedup_vs_mixed\n");
    emit_csv(Mode::Streaming, t0, t2);
    emit_csv(Mode::Hot, t1, t2);
    emit_csv(Mode::Mixed, t2, t2);
    emit_csv(Mode::Persist, t3, t2);
    emit_csv(Mode::Thrash, t4, t2);
  } else if (cfg.mode == Mode::Mixed) {
    std::vector<float> s;
    const float med = run_mode(Mode::Mixed, w, &s, cfg.warmup, cfg.runs);
    print_row(cfg.csv_only, Mode::Mixed, med, s, med);
  } else {
    std::vector<float> s_ref, s;
    const float mixed_ms = run_mode(Mode::Mixed, w, &s_ref, cfg.warmup, cfg.runs);
    if (!cfg.csv_only) std::printf("(mixed reference median=%.4f ms)\n", mixed_ms);
    const float med = run_mode(cfg.mode, w, &s, cfg.warmup, cfg.runs);
    print_row(cfg.csv_only, cfg.mode, med, s, mixed_ms);
  }

  float probe = 0.f;
  CUDA_CHECK(cudaMemcpy(&probe, w.d_out, sizeof(float), cudaMemcpyDeviceToHost));
  if (!cfg.csv_only) std::printf("probe_out0=%.3f\n", probe);

  apply_policy(w, false, 0.f);
  CUDA_CHECK(cudaStreamDestroy(w.stream));
  CUDA_CHECK(cudaFree(w.d_in));
  CUDA_CHECK(cudaFree(w.d_out));
  return 0;
}
