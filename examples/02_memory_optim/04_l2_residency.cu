/**
 * [Module B] B-04. L2 Cache 行为与 Residency：Access Policy / Persisting vs Streaming
 *
 * 目标（工程闭环）：
 * - 用最小 micro-bench 复现三类典型场景：
 *   A) Streaming：一次性扫过大数组（policy 基本无收益）
 *   B) Hot Reuse：反复访问同一工作集（工作集越贴近 L2，越容易快）
 *   C) Residency：set-aside + access policy window，让“热点子集”更稳定驻留
 *   D) Thrashing：window >> set-aside 且 hitRatio 过高，出现抖动/不升反降
 *
 * 输出（证据三件套的可复现骨架）：
 * - time：CUDA Event 计时（每次 kernel 的平均 ms）
 * - DRAM/L2：建议用 Nsight Compute（见文末提示）
 *
 * 兼容性：
 * - 代码基于 CUDA 12/13 共同子集（cudaLimitPersistingL2CacheSize + AccessPolicyWindow）
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                   cudaGetErrorString(err), (int)err, __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE); \
    } \
  } while (0)

static inline size_t mb_to_bytes(double mb) {
  if (mb <= 0.0) return 0;
  return (size_t)(mb * 1024.0 * 1024.0);
}

static void set_persisting_l2_bytes(size_t bytes) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  size_t max_bytes = (size_t)prop.persistingL2CacheMaxSize;
  if (bytes > max_bytes) bytes = max_bytes;
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes));
}

static void set_access_policy_window(cudaStream_t stream,
                                     const void* base_ptr,
                                     size_t num_bytes,
                                     float hit_ratio) {
  cudaAccessPolicyWindow w{};
  w.base_ptr  = const_cast<void*>(base_ptr);
  w.num_bytes = num_bytes;
  w.hitRatio  = hit_ratio;
  w.hitProp   = cudaAccessPropertyPersisting;
  w.missProp  = cudaAccessPropertyStreaming;

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = w;
  CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void disable_access_policy_window(cudaStream_t stream) {
  cudaAccessPolicyWindow w{};
  w.base_ptr  = nullptr;
  w.num_bytes = 0;
  w.hitRatio  = 0.0f;
  w.hitProp   = cudaAccessPropertyNormal;
  w.missProp  = cudaAccessPropertyNormal;
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = w;
  CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void reset_persisting_l2() {
  CUDA_CHECK(cudaCtxResetPersistingL2Cache());
}

// ---------------- Kernels ----------------

// Streaming：grid-stride 扫过整个数组，做一点点 FMA 防止优化掉
__global__ void streaming_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n,
                                 int iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // 用多条独立累加链打断单链 RAW 依赖，提升 ILP，减少 scoreboard 等待。
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int r = 0; r < iters; ++r) {
    int i = tid;

    // 手工展开 4x：每轮消费 4 个 stride 位置，增加并行可发射机会。
    for (; i + 3 * stride < n; i += 4 * stride) {
      float x0 = in[i];
      float x1 = in[i + stride];
      float x2 = in[i + 2 * stride];
      float x3 = in[i + 3 * stride];

      acc0 = fmaf(x0, 1.000001f, acc0);
      acc1 = fmaf(x1, 1.000001f, acc1);
      acc2 = fmaf(x2, 1.000001f, acc2);
      acc3 = fmaf(x3, 1.000001f, acc3);
    }

    // 处理尾项
    for (; i < n; i += stride) {
      float x = in[i];
      acc0 = fmaf(x, 1.000001f, acc0);
    }
  }

  float acc = (acc0 + acc1) + (acc2 + acc3);
  if (tid < n) out[tid] = acc;
}

// Hot reuse：每个线程围绕一个小集合反复读（更容易打出 L2 hit）
__global__ void hot_reuse_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n,
                                 int iters,
                                 int hot_mask) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  // 每线程选择一个基址，限制在 hot set 内
  int base = (tid * 1315423911u) & hot_mask;
  float acc = 0.0f;
  #pragma unroll 4
  for (int r = 0; r < iters; ++r) {
    int idx = (base + (r * 17)) & hot_mask;
    float x = in[idx];
    acc = fmaf(x, 1.000003f, acc);
  }
  out[tid] = acc;
}

// Mixed（用于 residency / thrashing）：访问 window 内的“热点子集” + “冷数据”
// - 热点访问更频繁
// - 冷数据访问用于制造污染压力（window 大时更明显）
__global__ void mixed_window_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int n,
                                   int iters,
                                   int window_n,
                                   int hot_n,
                                   int cold_stride,
                                   unsigned seed_base) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  unsigned seed = ((unsigned)tid ^ seed_base) * 1664525u + 1013904223u;
  float acc = 0.0f;

  // 约 3:1 热/冷访问比例（不追求精确，追求“复用 + 压力”共存）
  for (int r = 0; r < iters; ++r) {
    seed = seed * 1664525u + 1013904223u;
    bool do_hot = (seed & 3u) != 0u;

    int idx = 0;
    if (do_hot) {
      // 热点：限制在 [0, hot_n)
      int hn = (hot_n > 0) ? hot_n : 1;
      int j = (int)(seed % (unsigned)hn);
      idx = j;
    } else {
      // 冷数据：在 window 范围内跨步走，制造替换压力
      int wn = (window_n > 0) ? window_n : 1;
      int j = (int)((seed + (unsigned)r * 97u) % (unsigned)wn);
      idx = (j * cold_stride) % wn;
    }

    float x = in[idx];
    acc = fmaf(x, 1.000007f, acc);
  }

  out[tid] = acc;
}

template <class LaunchFn>
static float time_kernel(LaunchFn launch, int warmup, int repeats, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) launch(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  CUDA_CHECK(cudaEventRecord(s, stream));
  for (int i = 0; i < repeats; ++i) launch(stream);
  CUDA_CHECK(cudaEventRecord(e, stream));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));

  return ms / float(repeats);
}

static int next_pow2(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

static float mean_of(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  float sum = std::accumulate(v.begin(), v.end(), 0.0f);
  return sum / float(v.size());
}

static float median_of(std::vector<float> v) {
  if (v.empty()) return 0.0f;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  if (n % 2 == 1) return v[n / 2];
  return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static void print_usage(const char* prog) {
  std::printf(
      "Usage:\n"
      "  %s [data_mb iters set_aside_mb window_mb hit_ratio]\n"
      "  %s [--data-mb M] [--iters N] [--set-aside-mb M] [--window-mb M] [--hit-ratio R] [--hot-ratio R] [--runs N] [--seed U] [--csv-only]\n"
      "\n"
      "Args:\n"
      "  data_mb / --data-mb             Total input size in MB (default: 64)\n"
      "  iters / --iters                 Loop count per thread (default: 2048)\n"
      "  set_aside_mb / --set-aside-mb   L2 persisting set-aside size in MB (default: 8)\n"
      "  window_mb / --window-mb         Access policy window size in MB (default: 32)\n"
      "  hit_ratio / --hit-ratio         Access policy hitRatio in [0,1] (default: 0.25)\n"
      "  --hot-ratio                     Hot subset ratio in [0,1] for mixed workload (default: 0.25)\n"
      "  --runs                          Number of repeated runs for robust stats (default: 1)\n"
      "  --seed                          RNG seed for mixed workloads (default: 12345)\n"
      "  --csv-only                      Print CSV line only (for batch runs)\n",
      prog, prog);
}

int main(int argc, char** argv) {
  // 参数（尽量给出能“默认跑通”的配置）
  // data_mb: 工作集大小（MB）
  // iters:   内层循环次数（放大差异）
  // set_aside_mb / window_mb / hit_ratio: residency 参数
  double data_mb = 64.0;
  int iters = 2048;
  double set_aside_mb = 8.0;
  double window_mb = 32.0;
  float hit_ratio = 0.25f;
  float hot_ratio = 0.25f;
  int warmup = 3;
  int repeats = 20;
  int runs = 1;
  unsigned seed_base = 12345u;
  bool csv_only = false;

  bool has_named_args = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
    if (std::strncmp(argv[i], "--", 2) == 0) {
      has_named_args = true;
      break;
    }
  }

  if (has_named_args) {
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--data-mb") == 0 && i + 1 < argc) {
        data_mb = std::atof(argv[++i]);
      } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
        iters = std::atoi(argv[++i]);
      } else if (std::strcmp(argv[i], "--set-aside-mb") == 0 && i + 1 < argc) {
        set_aside_mb = std::atof(argv[++i]);
      } else if (std::strcmp(argv[i], "--window-mb") == 0 && i + 1 < argc) {
        window_mb = std::atof(argv[++i]);
      } else if (std::strcmp(argv[i], "--hit-ratio") == 0 && i + 1 < argc) {
        hit_ratio = (float)std::atof(argv[++i]);
      } else if (std::strcmp(argv[i], "--hot-ratio") == 0 && i + 1 < argc) {
        hot_ratio = (float)std::atof(argv[++i]);
      } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
        runs = std::atoi(argv[++i]);
      } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
        seed_base = (unsigned)std::strtoul(argv[++i], nullptr, 10);
      } else if (std::strcmp(argv[i], "--csv-only") == 0) {
        csv_only = true;
      } else {
        std::fprintf(stderr, "Unknown or incomplete argument: %s\n\n", argv[i]);
        print_usage(argv[0]);
        return 1;
      }
    }
  } else {
    if (argc >= 2) data_mb = std::atof(argv[1]);
    if (argc >= 3) iters = std::atoi(argv[2]);
    if (argc >= 4) set_aside_mb = std::atof(argv[3]);
    if (argc >= 5) window_mb = std::atof(argv[4]);
    if (argc >= 6) hit_ratio = (float)std::atof(argv[5]);
  }

  hit_ratio = std::min(1.0f, std::max(0.0f, hit_ratio));
  hot_ratio = std::min(1.0f, std::max(0.0f, hot_ratio));
  runs = std::max(1, runs);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const size_t bytes = mb_to_bytes(data_mb);
  const int n = (int)(bytes / sizeof(float));
  if (n <= 0) {
    std::fprintf(stderr, "Invalid data_mb=%.3f\n", data_mb);
    return 1;
  }

  if (!csv_only) {
    std::printf("=== [Module B] B-04 L2 Residency Control (CUDA 13 / compat 12) ===\n");
    std::printf("GPU: %s\n", prop.name);
    std::printf("data_mb=%.2f (n=%d floats), iters=%d\n", data_mb, n, iters);
    std::printf("set_aside_mb=%.2f, window_mb=%.2f, hit_ratio=%.3f, hot_ratio=%.3f, runs=%d, seed=%u\n",
                set_aside_mb, window_mb, hit_ratio, hot_ratio, runs, seed_base);
    std::printf("device limits: persistingL2CacheMaxSize=%.2f MB, accessPolicyMaxWindowSize=%.2f MB\n",
                prop.persistingL2CacheMaxSize / (1024.0 * 1024.0),
                prop.accessPolicyMaxWindowSize / (1024.0 * 1024.0));
    std::printf("Tip: use NCU to check DRAM bytes + L2 hit. Example in article B-04.\n\n");
  }

  std::vector<float> h_in(n);
  for (int i = 0; i < n; ++i) h_in[i] = float((i * 131) % 1024) * 0.001f;

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  dim3 block(256);
  dim3 grid(std::min(65535, (n + (int)block.x - 1) / (int)block.x));

  // 1) Hot reuse 需要的参数准备
  // hot set: 让它等于 min(data, L2*1.0) 的 2^k，方便 mask
  int hot_n = (int)(std::min(data_mb, prop.l2CacheSize / (1024.0 * 1024.0)) * 1024.0 * 1024.0 / sizeof(float));
  hot_n = std::max(4096, hot_n);
  hot_n = std::min(hot_n, n);
  int hot_pow2 = next_pow2(hot_n);
  int hot_mask = hot_pow2 - 1;
  // 保证 mask 不越界：如果 pow2 > n，则限制在 n 的下一档
  while (hot_pow2 > n) {
    hot_pow2 >>= 1;
    hot_mask = hot_pow2 - 1;
  }

  // 2) Mixed baseline / Residency：同一 workload，分别 policy off / on
  // window 默认限定在 data 范围内；hot_n 取 window 的子集用于“热点访问更频繁”
  size_t set_aside_bytes = mb_to_bytes(set_aside_mb);
  set_aside_bytes = std::min(set_aside_bytes, (size_t)prop.persistingL2CacheMaxSize);
  double effective_set_aside_mb = (double)set_aside_bytes / (1024.0 * 1024.0);
  size_t window_bytes = std::min(mb_to_bytes(window_mb), bytes);
  window_bytes = std::min(window_bytes, (size_t)prop.accessPolicyMaxWindowSize);
  double effective_window_mb = (double)window_bytes / (1024.0 * 1024.0);
  int window_n = (int)(window_bytes / sizeof(float));
  window_n = std::max(1, window_n);
  int hot_subset_n = (int)std::max(1.0, (double)window_n * (double)hot_ratio);

  std::vector<float> stream_runs, hot_runs, c0_runs, c1_runs, d_runs;
  stream_runs.reserve((size_t)runs);
  hot_runs.reserve((size_t)runs);
  c0_runs.reserve((size_t)runs);
  c1_runs.reserve((size_t)runs);
  d_runs.reserve((size_t)runs);

  for (int run = 0; run < runs; ++run) {
    auto launch_mixed_baseline = [&](cudaStream_t s) {
      mixed_window_kernel<<<grid, block, 0, s>>>(d_in, d_out, n, iters, window_n, hot_subset_n, /*cold_stride*/ 17, seed_base + (unsigned)run);
      CUDA_CHECK(cudaGetLastError());
    };
    auto launch_residency = [&](cudaStream_t s) {
      mixed_window_kernel<<<grid, block, 0, s>>>(d_in, d_out, n, iters, window_n, hot_subset_n, /*cold_stride*/ 17, seed_base + (unsigned)run);
      CUDA_CHECK(cudaGetLastError());
    };
    auto launch_thrash = [&](cudaStream_t s) {
      mixed_window_kernel<<<grid, block, 0, s>>>(d_in, d_out, n, iters, window_n, hot_subset_n, /*cold_stride*/ 97, seed_base + (unsigned)run);
      CUDA_CHECK(cudaGetLastError());
    };

    // 1) Streaming baseline（policy off）
    disable_access_policy_window(stream);
    reset_persisting_l2();
    auto launch_stream = [&](cudaStream_t s) {
      streaming_kernel<<<grid, block, 0, s>>>(d_in, d_out, n, iters);
      CUDA_CHECK(cudaGetLastError());
    };
    stream_runs.push_back(time_kernel([&](cudaStream_t s) { launch_stream(s); }, warmup, repeats, stream));

    // 2) Hot reuse baseline（policy off）
    disable_access_policy_window(stream);
    reset_persisting_l2();
    auto launch_hot = [&](cudaStream_t s) {
      hot_reuse_kernel<<<grid, block, 0, s>>>(d_in, d_out, n, iters, hot_mask);
      CUDA_CHECK(cudaGetLastError());
    };
    hot_runs.push_back(time_kernel([&](cudaStream_t s) { launch_hot(s); }, warmup, repeats, stream));

    // 3) Mixed baseline（policy off）
    disable_access_policy_window(stream);
    reset_persisting_l2();
    c0_runs.push_back(time_kernel([&](cudaStream_t s) { launch_mixed_baseline(s); }, warmup, repeats, stream));

    // 4) Residency（policy on）
    set_persisting_l2_bytes(set_aside_bytes);
    set_access_policy_window(stream, d_in, window_bytes, hit_ratio);
    reset_persisting_l2();
    c1_runs.push_back(time_kernel([&](cudaStream_t s) { launch_residency(s); }, warmup, repeats, stream));

    // 5) Thrashing（policy on）
    set_persisting_l2_bytes(set_aside_bytes);
    set_access_policy_window(stream, d_in, window_bytes, 1.0f);
    reset_persisting_l2();
    d_runs.push_back(time_kernel([&](cudaStream_t s) { launch_thrash(s); }, warmup, repeats, stream));
  }

  const float ms_stream = median_of(stream_runs);
  const float ms_hot = median_of(hot_runs);
  const float ms_mixed_base = median_of(c0_runs);
  const float ms_res = median_of(c1_runs);
  const float ms_thr = median_of(d_runs);
  const float mean_stream = mean_of(stream_runs);
  const float mean_hot = mean_of(hot_runs);
  const float mean_c0 = mean_of(c0_runs);
  const float mean_c1 = mean_of(c1_runs);
  const float mean_d = mean_of(d_runs);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  if (csv_only) {
    std::printf("CSV,time_ms,stream_med=%.4f,hot_med=%.4f,mixed_base_med=%.4f,residency_med=%.4f,thrashing_med=%.4f,stream_mean=%.4f,hot_mean=%.4f,mixed_base_mean=%.4f,residency_mean=%.4f,thrashing_mean=%.4f,data_mb=%.2f,iters=%d,set_aside_mb=%.2f,window_mb=%.2f,hit_ratio=%.3f,hot_ratio=%.3f,runs=%d,seed=%u\n",
                ms_stream, ms_hot, ms_mixed_base, ms_res, ms_thr,
                mean_stream, mean_hot, mean_c0, mean_c1, mean_d,
                data_mb, iters, effective_set_aside_mb, effective_window_mb, hit_ratio, hot_ratio, runs, seed_base);
  } else {
    std::printf("[A] Streaming (policy off)  : median=%.4f ms, mean=%.4f ms\n", ms_stream, mean_stream);
    std::printf("[B] Hot reuse (policy off)  : median=%.4f ms, mean=%.4f ms\n", ms_hot, mean_hot);
    std::printf("[C0] Mixed baseline (off)   : %.4f ms  (window=%.2fMB, hotRatio=%.3f)\n",
                ms_mixed_base, effective_window_mb, hot_ratio);
    std::printf("[C1] Mixed + residency (on) : %.4f ms  (set-aside=%.2fMB, window=%.2fMB, hitRatio=%.3f)\n",
                ms_res, effective_set_aside_mb, effective_window_mb, hit_ratio);
    std::printf("[D] Thrashing (policy on)   : %.4f ms  (hitRatio forced 1.0)\n", ms_thr);
    std::printf("  (mean) C0=%.4f ms, C1=%.4f ms, D=%.4f ms, runs=%d\n", mean_c0, mean_c1, mean_d, runs);
    std::printf("CSV,time_ms,stream_med=%.4f,hot_med=%.4f,mixed_base_med=%.4f,residency_med=%.4f,thrashing_med=%.4f,stream_mean=%.4f,hot_mean=%.4f,mixed_base_mean=%.4f,residency_mean=%.4f,thrashing_mean=%.4f,data_mb=%.2f,iters=%d,set_aside_mb=%.2f,window_mb=%.2f,hit_ratio=%.3f,hot_ratio=%.3f,runs=%d,seed=%u\n",
                ms_stream, ms_hot, ms_mixed_base, ms_res, ms_thr,
                mean_stream, mean_hot, mean_c0, mean_c1, mean_d,
                data_mb, iters, effective_set_aside_mb, effective_window_mb, hit_ratio, hot_ratio, runs, seed_base);
    std::printf("\nInterpretation:\n");
    std::printf("- [A] 是纯 streaming：通常 DRAM bytes 高，L2 hit 低；policy 很难带来稳定收益。\n");
    std::printf("- [B] 有复用：如果工作集贴近/小于 L2，L2 hit 会明显抬升，时间下降。\n");
    std::printf("- [C1] 与 [C0] 是同一 workload 的公平对照：用它判断 residency 是否真正带来收益。\n");
    if (effective_window_mb > effective_set_aside_mb) {
      std::printf("- 当前属于 window > set-aside：更依赖 hitRatio 管控，配置不当时收益容易被稀释。\n");
    } else {
      std::printf("- 当前属于 window <= set-aside：这是更容易出现正收益的配置区间。\n");
    }
    std::printf("- [D] 是更激进的替换压力场景：若 [D] 慢于 [C1]，说明 hitRatio=1.0 可能过于乐观。\n");
    std::printf("\nNCU 建议（示例）：\n");
    std::printf("  ncu --set full --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_hit_rate.pct --target-processes all ./bin/02_memory_optim_04_l2_residency\n");
  }

  disable_access_policy_window(stream);
  reset_persisting_l2();

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}

