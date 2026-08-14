/**
 * [Module C] C-06. CUDA Graph 与 Launch Overhead
 *
 * 主命题：短核链上 stream 逐次 launch vs capture-replay 的端到端墙钟。
 *
 * 模式：
 *   stream      : 循环 launch n_nodes 个短核 × chain_reps（基线）
 *   graph       : capture 同序列 → instantiate（计时外）→ GraphLaunch × chain_reps
 *   sweep       : 扫 n_nodes∈{1,2,4,8,16,32,64}（固定极短 work）主曲线
 *   sweep_work  : 固定 n_nodes，扫 work（收益收窄轴）
 *   modes       : 定点 + instantiate 一次成本（host chrono）
 *
 * 主证据：CUDA event median（整条热路径；instantiate 不进热路径）。
 * host CPU chrono of single Launch API = 可选，本章默认不做。
 * 硬件：不限 sm_90+。
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",             \
                   cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__);   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class Mode { Stream = 0, Graph = 1, Sweep = 2, SweepWork = 3, Modes = 4 };

static const char* mode_name(Mode m) {
  switch (m) {
    case Mode::Stream: return "stream";
    case Mode::Graph: return "graph";
    case Mode::Sweep: return "sweep";
    case Mode::SweepWork: return "sweep_work";
    case Mode::Modes: return "modes";
    default: return "unknown";
  }
}

static Mode parse_mode(const char* s) {
  if (std::strcmp(s, "stream") == 0) return Mode::Stream;
  if (std::strcmp(s, "graph") == 0) return Mode::Graph;
  if (std::strcmp(s, "sweep") == 0) return Mode::Sweep;
  if (std::strcmp(s, "sweep_work") == 0) return Mode::SweepWork;
  if (std::strcmp(s, "modes") == 0) return Mode::Modes;
  std::fprintf(stderr,
               "Invalid --mode=%s (expected stream|graph|sweep|sweep_work|modes)\n", s);
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

// Short kernel: grid-stride optional; default tiny n so launch tax dominates when work=0/small.
__global__ void kernel_short(float* __restrict__ data, int n, int work) {
  const int gsize = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gsize) {
    float v = data[i];
    for (int w = 0; w < work; ++w) {
      v = v * 1.0000001f + 0.0000001f;
    }
    data[i] = v;
  }
}

struct BenchConfig {
  Mode mode = Mode::Sweep;
  int n = 4096;           // small footprint: emphasize launch, not bandwidth
  int n_nodes = 16;
  int work = 1;           // FMA iters inside kernel; 0 = almost empty
  int chain_reps = 200;   // repeat whole chain inside one timed region
  int block = 256;
  int grid = 0;           // 0 = auto from n
  int runs = 7;
  int warmup = 2;
  int device = 0;
  bool csv_only = false;
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --mode <stream|graph|sweep|sweep_work|modes> [options]\n"
      "  --n <elems>         element count (default 4096; small → launch-bound)\n"
      "  --n-nodes <n>       kernels per chain (default 16)\n"
      "  --work <iters>      FMA loops per element (default 1)\n"
      "  --chain-reps <n>    repeat chain inside one timed region (default 200)\n"
      "  --block <thr>       threads/block (default 256)\n"
      "  --grid <blocks>     grid size (default 0 = ceil(n/block))\n"
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
    } else if (std::strcmp(argv[i], "--n") == 0) {
      c.n = std::atoi(need("--n"));
    } else if (std::strcmp(argv[i], "--n-nodes") == 0) {
      c.n_nodes = std::atoi(need("--n-nodes"));
    } else if (std::strcmp(argv[i], "--work") == 0) {
      c.work = std::atoi(need("--work"));
    } else if (std::strcmp(argv[i], "--chain-reps") == 0) {
      c.chain_reps = std::atoi(need("--chain-reps"));
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
  if (c.n <= 0 || c.n_nodes < 1 || c.n_nodes > 256 || c.work < 0 || c.chain_reps < 1) {
    std::fprintf(stderr, "ERROR: invalid --n / --n-nodes / --work / --chain-reps\n");
    std::exit(EXIT_FAILURE);
  }
  if (c.block < 32 || c.block > 1024 || (c.block % 32) != 0) {
    std::fprintf(stderr, "ERROR: --block must be in [32,1024], multiple of 32\n");
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

static void launch_stream_chain(cudaStream_t stream, float* d, int n, int work, int n_nodes,
                                int chain_reps, int grid, int block) {
  for (int r = 0; r < chain_reps; ++r) {
    for (int k = 0; k < n_nodes; ++k) {
      kernel_short<<<grid, block, 0, stream>>>(d, n, work);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

struct GraphBundle {
  cudaGraphExec_t exec = nullptr;
  double instantiate_ms = 0.0;
};

static GraphBundle build_graph(cudaStream_t stream, float* d, int n, int work, int n_nodes,
                               int grid, int block) {
  GraphBundle out;
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int k = 0; k < n_nodes; ++k) {
    kernel_short<<<grid, block, 0, stream>>>(d, n, work);
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGetLastError());

  const auto t0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGraphInstantiate(&out.exec, graph, nullptr, nullptr, 0));
  const auto t1 = std::chrono::steady_clock::now();
  out.instantiate_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  CUDA_CHECK(cudaGraphDestroy(graph));
  return out;
}

static void launch_graph_chain(cudaStream_t stream, cudaGraphExec_t exec, int chain_reps) {
  for (int r = 0; r < chain_reps; ++r) {
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
  }
  CUDA_CHECK(cudaGetLastError());
}

static void reset_data(float* d, const std::vector<float>& h) {
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
}

static float checksum_host(const std::vector<float>& h) {
  double s = 0.0;
  for (float v : h) s += static_cast<double>(v);
  return static_cast<float>(s);
}

static void verify_equal(float* d, const std::vector<float>& expect, bool quiet,
                         const char* tag) {
  std::vector<float> got(expect.size());
  CUDA_CHECK(cudaMemcpy(got.data(), d, expect.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const float tol = 1e-3f;
  for (size_t i = 0; i < expect.size(); ++i) {
    if (std::fabs(got[i] - expect[i]) > tol * (std::fabs(expect[i]) + 1.0f)) {
      std::fprintf(stderr, "ERROR: %s mismatch at %zu got=%g expect=%g\n", tag, i, got[i],
                   expect[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (!quiet) {
    std::printf("verify %s OK (checksum≈%.6g)\n", tag, checksum_host(expect));
  }
}

// Run reference on host: apply work*n_nodes*chain_reps transforms? Too heavy.
// Instead: run stream once on device as golden, compare graph to that.
static void device_ref_stream(cudaStream_t stream, float* d, const std::vector<float>& h_in,
                              int n, int work, int n_nodes, int chain_reps, int grid,
                              int block, std::vector<float>* h_out) {
  reset_data(d, h_in);
  launch_stream_chain(stream, d, n, work, n_nodes, chain_reps, grid, block);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  h_out->resize(h_in.size());
  CUDA_CHECK(cudaMemcpy(h_out->data(), d, h_in.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

static const char* kCsvHeader =
    "tag,n_nodes,work,n,chain_reps,median_ms,p10_ms,p90_ms,instantiate_ms";

static void print_row(const char* tag, int n_nodes, int work, int n, int chain_reps,
                      float med_ms, const std::vector<float>& samples, double inst_ms,
                      bool csv_only) {
  const float p10 = percentile_of(samples, 10.0f);
  const float p90 = percentile_of(samples, 90.0f);
  if (csv_only) {
    std::printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n", tag, n_nodes, work, n, chain_reps,
                med_ms, p10, p90, inst_ms);
  } else {
    std::printf("%-10s nodes=%3d work=%d n=%d reps=%d | median=%.4f ms "
                "(p10=%.4f p90=%.4f)",
                tag, n_nodes, work, n, chain_reps, med_ms, p10, p90);
    if (inst_ms > 0.0) std::printf(" | instantiate=%.3f ms", inst_ms);
    std::printf("\n");
  }
}

int main(int argc, char** argv) {
  const BenchConfig cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(cfg.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  const int grid =
      (cfg.grid > 0) ? cfg.grid
                     : std::max(1, (cfg.n + cfg.block - 1) / cfg.block);
  const bool quiet = cfg.csv_only;

  if (!quiet) {
    std::printf("GPU: %s | sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("mode=%s n=%d n_nodes=%d work=%d chain_reps=%d block=%d grid=%d "
                "runs=%d warmup=%d\n",
                mode_name(cfg.mode), cfg.n, cfg.n_nodes, cfg.work, cfg.chain_reps, cfg.block,
                grid, cfg.runs, cfg.warmup);
  }

  std::vector<float> h_in(static_cast<size_t>(cfg.n));
  for (int i = 0; i < cfg.n; ++i) {
    h_in[static_cast<size_t>(i)] = 0.001f * float((i % 997) + 1);
  }

  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, size_t(cfg.n) * sizeof(float)));
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<float> samples;
  std::vector<float> h_ref;

  auto run_stream = [&](int n_nodes, int work, bool verify) -> float {
    if (verify) {
      device_ref_stream(stream, d, h_in, cfg.n, work, n_nodes, cfg.chain_reps, grid,
                        cfg.block, &h_ref);
    }
    auto launch = [&]() {
      reset_data(d, h_in);
      launch_stream_chain(stream, d, cfg.n, work, n_nodes, cfg.chain_reps, grid, cfg.block);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    return time_launch_ms(launch, cfg.warmup, cfg.runs, &samples);
  };

  auto run_graph = [&](int n_nodes, int work, bool verify, double* inst_out) -> float {
    reset_data(d, h_in);
    GraphBundle gb = build_graph(stream, d, cfg.n, work, n_nodes, grid, cfg.block);
    if (inst_out) *inst_out = gb.instantiate_ms;

    if (verify) {
      // golden from stream path already in h_ref if caller verified stream first;
      // else compute ref now
      if (h_ref.empty()) {
        device_ref_stream(stream, d, h_in, cfg.n, work, n_nodes, cfg.chain_reps, grid,
                          cfg.block, &h_ref);
      }
      reset_data(d, h_in);
      launch_graph_chain(stream, gb.exec, cfg.chain_reps);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      verify_equal(d, h_ref, quiet, "graph");
    }

    auto launch = [&]() {
      reset_data(d, h_in);
      launch_graph_chain(stream, gb.exec, cfg.chain_reps);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    const float ms = time_launch_ms(launch, cfg.warmup, cfg.runs, &samples);
    CUDA_CHECK(cudaGraphExecDestroy(gb.exec));
    return ms;
  };

  if (cfg.mode == Mode::Sweep) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else std::printf("\n== sweep n_nodes (main curve: stream/graph @ work=%d) ==\n",
                     cfg.work);
    const int nodes_list[] = {1, 2, 4, 8, 16, 32, 64};
    bool first = true;
    for (int nn : nodes_list) {
      h_ref.clear();
      const float ms_s = run_stream(nn, cfg.work, first);
      first = false;
      print_row("stream", nn, cfg.work, cfg.n, cfg.chain_reps, ms_s, samples, 0.0,
                cfg.csv_only);
      double inst = 0.0;
      const float ms_g = run_graph(nn, cfg.work, true, &inst);
      print_row("graph", nn, cfg.work, cfg.n, cfg.chain_reps, ms_g, samples, inst,
                cfg.csv_only);
      const float sp = (ms_g > 0.f) ? (ms_s / ms_g) : 0.f;
      if (cfg.csv_only) {
        std::printf("speedup_stream_graph,%d,%d,%d,%d,%.6f,,,0\n", nn, cfg.work, cfg.n,
                    cfg.chain_reps, sp);
      } else {
        std::printf("  stream/graph @nodes=%d: %.3fx\n", nn, sp);
      }
    }
  } else if (cfg.mode == Mode::SweepWork) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else
      std::printf("\n== sweep_work (fixed n_nodes=%d; benefit should shrink) ==\n",
                  cfg.n_nodes);
    const int work_list[] = {0, 1, 8, 64, 512, 4096};
    bool first = true;
    for (int w : work_list) {
      h_ref.clear();
      const float ms_s = run_stream(cfg.n_nodes, w, first);
      first = false;
      print_row("stream", cfg.n_nodes, w, cfg.n, cfg.chain_reps, ms_s, samples, 0.0,
                cfg.csv_only);
      double inst = 0.0;
      const float ms_g = run_graph(cfg.n_nodes, w, true, &inst);
      print_row("graph", cfg.n_nodes, w, cfg.n, cfg.chain_reps, ms_g, samples, inst,
                cfg.csv_only);
      const float sp = (ms_g > 0.f) ? (ms_s / ms_g) : 0.f;
      if (cfg.csv_only) {
        std::printf("speedup_stream_graph,%d,%d,%d,%d,%.6f,,,0\n", cfg.n_nodes, w, cfg.n,
                    cfg.chain_reps, sp);
      } else {
        std::printf("  stream/graph @work=%d: %.3fx\n", w, sp);
      }
    }
  } else if (cfg.mode == Mode::Modes) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    else std::printf("\n== modes (fixed n_nodes=%d work=%d) ==\n", cfg.n_nodes, cfg.work);
    h_ref.clear();
    const float ms_s = run_stream(cfg.n_nodes, cfg.work, true);
    print_row("stream", cfg.n_nodes, cfg.work, cfg.n, cfg.chain_reps, ms_s, samples, 0.0,
              cfg.csv_only);
    double inst = 0.0;
    const float ms_g = run_graph(cfg.n_nodes, cfg.work, true, &inst);
    print_row("graph", cfg.n_nodes, cfg.work, cfg.n, cfg.chain_reps, ms_g, samples, inst,
              cfg.csv_only);
    if (!cfg.csv_only) {
      std::printf("speedup stream/graph=%.3fx\n", (ms_g > 0.f) ? (ms_s / ms_g) : 0.f);
      std::printf("instantiate once=%.3f ms (not in hot-path median)\n", inst);
    }
  } else if (cfg.mode == Mode::Stream) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    const float ms = run_stream(cfg.n_nodes, cfg.work, true);
    print_row("stream", cfg.n_nodes, cfg.work, cfg.n, cfg.chain_reps, ms, samples, 0.0,
              cfg.csv_only);
  } else if (cfg.mode == Mode::Graph) {
    if (cfg.csv_only) std::printf("%s\n", kCsvHeader);
    h_ref.clear();
    device_ref_stream(stream, d, h_in, cfg.n, cfg.work, cfg.n_nodes, cfg.chain_reps, grid,
                      cfg.block, &h_ref);
    double inst = 0.0;
    const float ms = run_graph(cfg.n_nodes, cfg.work, true, &inst);
    print_row("graph", cfg.n_nodes, cfg.work, cfg.n, cfg.chain_reps, ms, samples, inst,
              cfg.csv_only);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d));
  return EXIT_SUCCESS;
}
