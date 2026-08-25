# Module A: CUDA 基础架构

本目录包含 CUDA 核心概念和架构相关的示例代码，是《AI 系统性能工程》专栏 Module A 的配套实战代码。

## 📚 目录

| 章节 | 文件 | 核心内容 | 知识点 |
|------|------|----------|--------|
| **第 1 章** | `01_hello_modern.cu` | CUDA 核心概念总览 | Grid-Block-Thread 模型、错误检查、异步执行、Unified Memory |
| **第 2 章** | `02_hardware_query.cu` | GPU 硬件架构深度解析 | SM 架构、内存层次、L2 Cache、Tensor Core 能力、带宽分析 |
| **第 3 章** | `03_grid_mapping.cu` | CUDA 编程模型物理映射 | GTE 派发、occupancy API、尾波、`%smid` |
| **第 4 章** | `04_warp_divergence.cu` | 线程调度：SIMT / Divergence / Replay | mask 串行、ITS 前进、SMEM replay；`clock64` median |
| **第 5 章** | `05_kernel_structure.cu` | Kernel 结构与 ABI | 同一份头对 offsetof；`c[0]` / CALL；Occupancy API |
| **第 6 章** | `06_nvrtc_jit.cpp` | nvcc / Fatbin / NVRTC | 本机 `compute_XY`；编译墙钟；verify 7.0 |
| **第 7 章** | `07_memory_spaces.cu` | 内存空间 / UVA | 地址图；mapped 读 PASS；`localSizeBytes > 0` |
| **第 8 章** | `08_async_pipeline.cu` | Stream / Event / 流水线 | A serial / B depth-first / C breadth-first；CUDA event median |
| **第 9 章** | `09_debug_and_sanitizer.cu` | Compute Sanitizer | mode 0/1/2 → memcheck/racecheck/synccheck |
| **第 10 章** | `10_roofline_demo.cu` | Roofline / SOL | float4 BW + FMA TFLOPS + ridge AI；event median |

## 🚀 快速开始

### 编译构建

#### Linux 环境
```bash
# 在项目根目录
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8
```

#### Windows/CLion 环境
- 使用 CLion 直接构建（构建输出在 `cmake-build-debug` 或 `cmake-build-debug-visual-studio` 目录）
- 或手动构建：
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --parallel 8
```

### 运行示例

编译成功后，可执行文件位置：
- **Linux**: `build/bin/` 目录
- **Windows/CLion**: `cmake-build-debug/bin/` 或 `cmake-build-debug-visual-studio/bin/` 目录

```bash
# Linux: 在 build 目录下运行
./bin/01_cuda_basics_01_hello_modern
./bin/01_cuda_basics_02_hardware_query
./bin/01_cuda_basics_03_grid_mapping
./bin/01_cuda_basics_04_warp_divergence
./bin/01_cuda_basics_05_kernel_structure
./bin/01_cuda_basics_06_nvrtc_jit
./bin/01_cuda_basics_07_memory_spaces
./bin/01_cuda_basics_08_async_pipeline
./bin/01_cuda_basics_09_debug_and_sanitizer
./bin/01_cuda_basics_10_roofline_demo

# Windows/CLion: 在 cmake-build-debug/bin 目录下运行
# 或在 PowerShell 中（从项目根目录）
.\cmake-build-debug\bin\01_cuda_basics_01_hello_modern.exe
.\cmake-build-debug\bin\01_cuda_basics_06_nvrtc_jit.exe
.\cmake-build-debug\bin\01_cuda_basics_07_memory_spaces.exe
.\cmake-build-debug\bin\01_cuda_basics_08_async_pipeline.exe
.\cmake-build-debug\bin\01_cuda_basics_09_debug_and_sanitizer.exe
.\cmake-build-debug\bin\01_cuda_basics_10_roofline_demo.exe
```

---

## 📖 各章详细说明

### 第 1 章：CUDA 核心概念总览 (`01_hello_modern.cu`)

**现代版 Hello World**：展示 CUDA 12+ 的工程规范和生产级代码特征。

#### 核心知识点

1. **宏定义封装**：使用 `CUDA_CHECK` 宏包裹所有 CUDA API 调用，确保在出错时能打印具体的文件名和行号，并安全退出。这种 Fail Fast 策略在生产环境中至关重要。

2. **硬件感知**：在 Kernel 内部通过 `__CUDA_ARCH__` 宏判断当前硬件架构（编译期常量），同时在 Host 端使用 `cudaGetDeviceProperties` 进行运行时硬件查询。

3. **异步执行与同步**：演示 Kernel 启动的异步特性，以及 `cudaGetLastError()` 和 `cudaDeviceSynchronize()` 的正确使用。

4. **统一内存管理**：使用 `cudaMallocManaged` 简化内存分配，数据会在 CPU/GPU 间自动按需迁移（Page Migration）。

5. **线程索引计算**：展示 CUDA 的 Grid-Block-Thread 三层执行模型，通过 `global_id = blockIdx.x * blockDim.x + threadIdx.x` 计算全局线程索引。

#### 预期输出

```shell
[Host] Starting Modern CUDA Hello World...
[Host] GPU Name: NVIDIA GeForce RTX 5090
[Host] SM Count: 170
[Host] Compute Capability: 12.0
[Host] Launching Kernel...
[Device] Kernel running on SM arch sm_750
[Device] GridDim=1, BlockDim=32
[Host] Verifying results...
[Host] Verification PASSED! [OK]
```

#### 二进制分析工具

项目提供了 `01_fatbin_inspect.sh` 脚本，利用 `cuobjdump` 工具分析编译后的二进制文件：

```bash
cd examples/01_cuda_basics
bash 01_fatbin_inspect.sh
```

**注意**：脚本会自动检测构建目录：
- **Windows/CLion**: `cmake-build-debug/bin` 或 `cmake-build-debug-visual-studio/bin`
- **Linux**: `build/bin`

该脚本可以展示：
- **PTX（虚拟架构）**：中间表示代码，由驱动程序在运行时 JIT 编译到目标架构
- **SASS（真实架构）**：实际运行在 GPU 上的机器码
```shell
=== 1. Inspecting Virtual Architectures (PTX) ===
PTX is just-in-time compiled by the driver.
arch = sm_75

=== 2. Inspecting Real Architectures (SASS) ===
SASS is the actual machine code running on silicon.
arch = sm_75
```
这可以验证 CMake 配置中的 `CMAKE_CUDA_ARCHITECTURES` 是否生效。

---

### 第 2 章：GPU 硬件架构深度解析 (`02_hardware_query.cu`)

**硬件拓扑侦探**：挖掘 SM 架构、L2 Cache、Tensor Core 能力与带宽极限。

#### 核心知识点

1. **计算能力与架构识别**：按 CC 区分 Ampere / **Ada（不是 Hopper）** / Hopper / Blackwell 消费卡（sm_120）与数据中心卡。未知 CC 的 Core/SM 印 Unknown。

2. **SM 宏观拓扑**：SM 数；表驱动的 CUDA Core/SM（5090 按 128 推算）。

3. **内存体系**：
   - 全局内存容量、总线宽度
   - CUDA 12+ 无 `memoryClockRate` 时理论带宽印 `N/A`
   - **L2 Cache**（芯片级，不是 GPC 私有）

4. **SM 资源上限**：SMEM / 寄存器 / 线程 / `warpSize`

5. **现代特性**：UVA、Managed Memory；TMA / Cluster 只报 **ISA 门槛 sm_90+**，不报加速比（实测在 B-08）

#### 预期输出

```shell
=================================================================
   AI System Performance Lab - Hardware Topology Detective
=================================================================
Detected 1 CUDA Capable Device(s)

[Device 0]: NVIDIA GeForce RTX 5090
-----------------------------------------------------------------
  [Architecture]
    Compute Capability      : 12.0 (Blackwell consumer, e.g. RTX 50 / sm_120)
  [Compute Topology]
    Multiprocessors (SMs)   : 170
    CUDA Cores / SM         : 128
    Total CUDA Cores        : 21760
    GPU Clock Rate          : N/A (removed in CUDA 12+)
  [Memory Hierarchy]
    Global Memory (HBM/DDR) : 31.36 GB
    Memory Bus Width        : 512-bit
    Memory Clock Rate       : N/A (removed in CUDA 12+)
    Theoretical Bandwidth   : N/A (use nvml API for accurate value)
    L2 Cache Size           : 96.00 MB (chip-wide; not per-GPC)
  [SM Micro-Architecture]
    Max Shared Mem / Block  : 48.00 KB
    Max Shared Mem (Opt-in) : 99.00 KB (Dynamic)
    Max Registers / Block   : 65536
    Max Threads / Block     : 1024
    Max Threads / SM        : 1536
    Warp Size               : 32
  [Modern Features Support]
    Unified Addressing      : Yes
    Managed Memory          : Yes
    TMA ISA floor (sm_90+)  : Yes (measure in B-08)
    Cluster ISA floor       : Yes (measure in B-08)
```

#### 注意事项

- 在 CUDA 12+ 版本中，`clockRate` 和 `memoryClockRate` 字段已被移除，代码使用条件编译兼容新旧版本。
- 如需获取准确的时钟频率和带宽信息，建议使用 NVML (NVIDIA Management Library) API。

---

### 第 3 章：CUDA 编程模型物理映射 (`03_grid_mapping.cu`)

**Grid tracer**：`%smid` 看 Block 落在哪颗 SM；occupancy API 定波次。不是 kernel 计时。

#### 核心知识点

1. **PTX `%smid`**：`mov.u32 %0, %smid` 读物理 SM。不是公开稳定 ABI。
2. **occupancy API**：`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 定本 kernel 的 blocks/SM，再 launch `5 * wave + 1`。不要猜每 SM 4 个 Block。
3. **atomic 序号**：thread 0 抢序号，**不是 GTE 派发日志**。
4. **busy-wait**：`clock64` 只为把 Block 拉长；median 计时去 A-08。

#### 预期输出（数字以本机为准）

```text
[Host] GPU: NVIDIA GeForce RTX 5090
[Host] Compute Capability: 12.0
[Host] SM count: 170
[Host] Occupancy (this kernel, blockSize=1): <blocks/SM>
[Host] Wave size ≈ <SM × blocks/SM> blocks; launching <5 waves + 1 tail> blocks
[Host] Note: <<<N,1>>> occupancy is NOT a 256-thread + SMEM kernel.

[Analysis 1] Blocks finished per SM (min=... max=...; first 5 SMs):
  SM 00 : ...
  ...

[Analysis 2] Tail (atomic order, NOT a GTE log):
  Last numbered logical Block ... ran on SM ...

[Visualizer] logical Block -> SM (first 64):
  ...
[Conclusion] Every SM received at least one Block.
```

---

### 第 4 章：线程调度：SIMT, Divergence 与 Replay (`04_warp_divergence.cu`)

**lane0 `clock64` median**（warmup 后 21 次）。不是 CUDA event kernel 时间。

#### 核心知识点

1. **Divergence**：无分支 FMA vs 奇偶 `if`；结果写入 `sink` 防止 DCE（旧构建会打出 1 cycle）。
2. **Replay**：SMEM stride=1 vs stride=32。消冲突去 B-02。
3. **口径**：比值看形状，不要把 2× / 32× 当账单；event 去 A-08。

#### 预期输出（数字以本机为准）

```text
[Host] GPU: NVIDIA GeForce RTX 5090
[Host] Compute Capability: 12.0  sm_120
[Host] Metric: lane0 clock64 median (warmup=5, runs=21, iters=4096)
[Host] Grid: <<<1, 32>>> (one warp; ratios closer to textbook)
[Host] Not CUDA event kernel time (see A-08).

[Divergence] uniform cycles : ...
[Divergence] odd/even if    : ...
[Divergence] ratio          : ...x  (textbook isolation ~2x; not a bill)

[Replay]     stride-1 cycles : ...
[Replay]     stride-32       : ...
[Replay]     ratio           : ...x  (textbook 32-way ~32x; padding -> B-02)
```

---

---

### 第 5 章：Kernel 结构与 ABI (`05_kernel_structure.cu`)

**不是 kernel 墙钟。** 打印 Host/Device 布局，以及 `cudaFuncGetAttributes` + occupancy API。

#### 核心知识点

1. **同一份头**：Default / `alignas` / `#pragma pack(1)`。Host 与 Device 的 `offsetof` 应当一致；出事是两份定义。
2. **SASS**：`__noinline__` 常见 `CALL`；参数常见 `c[0x0][…]`。`cuobjdump -sass`，不要包 event。
3. **`__launch_bounds__(256, 4)`**：occupancy 契约。regs / local / blocks/SM 以本机打印为准。spill 去 B-03。

#### 预期输出（数字以本机为准）

```text
[Host] GPU: NVIDIA GeForce RTX 5090
[Host] Compute Capability: 12.0  sm_120
[Host] Not CUDA event kernel time (see A-08).

[Host] Same header, three layouts:
[Host]   DefaultLayout offsetof(b)=4 sizeof=8
[Host]   AlignedLayout offsetof(b)=4 sizeof=8
[Host]   PackedLayout  offsetof(b)=1 sizeof=5
[Device] ... same offsetof/sizeof; values 42 / 100 / 7

[Host] Inline kernels ran. SASS: cuobjdump -sass <bin>  (CALL / c[0x0]).

[Host] heavy_default: regs=... local=...B  occupancy=... blocks/SM @ 256 threads
[Host] heavy_bounded: regs=... local=...B  occupancy=... blocks/SM @ 256 threads
```

可选 SASS：`cuobjdump -sass ./bin/01_cuda_basics_05_kernel_structure`（Linux `grep -E 'CALL|c\\[0x0\\]'`，Windows `findstr CALL`）。仓库里的 `05_inspect_asm.sh` 仍可用。

---

### 第 6 章：工具链 / NVRTC (`06_nvrtc_jit.cpp`)

**不是 kernel 墙钟。** `nvrtcCompileProgram` 的 Host 毫秒是编译墙钟。

#### 核心知识点

1. **特化**：把 `5.0f` 写进源码字符串。证明能编、能跑，不是立即数加速比。
2. **架构**：`--gpu-architecture=compute_XY` 用本机 CC，避免 `CUDA_ERROR_INVALID_PTX`。
3. **加载**：Driver API `cuModuleLoadData` / `cuLaunchKernel`；内存仍可用 Runtime。

#### 预期输出（数字以本机为准）

```text
[Host] GPU: NVIDIA GeForce RTX 5090
[Host] Compute Capability: 12.0  sm_120
[Host] Not CUDA event kernel time (see A-08).
[NVRTC] specialized: out[i] = 5.0f * x[i] + y[i];
[NVRTC] arch: --gpu-architecture=compute_120
[NVRTC] compile host-ms: ...
[NVRTC] PTX bytes: ...
[Host] verify: PASS  expected 5.0*1.0+2.0=7.0
```

---

### 第 7 章：内存空间 / UVA (`07_memory_spaces.cu`)

**不是 kernel 墙钟，也不是 PCIe 账单。** 主证据是 mapped 读通，以及 spill kernel 的 `localSizeBytes > 0`。吞吐去 B-06，UM 去 B-05。

#### 核心知识点

1. **地址图**：`cudaMalloc` / `__device__` / `__shared__` / 取地址的 local / mapped Host。取地址 → Local（HBM），不是片上栈。
2. **UVA**：打印 `cudaDevAttrUnifiedAddressing`。统一的是 VA，不是 UM 迁页。
3. **Mapped**：kernel 读一个 `int`，期望 PASS。不要循环扫 Host。
4. **`__restrict__`**：别名契约；不报 `LDG.NC` 加速比。

#### 预期输出（数字以本机为准）

```text
[Host] GPU: NVIDIA GeForce RTX 5090
[Host] Compute Capability: 12.0  sm_120
[Host] UnifiedAddressing: yes
[Device] mapped host read: 999  expected 999
[Host] mapped read: PASS
[Host] force_local_memory_spill: regs=... localSizeBytes=...
[Host] spill: PASS  localSizeBytes > 0
```

SASS 可选：Linux `07_inspect_sass.sh`；Windows 用 `cuobjdump -sass ... | findstr LDL`。

---

### 第 8 章：Stream / Event / 流水线 (`08_async_pipeline.cu`)

**口径**：CUDA event median（warmup=2, runs=7）包整段 device 工作；`NonBlocking` 流在 record `stop` 前 `cudaDeviceSynchronize`。`clock64` busy-wait 只为让 overlap 可见，不是真实 AI。Pinned GB/s → B-06。

#### Mode

| Mode | 配置 |
|------|------|
| A | pageable + default stream 0 |
| B | pinned + 4× `cudaStreamNonBlocking` + **depth-first**（每 chunk：H2D→K→D2H） |
| C | 同 B 资源，**breadth-first**（先全部 H2D，再全部 K，再全部 D2H） |

#### 预期输出（RTX 5090 / `sm_120` 参考）

```
GPU: NVIDIA GeForce RTX 5090
sm_120  asyncEngineCount=2
Metric: CUDA event median (warmup=2, runs=7); not Host chrono
...

mode,median_ms
A_serial_pageable_default,11.825
B_depth_first_pinned,1.408
C_breadth_first_pinned,1.835

ratio A/B (serial / depth-first): 8.40x
ratio C/B (breadth / depth-first): 1.30x
```

本机形状：`A/B ≈ 8.40×`，`C/B ≈ 1.30×`。不报 PCIe GB/s（B-06）。

#### 可选旁证

```bash
cd examples/01_cuda_basics
bash 08_profile_nsys.sh
```

Linux/WSL；看 NSYS 时间轴 copy∥kernel。不进 TL;DR 数字。

---

### 第 9 章：Compute Sanitizer (`09_debug_and_sanitizer.cu`)

**口径**：故意种 bug，用 `compute-sanitizer` 归因。不是 timing bench。initcheck 无独立 mode。

| mode | bug | tool |
|------|-----|------|
| 0 | OOB write `data[N]`（本机 memcheck PASS） | `memcheck` |
| 1 | SMEM race（本机 Hazard PASS） | `racecheck` |
| 2 | `__syncwarp` mask 缺 thread（本机 Invalid arguments PASS） | `synccheck` |

```bash
./bin/01_cuda_basics_09_debug_and_sanitizer 0
compute-sanitizer --tool memcheck  ./bin/01_cuda_basics_09_debug_and_sanitizer 0
compute-sanitizer --tool racecheck ./bin/01_cuda_basics_09_debug_and_sanitizer 1
compute-sanitizer --tool synccheck ./bin/01_cuda_basics_09_debug_and_sanitizer 2
# or: bash examples/01_cuda_basics/09_run_sanitizer.sh
```

期望：报告里出现 planted kernel 名与 ERROR/Hazard。裸跑可能不崩。

---

### 第 10 章：Roofline / SOL (`10_roofline_demo.cu`)

**口径**：CUDA event median（warmup=2, runs=7）。ridge 用**实测** BW 与 TFLOPS，不用估 clock。NCU 可选。

| 段 | 内容 |
|------|------|
| A | `float4` 大缓冲 copy → `achieved_bw` GB/s；AI≈0 |
| B | 寄存器 FMA 探针 → `achieved_fp32` TFLOPS |
| C | `ridge_AI` + copy/compute 相对 ridge 的位置 |

```bash
./bin/01_cuda_basics_10_roofline_demo
# optional: bash examples/01_cuda_basics/10_profile_roofline.sh
```

期望（RTX 5090 参考）：`achieved_bw≈1954 GB/s`，`achieved_fp32≈49 TFLOPS`，`ridge_AI≈25`；copy memory 侧、FMA compute 侧。绝对 GB/s 可含短核/L2 噪声。

---


## 🔧 工具脚本

- `01_fatbin_inspect.sh`：二进制文件分析工具，用于查看 PTX 和 SASS 代码
- `05_inspect_asm.sh`：SASS 汇编分析工具，用于验证函数内联行为
- `07_inspect_sass.sh`：SASS 内存分析工具，用于检测 Local Memory Spilling 和 `__restrict__` 优化效果
- `08_profile_nsys.sh`：性能分析脚本（Linux/WSL 专用），使用 Nsight Systems 分析异步流水线性能
- `09_run_sanitizer.sh`：调试工具脚本，使用 Compute Sanitizer 检测内存越界、数据竞争和非法同步
- `10_profile_roofline.sh`：性能分析脚本（Linux/WSL 专用），使用 Nsight Compute 进行 Roofline 性能建模

## 📝 注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台
- 兼容 CUDA 12.0+ 版本（部分字段在 CUDA 12+ 中已移除，使用条件编译处理）
