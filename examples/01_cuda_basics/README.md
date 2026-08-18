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
| **第 7 章** | `07_memory_spaces.cu` | 内存模型全景 | 地址空间探测、UVA Zero-Copy、Local Memory Spilling、__restrict__ 优化 |
| **第 8 章** | `08_async_pipeline.cu` | 异步执行模型 | Pinned Memory、多 Stream 并发、Depth-First 调度、流水线 Overlap |
| **第 9 章** | `09_debug_and_sanitizer.cu` | 调试与错误诊断 | Compute Sanitizer、内存越界检测、数据竞争检测、非法同步检测 |
| **第 10 章** | `10_roofline_demo.cu` | 性能建模第一性原理 | Roofline 模型、带宽极限测试、算力极限测试、Arithmetic Intensity |

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
[Host] Compute Capability: 12.0
[Host] Metric: lane0 clock64 median (warmup=5, runs=21, iters=4096)
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

### 第 7 章：内存模型全景 (`07_memory_spaces.cu`)

**内存层次深度解析**：探索 CUDA 的地址空间、UVA Zero-Copy、Local Memory Spilling 与 `__restrict__` 优化。

#### 核心知识点

1. **地址空间探测**：
   - **Global Memory (HBM)**：设备全局内存，通过 `cudaMalloc` 分配
   - **Global Variable (Static)**：设备静态变量，使用 `__device__` 声明
   - **Shared Memory (SRAM)**：每个 Block 共享的片上高速缓存
   - **Local Variable (Stack)**：线程局部变量，通常存储在寄存器中
   - **Host Pinned Memory (UVA)**：主机固定内存，可通过 UVA 直接访问

2. **UVA Zero-Copy 实战**：
   - 使用 `cudaHostAllocMapped` 分配主机固定内存
   - 通过 Unified Virtual Addressing (UVA) 实现 GPU 直接访问 CPU 内存
   - 验证 Zero-Copy 功能（无需显式 `cudaMemcpy`）
   - **性能警告**：Zero-Copy 走 PCIe 总线（~64GB/s），远慢于 HBM（~2000GB/s）

3. **Local Memory Spilling（寄存器溢出）**：
   - 当局部变量过多或使用动态索引时，编译器会将数据溢出到 Local Memory
   - Local Memory 实际存储在 HBM 中，访问延迟极高（~400 cycles）
   - 在 SASS 代码中表现为 `LDL`（Local Load）和 `STL`（Local Store）指令
   - 会污染 L1 Cache，严重影响性能

4. **`__restrict__` 优化**：
   - 向编译器保证指针不会重叠（Aliasing）
   - 允许编译器进行更激进的优化：
     - 向量化加载（`LDG.128`）
     - 使用 Texture Cache（`LDG.NC`）
     - 减少内存访问指令数量
   - 对比无 `__restrict__` 和有 `__restrict__` 的 Kernel，观察 SASS 代码差异

#### 预期输出

```shell
[Host] Starting Memory Hierarchy Analysis...
[Host] Launching Address Probe...

[Device] === Memory Address Map ===
  Global Memory (HBM) Ptr:    0x78138b000000
  Global Variable (Static):   0x78138b400800
  Shared Memory (SRAM):       0x781400000400 (Small offset usually)
  Local Variable (Stack):     0x78139dfffce0 (If address taken -> Local Mem)
  Host Pinned Ptr (UVA/PCIe): 0x78138b200000
================================

[Device] Read from Host Pinned Memory: 999 (Success! UVA works)

[Host] To see Local Memory Spilling instructions (LDL/STL),
       please run the accompanying '07_inspect_sass.sh' script.
```

#### SASS 分析工具

项目提供了 `07_inspect_sass.sh` 脚本，用于分析 SASS 代码中的内存访问模式：

```bash
cd examples/01_cuda_basics
bash 07_inspect_sass.sh
```

**注意**：脚本会自动检测构建目录（支持 Windows/CLion 和 Linux 两种构建方式）。

该脚本可以：
- **检测 Local Memory Spilling**：搜索 `STL`/`LDL` 指令，验证寄存器溢出
- **对比 `__restrict__` 优化**：列出相关函数，便于手动对比 SASS 代码差异
- **验证内存访问模式**：识别向量化加载和 Texture Cache 使用
```shell
========================================================
   SASS Analysis: Local Memory Spilling
========================================================
Searching for Local Store (STL) and Local Load (LDL) instructions...
These indicate data is spilling to HBM (Slow!).

                                                                                      /* 0x000fc40000000028 */
        /*00c0*/                   FFMA R9, R3.reuse, R40.reuse, 5 ;                  /* 0x40a0000003097423 */
                                                                                      /* 0x0c0fe40000000028 */
        /*00d0*/                   FFMA R8, R3.reuse, R40.reuse, 4 ;                  /* 0x4080000003087423 */
                                                                                      /* 0x0c0fe20000000028 */
        /*00e0*/                   STL.128 [R1], R4 ;                                 /* 0x0000000401007387 */
--
                                                                                      /* 0x0c0fe40000000028 */
        /*0110*/                   FFMA R13, R3.reuse, R40.reuse, 9 ;                 /* 0x41100000030d7423 */
                                                                                      /* 0x0c0fe40000000028 */
        /*0120*/                   FFMA R12, R3.reuse, R40.reuse, 8 ;                 /* 0x41000000030c7423 */
                                                                                      /* 0x0c0fe20000000028 */
        /*0130*/                   STL.128 [R1+0x10], R8 ;                            /* 0x0000100801007387 */
--
                                                                                      /* 0x0c0fe40000000028 */
        /*0160*/                   FFMA R29, R3.reuse, R40.reuse, 37 ;                /* 0x42140000031d7423 */
                                                                                      /* 0x0c0fe40000000028 */
        /*0170*/                   FFMA R28, R3.reuse, R40.reuse, 36 ;                /* 0x42100000031c7423 */
                                                                                      /* 0x0c0fe20000000028 */
        /*0180*/                   STL.128 [R1+0x20], R12 ;                           /* 0x0000200c01007387 */

NOTE: If you see STL/LDL inside 'force_local_memory_spill', spilling occurred.
========================================================

========================================================
   SASS Analysis: __restrict__ Optimization
========================================================
Comparing No-Restrict vs With-Restrict kernels...
Ideally, 'With-Restrict' might use LDG.NC (Non-coherent/Texture) or fewer instructions.

[Functions found in binary]
                Function : _Z17add_with_restrictPfS_S_i
                Function : _Z15add_no_restrictPfS_S_i

Tip: Use 'cuobjdump -sass ... > out.txt' to manually compare the assembly.
========================================================
```
#### 注意事项

- UVA Zero-Copy 适合小数据量或随机访问模式，大数据量传输应使用 `cudaMemcpy`
- Local Memory Spilling 是性能杀手，应尽量避免：
  - 减少局部数组大小
  - 使用 Shared Memory 替代大局部数组
  - 避免对局部数组使用动态索引
- `__restrict__` 是性能优化的重要工具，但需要确保指针确实不重叠

---

### 第 8 章：异步执行模型 (`08_async_pipeline.cu`)

**Pipeline Concurrency**：实现 H2D -> Compute -> D2H 三级流水线，最大化 GPU 利用率。

#### 核心知识点

1. **Pinned Memory（页锁定内存）的必要性**：
   - 使用 `cudaMallocHost` 分配 Pinned Memory，允许 DMA 引擎直接访问
   - Pageable Memory（普通 `malloc`）会导致驱动介入进行临时拷贝，无法实现真正的异步传输
   - Pinned Memory 是异步传输的前提条件

2. **多 Stream 并发**：
   - 创建多个 CUDA Stream（使用 `cudaStreamCreateWithFlags` 和 `cudaStreamNonBlocking`）
   - 不同 Stream 中的操作可以并发执行，掩盖 PCIe 传输延迟
   - 理想情况下，当 Stream 0 在执行计算时，Stream 1 可以在进行数据传输

3. **Depth-First 调度策略**：
   - 按 Chunk 顺序循环分配 Stream（`stream_idx = i % n_streams`）
   - 每个 Stream 依次执行：H2D Copy -> Compute -> D2H Copy
   - 这种模式能最大化 Overlap：当 Stream 0 在计算时，Stream 1 在拷贝

4. **流水线 Overlap 验证**：
   - 对比串行模式（Pageable Memory + Default Stream）vs 异步流水线模式（Pinned Memory + Multi-Streams）
   - 使用 Nsight Systems 可视化时间线，观察 Copy 和 Compute 的重叠
   - 理想情况下，异步流水线能显著提升吞吐量

#### 预期输出

```
GPU: NVIDIA GeForce RTX 5090
Data Size: 32.00 MB, Chunk Size: 1.00 MB

[Serial] Starting processing 32 chunks...
[Serial] Total Time: 40.20 ms
------------------------------------------------
[Pipeline] Starting processing 32 chunks with 4 streams...
[Pipeline] Total Time: 1.32 ms
```

#### 性能分析工具

项目提供了 `08_profile_nsys.sh` 脚本，使用 Nsight Systems 进行性能分析：

```bash
cd examples/01_cuda_basics
bash 08_profile_nsys.sh
```

**注意**：
- 脚本会自动检测构建目录（支持 Windows/CLion 和 Linux 两种构建方式）
- **仅支持 Linux/WSL 环境**（Nsight Systems 需要 Linux 环境）
- 脚本会生成 `.nsys-rep` 文件，需要在 Nsight Systems GUI 中打开

该脚本可以：
- **追踪 CUDA API 调用**：记录所有 `cudaMemcpyAsync` 和 Kernel Launch
- **可视化时间线**：在 Nsight Systems GUI 中查看 Copy 和 Compute 的重叠情况
- **验证 Overlap 效果**：观察 "CUDA HW" 行中的并发执行情况
```shell
========================================================
   Profiling with Nsight Systems (nsys)
========================================================
Tracing CUDA API and GPU Workload...

Collecting data...
GPU: NVIDIA GeForce RTX 5090
Data Size: 32.00 MB, Chunk Size: 1.00 MB

[Serial] Starting processing 32 chunks...
[Serial] Total Time: 14.20 ms
------------------------------------------------
[Pipeline] Starting processing 32 chunks with 4 streams...
[Pipeline] Total Time: 1.41 ms
Generating '/tmp/nsys-report-c086.qdstrm'
[1/1] [========================100%] pipeline_trace.nsys-rep
Generated:
        /data/AI-System-Performance-Lab/build/pipeline_trace.nsys-rep

========================================================
Done! Please open 'pipeline_trace.nsys-rep' in Nsight Systems GUI.
Look for the 'CUDA HW' row to see the overlap.
========================================================
```
#### 技术细节

- **异步传输**：`cudaMemcpyAsync` 需要 Pinned Memory 才能实现真正的异步
- **Stream 同步**：使用 `cudaDeviceSynchronize()` 等待所有 Stream 完成
- **计算负载模拟**：使用 `clock64()` 进行忙等待，模拟重计算任务
- **Chunk 大小调优**：切得太小会导致 Launch Overhead 占比过高，切得太大 Overlap 效果差

#### 注意事项

- Pinned Memory 分配会占用系统内存，不要过度使用
- Stream 数量需要根据硬件能力调整（通常 4-8 个 Stream 效果较好）
- 理想的 Overlap 是 Compute Time ≈ Copy Time，需要根据实际负载调整 `KERNEL_LOAD` 参数
- Windows 环境下无法直接运行 `nsys`，需要在 WSL 或 Linux 环境中使用

---

### 第 9 章：调试与错误诊断 (`09_debug_and_sanitizer.cu`)

**Bug Generator**：故意制造三种典型 GPU 错误，演示 Compute Sanitizer 的检测能力。

#### 核心知识点

1. **Compute Sanitizer 工具套件**：
   - **Memcheck**：检测内存越界访问、未初始化内存使用、内存泄漏
   - **Racecheck**：检测 Shared Memory 和 Global Memory 的数据竞争
   - **Synccheck**：检测非法同步操作（如分支发散中的 `__syncthreads()`）
   - 这些工具是 CUDA 官方提供的运行时错误检测工具，类似于 Valgrind

2. **内存越界检测（Out-of-Bounds）**：
   - 演示当线程索引超出分配的内存范围时的行为
   - `oob_kernel` 中，当 `idx == n` 时发生越界写入
   - Memcheck 能够精确定位越界访问的位置和线程索引

3. **数据竞争检测（Race Condition）**：
   - 演示多个线程同时读写 Shared Memory 同一地址的问题
   - `race_kernel` 中，所有线程同时执行 `s_val += 1`，结果未定义
   - Racecheck 能够检测到这种竞争条件，并报告冲突的线程

4. **非法同步检测（Illegal Synchronization）**：
   - 演示在分支发散区域调用 `__syncthreads()` 的问题
   - `illegal_sync_kernel` 中，只有一半线程能到达同步点，导致死锁
   - Synccheck 能够检测到这种非法同步，并报告发散的分支

#### 运行方式

```bash
# 直接运行（会触发错误，但可能不会立即报错）
./bin/01_cuda_basics_09_debug_and_sanitizer 0  # Out-of-Bounds
./bin/01_cuda_basics_09_debug_and_sanitizer 1  # Race Condition
./bin/01_cuda_basics_09_debug_and_sanitizer 2  # Illegal Sync

# 使用 Sanitizer 检测（推荐）
cd examples/01_cuda_basics
bash 09_run_sanitizer.sh
```

#### 预期输出（使用 Sanitizer）

```shell
=========
========= Barrier error detected. Divergent thread(s) in warp.
=========     at illegal_sync_kernel(int *)+0xa0 in 09_debug_and_sanitizer.cu:61
=========     by thread (19,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x1754] in 01_cuda_basics_09_debug_and_sanitizer
=========
========= Barrier error detected. Divergent thread(s) in warp.
=========     at illegal_sync_kernel(int *)+0xa0 in 09_debug_and_sanitizer.cu:61
=========     by thread (21,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x1754] in 01_cuda_basics_09_debug_and_sanitizer
=========
========= Barrier error detected. Divergent thread(s) in warp.
=========     at illegal_sync_kernel(int *)+0xa0 in 09_debug_and_sanitizer.cu:61
=========     by thread (23,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x1754] in 01_cuda_basics_09_debug_and_sanitizer
=========
========= Barrier error detected. Divergent thread(s) in warp.
=========     at illegal_sync_kernel(int *)+0xa0 in 09_debug_and_sanitizer.cu:61
=========     by thread (25,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x1754] in 01_cuda_basics_09_debug_and_sanitizer
=========
========= Barrier error detected. Divergent thread(s) in warp.
=========     at illegal_sync_kernel(int *)+0xa0 in 09_debug_and_sanitizer.cu:61
=========     by thread (27,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x1754] in 01_cuda_basics_09_debug_and_sanitizer
=========
========= Barrier error detected. Divergent thread(s) in warp.
=========     at illegal_sync_kernel(int *)+0xa0 in 09_debug_and_sanitizer.cu:61
=========     by thread (29,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x1754] in 01_cuda_basics_09_debug_and_sanitizer
=========
========= Barrier error detected. Divergent thread(s) in warp.
=========     at illegal_sync_kernel(int *)+0xa0 in 09_debug_and_sanitizer.cu:61
=========     by thread (31,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x1754] in 01_cuda_basics_09_debug_and_sanitizer
=========
CUDA Error: unspecified launch failure at /data/AI-System-Performance-Lab/examples/01_cuda_basics/09_debug_and_sanitizer.cu:104
========= Target application returned an error
========= ERROR SUMMARY: 16 errors

Done. Analyze the output above to find the bugs.
```

#### 调试工具

项目提供了 `09_run_sanitizer.sh` 脚本，自动运行三种 Sanitizer 工具：

```bash
cd examples/01_cuda_basics
bash 09_run_sanitizer.sh
```

**注意**：
- 脚本会自动检测构建目录（支持 Windows/CLion 和 Linux 两种构建方式）
- **需要安装 CUDA Toolkit**（Compute Sanitizer 随 CUDA Toolkit 一起安装）
- 脚本会依次运行三种检测工具，输出详细的错误信息

该脚本可以：
- **自动运行 Memcheck**：检测内存越界和泄漏
- **自动运行 Racecheck**：检测数据竞争
- **自动运行 Synccheck**：检测非法同步

#### 技术细节

- **Compute Sanitizer**：CUDA 11.0+ 提供的运行时错误检测工具
- **内存越界**：可能导致程序崩溃或数据损坏，但有时可能不会立即报错
- **数据竞争**：结果未定义，可能导致难以调试的 Bug
- **非法同步**：会导致死锁或未定义行为，Synccheck 能够检测到

#### 注意事项

- Compute Sanitizer 会显著降低程序性能（通常慢 10-100 倍），仅用于调试
- 某些错误（如异步错误）可能不会立即报错，需要等待同步点
- 建议在开发阶段定期使用 Sanitizer 检查代码
- Windows 环境下 Compute Sanitizer 功能有限，建议在 Linux/WSL 环境中使用

---

### 第 10 章：性能建模第一性原理 (`10_roofline_demo.cu`)

**Roofline Empirical Prober**：实测硬件的带宽极限（Bandwidth）与算力极限（FLOPs），构建 Roofline 性能模型。

#### 核心知识点

1. **Roofline 模型基础**：
   - **带宽极限（Memory Bound）**：当 Arithmetic Intensity (AI) 较低时，性能受限于内存带宽
   - **算力极限（Compute Bound）**：当 AI 较高时，性能受限于计算能力
   - **Roofline 曲线**：描述不同 AI 值下的性能上限，帮助识别性能瓶颈

2. **带宽测试（Bandwidth Kernel）**：
   - 使用 `float4` 向量化读写，生成 128-bit LDG/STG 指令，最大化总线利用率
   - Grid-Stride Loop 模式，确保所有线程都有工作
   - AI = 0（纯内存拷贝，无计算），用于测试内存带宽上限
   - 数据规模足够大（64MB），避开 L2 Cache，直接测试 HBM 带宽

3. **算力测试（Compute Kernel）**：
   - 使用寄存器级 FMA（Fused Multiply-Add）密集计算
   - 4 条独立的指令流（ILP），填满流水线
   - 极高的 AI 值，确保瓶颈完全在 ALU
   - `#pragma unroll` 展开循环，减少分支指令占比

4. **理论峰值计算**：
   - **带宽峰值**：Memory Clock × Bus Width × 2 (DDR) / 8
   - **算力峰值**：SM Clock × SMs × Cores/SM × 2 (FMA) / 1e9
   - 注意：CUDA 12+ 中 `clockRate` 和 `memoryClockRate` 字段已移除，需要使用 NVML API 获取准确值

#### 预期输出

```
----------------------------------------------------------------
[Theoretical Peaks] Device: NVIDIA GeForce RTX 5090 (SMs: 170)
  > Memory Clock      : N/A (removed in CUDA 12+, using estimate: 1.00 GHz)
  > Memory Bus Width  : 512-bit
  > Peak Bandwidth    : 128.00 GB/s (Estimated, use NVML for accurate value)
  > SM Clock          : N/A (removed in CUDA 12+, using estimate: 1.50 GHz)
  > Peak FP32 Compute : 65.28 TFLOPS (Estimated)
  > Note              : For accurate clock rates, use NVML API
----------------------------------------------------------------

[Micro-Bench 1] Measuring HBM Bandwidth...
  > Achieved Bandwidth: 2087.03 GB/s

[Micro-Bench 2] Measuring FP32 Compute Peak...
  > Achieved Compute  : 87.20 TFLOPS
```

#### 性能分析工具

项目提供了 `10_profile_roofline.sh` 脚本，使用 Nsight Compute 进行 Roofline 分析：

```bash
cd examples/01_cuda_basics
bash 10_profile_roofline.sh
```

**注意**：
- 脚本会自动检测构建目录（支持 Windows/CLion 和 Linux 两种构建方式）
- **需要安装 Nsight Compute**（随 CUDA Toolkit 一起安装）
- 脚本会生成 `.ncu-rep` 文件，需要在 Nsight Compute GUI 中打开

该脚本可以：
- **自动运行 Roofline 分析**：使用 `--set roofline` 收集 Roofline 数据
- **生成 Roofline 图表**：在 Nsight Compute GUI 中可视化性能瓶颈
- **识别性能边界**：观察 Memory Bound 和 Compute Bound 两个点
```shell
Using binary: /data/AI-System-Performance-Lab/build/bin/01_cuda_basics_10_roofline_demo

==========================================================
   Profiling Roofline with Nsight Compute (ncu)
==========================================================
Output: roofline_report.ncu-rep

==ERROR== unrecognised option '--output'. Use --help for further details.

==========================================================
Done!
Please open 'roofline_report.ncu-rep' in Nsight Compute GUI.
You will see two dots on the chart:
  1. One hitting the sloped ceiling (Memory Bound)
  2. One hitting the flat ceiling (Compute Bound)
==========================================================
```
#### 技术细节

- **Arithmetic Intensity (AI)**：计算量与数据量的比值，单位是 FLOPs/Byte
- **Grid-Stride Loop**：确保所有线程都有工作，即使 Grid 大小小于数据规模
- **FMA 指令**：融合乘加指令，每个周期可以执行一次乘法和一次加法
- **ILP (Instruction Level Parallelism)**：指令级并行，通过独立的指令流填满流水线

#### 注意事项

- 数据规模需要足够大（建议 ≥ 64MB），以避开 L2 Cache 的影响
- 理论峰值计算需要准确的时钟频率，CUDA 12+ 需要使用 NVML API
- Roofline 模型是性能优化的指导工具，帮助识别瓶颈并指导优化方向
- 实测值通常会低于理论峰值，因为实际代码存在各种开销（调度、同步等）

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
