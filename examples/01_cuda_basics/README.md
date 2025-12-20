# Module A: CUDA 基础架构

本目录包含 CUDA 核心概念和架构相关的示例代码，是《AI 系统性能工程》专栏 Module A 的配套实战代码。

## 📚 目录

| 章节 | 文件 | 核心内容 | 知识点 |
|------|------|----------|--------|
| **第 1 章** | `01_hello_modern.cu` | CUDA 核心概念总览 | Grid-Block-Thread 模型、错误检查、异步执行、Unified Memory |
| **第 2 章** | `02_hardware_query.cu` | GPU 硬件架构深度解析 | SM 架构、内存层次、L2 Cache、Tensor Core 能力、带宽分析 |
| **第 3 章** | `03_grid_mapping.cu` | CUDA 编程模型物理映射 | GigaThread Engine 调度、SM 映射、Wavefront 效应、PTX 内联汇编 |
| **第 4 章** | `04_warp_divergence.cu` | 线程调度：SIMT, Divergence 与 Replay | Warp 发散、Bank Conflict、指令重播、性能量化 |
| **第 5 章** | `05_kernel_structure.cu` | Kernel 结构与 ABI 分析 | 结构体对齐陷阱、函数内联控制、Launch Bounds 优化 |
| **第 6 章** | `06_nvrtc_jit.cpp` | NVRTC 运行时编译与 Driver API | 运行时特化、PTX 动态加载、架构自适应 |
| **第 7 章** | `07_memory_spaces.cu` | 内存模型全景 | 地址空间探测、UVA Zero-Copy、Local Memory Spilling、__restrict__ 优化 |
| **第 8 章** | `08_async_pipeline.cu` | 异步执行模型 | Pinned Memory、多 Stream 并发、Depth-First 调度、流水线 Overlap |
| **第 9 章** | `09_debug_and_sanitizer.cu` | 调试与错误诊断 | Compute Sanitizer、内存越界检测、数据竞争检测、非法同步检测 |

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

# Windows/CLion: 在 cmake-build-debug/bin 目录下运行
# 或在 PowerShell 中（从项目根目录）
.\cmake-build-debug\bin\01_cuda_basics_01_hello_modern.exe
.\cmake-build-debug\bin\01_cuda_basics_06_nvrtc_jit.exe
.\cmake-build-debug\bin\01_cuda_basics_07_memory_spaces.exe
.\cmake-build-debug\bin\01_cuda_basics_08_async_pipeline.exe
.\cmake-build-debug\bin\01_cuda_basics_09_debug_and_sanitizer.exe
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

```
[Host] Starting Modern CUDA Hello World...
[Host] GPU Name: NVIDIA GeForce RTX 4090
[Host] SM Count: 128
[Host] Compute Capability: 8.9
[Host] Launching Kernel...
[Device] Kernel running on SM arch sm_890
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

这可以验证 CMake 配置中的 `CMAKE_CUDA_ARCHITECTURES` 是否生效。

---

### 第 2 章：GPU 硬件架构深度解析 (`02_hardware_query.cu`)

**硬件拓扑侦探**：挖掘 SM 架构、L2 Cache、Tensor Core 能力与带宽极限。

#### 核心知识点

1. **计算能力与架构识别**：根据 Compute Capability 识别 GPU 架构（Hopper、Ampere、Volta 等），并推算每个 SM 的 CUDA Core 数量。

2. **SM 宏观拓扑**：展示多处理器数量、CUDA Core 总数等宏观计算资源。

3. **内存体系分析**：
   - 全局内存容量（HBM/DDR）
   - 内存总线宽度
   - 理论峰值带宽计算
   - **L2 Cache 大小**（关键性能指标，影响数据驻留控制）

4. **SM 微架构资源**（Occupancy 分析关键）：
   - Shared Memory 限制（每 Block 和动态分配）
   - 寄存器数量限制
   - 线程数限制（每 Block 和每 SM）
   - Warp 大小

5. **现代特性支持侦测**：
   - Unified Addressing (UVA)
   - Managed Memory (Page Migration)
   - TMA (Tensor Memory Accelerator) - Hopper 架构
   - Thread Block Clusters - Hopper 架构

#### 预期输出

```
=================================================================
   AI System Performance Lab - Hardware Topology Detective   
=================================================================
Detected 1 CUDA Capable Device(s)

[Device 0]: NVIDIA GeForce RTX 4090
-----------------------------------------------------------------
  [Architecture]
    Compute Capability      : 8.9 (Ampere / Ada class)
  [Compute Topology]
    Multiprocessors (SMs)   : 128
    CUDA Cores / SM         : 128
    Total CUDA Cores        : 16384
    GPU Clock Rate          : N/A (removed in CUDA 12+)
  [Memory Hierarchy]
    Global Memory (HBM/DDR) : 24.00 GB
    Memory Bus Width        : 384-bit
    Memory Clock Rate       : N/A (removed in CUDA 12+)
    Theoretical Bandwidth   : N/A (use nvml API for accurate value)
    L2 Cache Size           : 72.00 MB (Key for residency control)
  [SM Micro-Architecture]
    Max Shared Mem / Block  : 48.00 KB
    Max Shared Mem (Opt-in) : 164.00 KB (Dynamic)
    Max Registers / Block   : 65536
    Max Threads / Block     : 1024
    Max Threads / SM        : 1536
    Warp Size               : 32
  [Modern Features Support]
    Unified Addressing      : Yes
    Managed Memory          : Yes
    TMA (Tensor Mem Accel)  : No
    Thread Block Clusters   : No
```

#### 注意事项

- 在 CUDA 12+ 版本中，`clockRate` 和 `memoryClockRate` 字段已被移除，代码使用条件编译兼容新旧版本。
- 如需获取准确的时钟频率和带宽信息，建议使用 NVML (NVIDIA Management Library) API。

---

### 第 3 章：CUDA 编程模型物理映射 (`03_grid_mapping.cu`)

**Grid Mapper**：可视化 GigaThread Engine 的调度逻辑与物理 SM 映射。

#### 核心知识点

1. **PTX 内联汇编**：使用 `asm volatile("mov.u32 %0, %smid;")` 直接读取硬件特殊寄存器 `%smid`，获取 Block 实际运行的物理 SM ID。这是比 CUDA C++ API 更底层的操作，兼容所有架构。

2. **执行顺序追踪**：通过 Global Atomic 操作（`atomicAdd`）追踪 Block 的真实执行顺序（Execution Order），验证 GigaThread Engine 的调度策略。

3. **Wavefront 效应观察**：
   - 通过模拟计算负载（Busy Wait）拉长 Block 执行时间
   - 观察多波次（Wave）调度模式
   - 检测尾部效应（Tail Effect）：最后一个 Block 可能独占 GPU

4. **负载均衡分析**：
   - 统计每个 SM 处理的 Block 数量
   - 验证 Round-Robin 分配策略
   - 可视化 Block ID 到 SM ID 的映射关系

#### 预期输出

```
[Host] Starting Grid Scheduler Tracer...
[Host] GPU: NVIDIA GeForce RTX 4090, Total SMs: 128
[Host] Launching 2561 Blocks (approx 2560 full waves + 1 tail)

[Analysis 1] SM Load Balance (Top 5 & Bottom 5):
  SM 00 processed 20 blocks
  SM 01 processed 20 blocks
  ...

[Analysis 2] Tail Effect Detection:
  The very last block to run was logical Block 2560
  It ran on physical SM 0
  Note: While this block was running, other SMs might have been IDLE if the grid size wasn't aligned to waves.

[Visualizer] Logical Block ID -> Physical SM ID (First 64 Blocks):
  Blocks 000-015:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  ...

[Conclusion] GigaThread Engine successfully distributed work across ALL 128 SMs. ✅
```

#### 技术细节

- **SM ID 读取**：使用 PTX 汇编直接访问硬件寄存器，比软件 API 更底层、更准确
- **原子操作**：`atomicAdd` 保证执行顺序的全局一致性，用于追踪调度顺序
- **时钟计数**：使用 `clock64()` 记录 Block 启动时间戳，可用于分析调度延迟

---

### 第 4 章：线程调度：SIMT, Divergence 与 Replay (`04_warp_divergence.cu`)

**Micro-benchmark**：量化分支发散与 Shared Memory Bank Conflict 的物理代价。

#### 核心知识点

1. **Warp Divergence（分支发散）**：
   - **Baseline 模式**：所有线程执行相同路径，SIMT 单元满载运行
   - **Divergent 模式**：奇偶线程走不同分支，硬件串行化执行（先执行偶数线程，再执行奇数线程）
   - **性能影响**：理论吞吐量减半（理想情况 2.0x 性能损失）

2. **Bank Conflict（存储体冲突）**：
   - **无冲突访问**：Stride=1，32 个线程访问不同 Bank，1 个周期完成
   - **32-way 冲突**：Stride=32，所有线程访问同一 Bank，指令重播 32 次
   - **性能影响**：理论延迟增加 32 倍（理想情况 32.0x 性能损失）

3. **性能测量技术**：
   - 使用 `clock64()` 进行高精度周期计数
   - 通过 `#pragma unroll` 减少循环开销，突出被测操作
   - 使用 `volatile` 防止编译器优化掉内存访问

4. **编译器优化防护**：
   - 使用 `volatile` 变量防止编译器优化
   - 通过死代码路径（`if (val == 999999.0f)`）防止死代码消除

#### 预期输出

```
=================================================================
   AI System Performance Lab - SIMT & Replay Analyzer   
=================================================================
Running on GPU: NVIDIA GeForce RTX 4090 (Arch sm_89)

[Experiment 1] Measuring Warp Divergence Cost (ALU)
  Baseline (No Branch) Cycles : 2000
  Divergent (If-Else) Cycles  : 4000
  >> Performance Penalty      : 2.00x Slower (Ideal: 2.0x)

[Experiment 2] Measuring Instruction Replay Cost (Shared Mem)
  Linear Access (No Conflict) : 1000 cycles
  Stride-32 (32-way Conflict) : 32000 cycles
  >> Replay Penalty           : 32.00x Slower (Ideal: 32.0x)

Note: 'Ideal' assumes pure isolation. Real hardware pipelines may hide some latency.
```

#### 实验设计

- **实验 1：Math Divergence**
  - 对比无分支代码 vs 奇偶分支代码
  - 验证 ALU 利用率减半（SIMT 串行化）
  - 使用浮点运算制造计算负载

- **实验 2：Bank Conflict Replay**
  - 对比无冲突访问 vs 32-way Bank Conflict
  - 验证指令重播机制
  - 使用 Shared Memory 读后写操作制造依赖链

#### 注意事项

- 实际硬件流水线可能会隐藏部分延迟，因此实测值可能略低于理想值
- `clock64()` 返回的是 SM 时钟周期，不是绝对时间
- 循环次数（`PER_KERNEL_ITERS`）需要足够大以摊薄测量开销，但不要触发 TDR（超时检测与恢复）

---

---

### 第 5 章：Kernel 结构与 ABI 分析 (`05_kernel_structure.cu`)

**ABI 深度解析**：揭示 Host-Device 边界上的陷阱与优化技巧。

#### 核心知识点

1. **结构体对齐陷阱**：
   - Host 编译器（GCC/MSVC）和 Device 编译器（NVCC）可能采用不同的 Padding 策略
   - `__align__` 关键字强制对齐，避免结构体布局不一致导致的 Bug
   - 演示不同对齐策略对内存布局的影响

2. **函数内联控制**：
   - `__noinline__`：强制函数不被内联，用于调试和 ABI 分析
   - `__forceinline__`：强制内联，消除函数调用开销
   - 使用 `cuobjdump` 验证内联行为（查找 CAL 指令）

3. **Launch Bounds 优化**：
   - `__launch_bounds__(MAX_THREADS, MIN_BLOCKS)` 提示编译器优化寄存器使用
   - 影响 Occupancy 和寄存器分配策略
   - 通过 `-Xptxas=-v` 查看寄存器使用情况

#### 预期输出

```
==================================================================
   AI System Performance Lab - Kernel Structure Analyzer   
==================================================================
[Part 1] Testing Struct Alignment:
  DangerousStruct size (Host): 12 bytes
  DangerousStruct size (Device): 12 bytes
  SafeStruct size (Host): 16 bytes (aligned)
  SafeStruct size (Device): 16 bytes (aligned)

[Part 2] Testing Inline Behavior:
  Use 'cuobjdump -sass' to inspect CAL (Call) instructions
  __noinline__ functions should appear as CAL instructions

[Part 3] Testing Launch Bounds:
  Kernel without __launch_bounds__ uses: 32 registers
  Kernel with __launch_bounds__(256, 4) uses: 24 registers
  >> Register pressure reduced, Occupancy improved!
```

#### ABI 分析工具

项目提供了 `05_inspect_asm.sh` 脚本，用于分析 SASS 代码中的内联行为：

```bash
cd examples/01_cuda_basics
bash 05_inspect_asm.sh
```

**注意**：脚本会自动检测构建目录（支持 Windows/CLion 和 Linux 两种构建方式）。

该脚本会：
- 搜索 SASS 代码中的 `CAL`（Call）指令
- 验证 `__noinline__` 函数是否真的没有内联
- 分析函数调用的实际行为

#### 注意事项

- 结构体对齐问题在实际项目中可能导致难以调试的 Bug
- 函数内联会影响调试能力，但可以提升性能
- `__launch_bounds__` 需要根据实际 Occupancy 需求调整参数

---

### 第 6 章：NVRTC 运行时编译与 Driver API (`06_nvrtc_jit.cpp`)

**运行时特化 + 动态加载**：使用 NVRTC 在运行时生成 PTX，并通过 Driver API 加载执行。

#### 核心知识点

1. **运行时特化**：在 Host 端将常量（如 `scale=5.0f`）写入 Kernel 源码字符串，触发编译器常量折叠。  
2. **架构自适应**：运行时获取当前 GPU 的 Compute Capability，生成对应 `--gpu-architecture=compute_XY`，避免旧卡（如 1050, sm_61）出现 `CUDA_ERROR_INVALID_PTX`。  
3. **混合 API**：NVRTC（Runtime Compilation）+ Driver API（`cuModuleLoadData` / `cuLaunchKernel`）+ Runtime API（`cudaMalloc` / `cudaMemcpy`）混用。  
4. **日志与错误处理**：拉取 NVRTC 编译日志，Fail Fast。  

#### 运行

```bash
# Linux
./bin/01_cuda_basics_06_nvrtc_jit

# Windows (PowerShell)
.\cmake-build-debug\bin\01_cuda_basics_06_nvrtc_jit.exe
```

#### 预期输出（老卡示例：GTX 1050, sm_61）

```
[Host] Starting NVRTC JIT Compilation Demo...
[NVRTC] Specialized Source Code generated:
   out[i] = 5.0f * x[i] + y[i];
[NVRTC] PTX generated (... bytes).
[Host] Verification PASSED! Result is 7.0
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

```
[Host] Starting Memory Hierarchy Analysis...
[Host] Launching Address Probe...

[Device] === Memory Address Map ===
  Global Memory (HBM) Ptr:    0x7f8a00000000
  Global Variable (Static):   0x7f8a00001000
  Shared Memory (SRAM):       0x7f8a00000000 (Small offset usually)
  Local Variable (Stack):     0x7f8a00000000 (If address taken -> Local Mem)
  Host Pinned Ptr (UVA/PCIe): 0x7f8a00002000
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
GPU: NVIDIA GeForce RTX 4090
Data Size: 32.00 MB, Chunk Size: 1.00 MB

[Serial] Starting processing 32 chunks...
[Serial] Total Time: 245.67 ms
------------------------------------------------
[Pipeline] Starting processing 32 chunks with 4 streams...
[Pipeline] Total Time: 89.23 ms
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

```
==========================================================
   CASE 1: Detecting Out-of-Bounds Access (Memcheck)
==========================================================
========= COMPUTE-SANITIZER
========= Error: out of bounds access
=========     at 0x... in oob_kernel
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x... is out of bounds
...

==========================================================
   CASE 2: Detecting Data Race (Racecheck)
==========================================================
========= COMPUTE-SANITIZER
========= Error: Race reported between Read access at ...
=========     at 0x... in race_kernel
=========     by thread (1,0,0) in block (0,0,0)
=========     and Write access at ...
=========     by thread (0,0,0) in block (0,0,0)
...

==========================================================
   CASE 3: Detecting Illegal Sync (Synccheck)
==========================================================
========= COMPUTE-SANITIZER
========= Error: Barrier divergence detected
=========     at 0x... in illegal_sync_kernel
=========     Barrier reached by 16 threads, expected 32
...
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

## 🔧 工具脚本

- `01_fatbin_inspect.sh`：二进制文件分析工具，用于查看 PTX 和 SASS 代码
- `05_inspect_asm.sh`：SASS 汇编分析工具，用于验证函数内联行为
- `07_inspect_sass.sh`：SASS 内存分析工具，用于检测 Local Memory Spilling 和 `__restrict__` 优化效果
- `08_profile_nsys.sh`：性能分析脚本（Linux/WSL 专用），使用 Nsight Systems 分析异步流水线性能
- `09_run_sanitizer.sh`：调试工具脚本，使用 Compute Sanitizer 检测内存越界、数据竞争和非法同步

## 📝 注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台
- 兼容 CUDA 12.0+ 版本（部分字段在 CUDA 12+ 中已移除，使用条件编译处理）
