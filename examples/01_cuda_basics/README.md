# Module A: CUDA 基础架构

本目录包含 CUDA 核心概念和架构相关的示例代码，是《AI 系统性能工程》专栏 Module A 的配套实战代码。

## 📚 目录

| 章节 | 文件 | 核心内容 | 知识点 |
|------|------|----------|--------|
| **第 1 章** | `01_hello_modern.cu` | CUDA 核心概念总览 | Grid-Block-Thread 模型、错误检查、异步执行、Unified Memory |
| **第 2 章** | `02_hardware_query.cu` | GPU 硬件架构深度解析 | SM 架构、内存层次、L2 Cache、Tensor Core 能力、带宽分析 |
| **第 3 章** | `03_grid_mapping.cu` | CUDA 编程模型物理映射 | GigaThread Engine 调度、SM 映射、Wavefront 效应、PTX 内联汇编 |
| **第 4 章** | `04_warp_divergence.cu` | 线程调度：SIMT, Divergence 与 Replay | Warp 发散、Bank Conflict、指令重播、性能量化 |

## 🚀 快速开始

### 编译构建

```bash
# 在项目根目录
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8
```

### 运行示例

编译成功后，可执行文件位于 `build/bin` 目录下：

```bash
# 运行第 1 章示例
./bin/01_cuda_basics_01_hello_modern

# 运行第 2 章示例
./bin/01_cuda_basics_02_hardware_query

# 运行第 3 章示例
./bin/01_cuda_basics_03_grid_mapping

# 运行第 4 章示例
./bin/01_cuda_basics_04_warp_divergence
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

## 🔧 工具脚本

- `01_fatbin_inspect.sh`：二进制文件分析工具，用于查看 PTX 和 SASS 代码

## 📝 注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台
- 兼容 CUDA 12.0+ 版本（部分字段在 CUDA 12+ 中已移除，使用条件编译处理）
