# Module B: 内存优化

本目录包含 CUDA 内存优化相关的示例代码，是《AI 系统性能工程》专栏 Module B 的配套实战代码。

## 📚 目录

| 章节 | 文件 | 核心内容 | 知识点 |
|------|------|----------|--------|
| **第 11 章** | `01_global_mem_bandwidth.cu` | Global Memory 极致优化 | 物理层（对齐）、指令层（向量化/Async Copy）、缓存层（LDG.NT/L2 驻留） |
| **第 12 章** | `02_shared_mem_bank_conflict.cu` | Shared Memory Bank Conflict 分析 | Bank Conflict、Padding、XOR Swizzling |

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
- **Windows/CLion**: `cmake-build-debug/bin` 或 `cmake-build-debug-visual-studio/bin` 目录

```bash
# Linux: 在 build 目录下运行
./bin/02_memory_optim_01_global_mem_bandwidth

# Windows/CLion: 在 cmake-build-debug/bin 目录下运行
# 或在 PowerShell 中（从项目根目录）
.\cmake-build-debug\bin\02_memory_optim_01_global_mem_bandwidth.exe
```

---

## 📖 各章详细说明

### 第 11 章：Global Memory 极致优化 (`01_global_mem_bandwidth.cu`)

**Bandwidth Micro-Benchmark**：覆盖物理层、指令层与缓存层的所有优化手段，全面验证 Global Memory 性能优化技术。

#### 核心知识点

本示例包含 5 个测试项，从不同层次展示内存优化技术：

1. **[物理层] Misaligned Access（错位访问）**：
   - 故意制造错位访问（Offset=1），破坏 Coalescing 机制
   - 一个 Warp 的请求会分裂成多个 Memory Transactions
   - 导致带宽利用率大幅下降
   - 演示了内存对齐对性能的严重影响

2. **[指令层] Vectorized Copy（向量化拷贝）**：
   - 使用 `float4` 类型，强制生成 `LDG.E.128` 指令
   - 减少 75% 的指令发射压力
   - 提升内存访问效率，接近硬件带宽上限

3. **[缓存层] LDG.NT (Non-Temporal Load)**：
   - 使用 `__ldcs` 内建函数生成 `LDG.E.128.STREAM` 指令
   - 告诉硬件：这条数据读完就扔，不占用 L2 Cache 位置
   - 适合流式数据（Streaming Data）场景，避免污染 L2 Cache
   - 在数据只读一次的场景下，可以提升整体缓存命中率

4. **[架构层] Async Copy Pipeline（异步拷贝流水线）**：
   - 使用 CUDA 12+ 的 `cuda::pipeline` 和 `cuda::memcpy_async` API
   - 在 Ampere 架构上映射为 `cp.async` 指令，在 Hopper 上为 TMA 铺路
   - 实现 Global Memory → Shared Memory 的异步传输，绕过寄存器
   - 演示了现代 CUDA 编程中的 Pipeline 模式，适合 GEMM 等计算密集型 Kernel

5. **[缓存层] L2 Persistence（L2 缓存驻留控制）**：
   - 使用 CUDA 11.0+ 的 `cudaStreamSetAttribute` API
   - 通过 `cudaAccessPropertyPersisting` 锁定 L2 Cache
   - 模拟深度学习中的 Weight Reuse 场景
   - 对比默认 LRU 策略与显式驻留策略的性能差异

#### 预期输出

```
GPU: NVIDIA GeForce RTX 4090 | L2: 72.00 MB
Theoretical Bandwidth: 1008.00 GB/s

[1. Misaligned         ]  548.72 GB/s
[2. Vectorized float4 ]  883.45 GB/s
[3. LDG.NT (Stream)    ]  875.23 GB/s
[4. Async Copy         ]  890.12 GB/s

=== L2 Persistence Test (20MB Data, 50 Repeats) ===
[L2 Default (LRU)     ]  224.72 GB/s
[L2 Persisting         ]  444.44 GB/s
>> Improvement: 97.78%
```

#### 性能分析工具

项目提供了 `01_profile_bandwidth.sh` 脚本，使用 Nsight Compute 进行内存性能分析：

```bash
cd examples/02_memory_optim
bash 01_profile_bandwidth.sh
```

**注意**：
- 脚本会自动检测构建目录（支持 Windows/CLion 和 Linux 两种构建方式）
- **需要安装 Nsight Compute**（随 CUDA Toolkit 一起安装）
- 脚本会生成 `.ncu-rep` 文件，需要在 Nsight Compute GUI 中打开

该脚本可以：
- **验证向量化效果**：在 "SOL Memory" 部分查看向量化访问的效率
- **分析内存事务**：在 "Memory Analysis" 中查看 "Sectors/Request" 指标
- **对比不同访问模式**：观察 Misaligned vs Aligned vs Vectorized 的性能差异

#### 技术细节

- **Coalescing 机制**：当 Warp 中的线程访问连续对齐的内存时，硬件会将多个请求合并成一个 Memory Transaction
- **向量化访问**：使用 `float4` 等向量类型可以减少指令数量，提升指令发射效率
- **Non-Temporal Load**：`__ldcs` 内建函数生成流式加载指令，适合一次性读取的数据，避免占用 L2 Cache
- **Async Copy Pipeline**：`cuda::pipeline` 是 CUDA 12.0+ 引入的现代异步拷贝 API，支持多阶段流水线操作
- **L2 Cache 驻留**：通过 `cudaStreamSetAttribute` 可以控制 L2 Cache 的替换策略，适合权重重用场景
- **数据规模**：使用 64MB 数据确保足够大以触达 HBM 带宽墙，避开 L2 Cache 的影响

#### 注意事项

- **Misaligned Access** 是性能杀手，应尽量避免非对齐的内存访问
- **Vectorized Access** 可以显著提升性能，但需要确保数据对齐（128-byte 对齐）
- **LDG.NT** 适合流式数据场景，如果数据会被重复访问，使用普通加载可能更好
- **Async Copy Pipeline** 需要 CUDA 12.0+ 和计算能力 8.0+（Ampere 及以上架构）
- **L2 Persistence** 功能需要 CUDA 11.0+ 和计算能力 8.0+（Ampere 及以上架构）
- 实际带宽会受到硬件限制、PCIe 带宽等多种因素影响，实测值可能低于理论峰值

---

### 第 12 章：Shared Memory Bank Conflict 深度优化 (`02_shared_mem_bank_conflict.cu`)

**Shared Memory Bank Conflict Analyzer**：通过微基准测试对比 Naive / Padding / XOR Swizzling 三种访问方式下的 Bank Conflict 情况和性能差异。

#### 核心知识点

- **Naive 访问模式（32-way Conflict）**：
  - 使用标准布局 `__shared__ float tile[32][32]`
  - Warp 内 32 个线程按“列”访问同一 Bank，造成严重 32-way Bank Conflict
  - 实测 Cycles 显著放大，用于对比基线
- **Padding 访问模式（空间换时间）**：
  - 使用布局 `__shared__ float tile[32][33]`
  - 每行起始地址相差 33 个 word，满足 \(33 \bmod 32 = 1\)
  - 行与行的起始 Bank 轮转，从而消除 Conflict
- **XOR Swizzling（现代 Tensor Core 常用技巧）**：
  - 保持紧凑布局 `__shared__ float tile[32][32]`
  - 使用 `physical_col = logical_col ^ logical_row` 映射逻辑列到物理 Bank
  - 在不增加额外空间的前提下消除 Bank Conflict

#### 预期输出

运行可执行文件 `02_memory_optim_02_shared_mem_bank_conflict`（或 `.exe`）时，预期会看到类似输出（Cycles 仅供参考）：

```
[Naive]   32-way Conflict Cycles: 12345678
[Padding] Conflict-Free Cycles  : 456789
[Swizzle] XOR Pattern Cycles    : 432109

=== Performance Gain ===
Padding Speedup : 20.00x
Swizzle Speedup : 18.50x
>> Result: SUCCESS. Heavy conflicts detected and resolved.
```

#### 技术细节

- **Bank Conflict 本质**：一个 Warp 内多个线程同时访问同一 Bank，会被硬件拆分为多次序列化访问，导致实际 Latency 放大
- **Padding 技巧**：通过让每行起始地址在 Bank 空间中“轮转”，让同一列访问时落在不同 Bank
- **XOR Swizzling**：通过简单的 `XOR` 运算将 (row, col) 重新映射为物理地址，是现代 GEMM/Tensor Core Kernel 中常见的 shared memory 优化手段
- **clock64 计时**：用 `clock64()` 在设备端统计循环内的访问延迟，用于放大并对比不同模式下的 Cycles 差异

#### 注意事项

- 该示例默认单 Block、32 线程，方便观察单 Warp 级别的 Bank Conflict 行为
- 实际大规模 Kernel 中还需考虑多 Warp、多 Block 之间的调度与占用情况

---

## 🔧 工具脚本

- `01_profile_bandwidth.sh`：性能分析脚本（Linux/WSL 专用），使用 Nsight Compute 分析 Global Memory 访问模式和带宽利用率
- `02_profile_banks.sh`：Shared Memory Bank Conflict 分析脚本（Linux/WSL 专用），使用 Nsight Compute 采集共享内存 Wavefront 等指标，验证 Naive / Padding / Swizzling 的 Bank Conflict 差异

## 📝 注意事项

### CUDA 版本要求

- **最低要求**：CUDA 12.0+（因为使用了 `cuda::pipeline` API）
- **推荐版本**：CUDA 12.3+ 或 CUDA 13.1+（支持完整特性）
- **向后兼容性**：CUDA 13.1 完全向后兼容 CUDA 12.x 的代码和 API
- **架构要求**：
  - Async Copy Pipeline：需要计算能力 8.0+（Ampere 及以上）
  - L2 Persistence：需要计算能力 8.0+（Ampere 及以上）
  - LDG.NT：需要计算能力 7.0+（Volta 及以上）

### 其他注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台

