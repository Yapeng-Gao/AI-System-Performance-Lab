# Module B: 内存优化

本目录是专栏 Module B 的配套实验代码。正文在 `article/02_memory_optim/`。  
仓库整体目录用途见 [`docs/仓库架构与现状.md`](../../docs/仓库架构与现状.md)。

**正文插图**：原理用短 ASCII；B-05～B-08 实测图由 `docs/results/*.csv` + `scripts/plot_b0N_*.py` 生成。

## 目录

| 章节 | 文件 | 核心内容 | 结果 / 重画 |
|------|------|----------|-------------|
| **B-01** | `01_global_mem_bandwidth.cu` | Global Memory | — |
| **B-02** | `02_shared_mem_bank_conflict.cu` | Shared Bank / Swizzle | — |
| **B-03** | `03_register_spill.cu` | Register spill / Occupancy | — |
| **B-04** | `04_l2_residency.cu` | L2 residency | — |
| **B-05** | `05_unified_memory_pf.cu` + `05_profile_*.sh` | UM fault/prefetch/advise | `docs/results/B-05_*`；`python scripts/plot_b05_unified_memory.py` |
| **B-06** | `06_pinned_dma.cu` + `06_profile_*.sh` | Pinned / DMA / Overlap | `docs/results/B-06_*`；`python scripts/plot_b06_pinned_dma.py` |
| **B-07** | `07_cp_async_pipeline.cu` + profile/dump 脚本 | 设备内 async pipeline | `docs/results/B-07_*`；`python scripts/plot_b07_cp_async.py` |
| **B-08** | `08_tma_intro.cu` + profile/dump 脚本 | Hopper+ TMA bulk / tensor-map（**sm_90+**） | `docs/results/B-08_*`；`python scripts/plot_b08_tma.py` |

## 🚀 快速开始

### 编译构建

#### Linux 环境
```bash
# 在项目根目录
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8
```

> 提示：如果你新加了示例文件（例如 `04_l2_residency.cu`），请先重新执行 `cmake ..` 再 build；只执行 `cmake --build .` 不会自动发现新的 target。

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

### 第 13 章：Register Spilling 与 Occupancy (`03_register_spill.cu`)

该示例用于构造寄存器压力并观察 spilling 对性能的影响，核心是对比三种变体：

- **A) baseline（REGS=32）**：寄存器压力较低
- **B) high-reg（REGS=256）**：寄存器压力高，更容易触发 local spill
- **C) launch_bounds（2 blocks/SM 提示）**：演示 occupancy 约束与寄存器分配的权衡

#### 运行示例

```bash
# 默认参数
./bin/02_memory_optim_03_register_spill

# 自定义规模
./bin/02_memory_optim_03_register_spill 1048576 256
```

#### 分析建议

- 编译时加 `-Xptxas=-v`，对比不同变体的 `reg / spill loads / spill stores`
- 运行时对照 kernel 平均时间，结合编译日志判断是否出现“寄存器不够 -> spill -> 性能下降”
- 关注 `launch_bounds` 的双刃剑效应：它是资源契约，不保证一定更快

---

### 第 14 章 / B-04：L2 Residency 控制 (`04_l2_residency.cu`)

该示例用最小 micro-bench 复现五类场景（其中 C0/C1 是同 workload 的公平对照）：

- **A) Streaming（policy off）**：一次性数据流式访问，通常对 residency 不敏感
- **B) Hot Reuse（policy off）**：存在可复用热点，L2 命中提升更明显
- **C0) Mixed baseline（policy off）**：与 C1 完全同一 workload，用于公平对照
- **C1) Mixed + Residency（policy on）**：`set-aside + window + hitRatio` 的可控配置
- **D) Thrashing（policy on）**：`window >> set-aside` 且 `hitRatio=1.0` 的典型翻车配置

#### 运行示例

```bash
# 位置参数（兼容旧用法）
./bin/02_memory_optim_04_l2_residency 64 2048 8 32 0.25

# 命名参数（推荐）
./bin/02_memory_optim_04_l2_residency --data-mb 64 --iters 2048 --set-aside-mb 8 --window-mb 32 --hit-ratio 0.25

# 固定 seed + 仅输出 CSV（适合批量扫参）
./bin/02_memory_optim_04_l2_residency --data-mb 64 --iters 2048 --set-aside-mb 8 --window-mb 32 --hit-ratio 0.25 --seed 12345 --csv-only

# 固定 workload + 多次运行统计（推荐用于判断 C1 vs C0）
./bin/02_memory_optim_04_l2_residency --data-mb 32 --iters 4096 --set-aside-mb 24 --window-mb 16 --hit-ratio 1.0 --hot-ratio 0.25 --runs 7 --seed 12345
```

#### 参数说明

- `--data-mb`：输入数据规模（MB）
- `--iters`：循环次数（放大时延差异）
- `--set-aside-mb`：L2 persisting 预留大小
- `--window-mb`：policy window 大小（会自动截断到设备上限）
- `--hit-ratio`：`[0,1]`，用于描述窗口内“值得持久化”的比例
- `--hot-ratio`：`[0,1]`，用于控制 mixed workload 热点子集比例（与 policy hint 解耦）
- `--runs`：重复运行次数（输出 median/mean，建议 `5~11`）
- `--seed`：控制 mixed workload 的伪随机访问序列，便于复现实验
- `--csv-only`：仅输出一行 CSV，方便脚本采集结果

#### 输出口径

程序会输出 A/B/C0/C1/D 五组时间统计（median + mean），并提供一行 `CSV,time_ms,...` 便于汇总。  
建议配合 Nsight Compute 额外采集 `time + DRAM bytes + L2 hit` 三件套，形成完整证据链。

---

### 第 15 章 / B-05：Unified Memory 行为边界 (`05_unified_memory_pf.cu`)

该示例用于对比 UM 的三种策略：

- `fault`：仅 `cudaMallocManaged`，按需迁移
- `prefetch`：kernel 前执行 `cudaMemPrefetchAsync`
- `advise`：`SetPreferredLocation` + `SetAccessedBy`（**不含** `SetReadMostly`，本 kernel 会写回）

**RTX 5090 参考（n=16777216, iters=32, runs=5, warmup=1）**

| mode | median (ms) | 备注 |
|------|-------------|------|
| fault | 0.221 | 稳态与 prefetch 接近 |
| prefetch | 0.221 | 数据已热在 GPU，prefetch 边际收益小 |
| advise | 0.219 | 修复 ReadMostly 误用后与 fault 同阶 |

> 反例：对可写 UM 使用 `SetReadMostly` 时，同一配置 median 可达 **~124 ms**（约 560× 退化）。

**冷启动（fault, warmup=0, runs=3）**：`first≈29 ms`，`median≈0.23 ms`（首轮 fault 迁移 vs 稳态差 **~120×**）。采集：`WARMUP=0 RUNS=3 bash 05_profile_unified_memory.sh fault`

#### 运行示例

```bash
# fault-only
./bin/02_memory_optim_05_unified_memory_pf --mode fault --n 16777216 --iters 32 --runs 5 --warmup 1

# prefetch
./bin/02_memory_optim_05_unified_memory_pf --mode prefetch --n 16777216 --iters 32 --runs 5 --warmup 1

# advise
./bin/02_memory_optim_05_unified_memory_pf --mode advise --n 16777216 --iters 32 --runs 5 --warmup 1
```

#### 参数说明

- `--n`：元素数量（float）
- `--iters`：kernel 内循环次数
- `--mode`：`fault|prefetch|advise`
- `--runs`：统计轮次
- `--warmup`：不计入统计的预热轮次
- `--device`：目标 GPU 设备号
- `--csv-only`：仅输出 CSV 行，便于批量采集

#### 输出口径

输出包含：`first / median / p95 / mean`。  
建议将三种 mode 在同一输入规模下对照，先看首轮抖动是否显著下降，再结合 NSYS/NCU 做证据闭环。

#### NSYS 一键采集

```bash
cd examples/02_memory_optim
chmod +x 05_profile_unified_memory.sh
bash 05_profile_unified_memory.sh
# 仅采集某一模式：bash 05_profile_unified_memory.sh fault
```

生成 `um_fault_trace.nsys-rep`、`um_prefetch_trace.nsys-rep`、`um_advise_trace.nsys-rep`，在 Nsight Systems 中查看 **UVM page fault / page migration** 时间线。

---

### 第 16 章 / B-06：Pinned Memory 与 DMA (`06_pinned_dma.cu`)

对照六种 Host↔Device 路径：

| mode | 含义 |
|------|------|
| `pageable` | `malloc` + `cudaMemcpyAsync` H2D（伪异步地板） |
| `pinned` | `cudaMallocHost` + Async H2D |
| `serial` | pinned + **1** stream 切块 H2D→Kernel（overlap 公平基线） |
| `overlap` | pinned + **多** stream 切块（跨 chunk 重叠） |
| `bidir` | pinned + H2D∥D2H（合计吞吐） |
| `mapped` | `cudaHostAllocMapped`，kernel 直读 host（有效带宽 ≠ memcpy） |

#### 运行示例

```bash
./bin/02_memory_optim_06_pinned_dma --mode pageable --mb 256 --runs 5
./bin/02_memory_optim_06_pinned_dma --mode pinned   --mb 256 --runs 5
./bin/02_memory_optim_06_pinned_dma --mode serial   --mb 256 --chunk-mb 16 --kernel-iters 8
./bin/02_memory_optim_06_pinned_dma --mode overlap  --mb 256 --chunk-mb 16 --streams 4 --kernel-iters 8
./bin/02_memory_optim_06_pinned_dma --mode bidir    --mb 256 --runs 5
./bin/02_memory_optim_06_pinned_dma --mode mapped   --mb 64  --runs 5
```

批量跑 + 可选 NSYS：

```bash
bash examples/02_memory_optim/06_profile_pinned_dma.sh
DO_NSYS=1 bash examples/02_memory_optim/06_profile_pinned_dma.sh overlap
```

#### NSYS（验证真 overlap）

```bash
nsys profile -o pinned_overlap --force-overwrite true \
  ./bin/02_memory_optim_06_pinned_dma --mode overlap --mb 256 --chunk-mb 16 --streams 4 --kernel-iters 8
```

在时间线确认 **某 chunk 的 Copy** 与 **另一 chunk 的 Kernel** 是否存在重叠窗口。  
判定 overlap 成功（与正文一致）：

1. 端到端快于同参数 `--mode serial`
2. copy-bound 时 `overlap` 逼近同 size `--mode pinned`（哪怕相对 serial 只快几个点）

配套文章：`article/02_memory_optim/B-06*.md`（封面见同目录 `assets/`）。

---

### 第 17 章 / B-07：Async Copy / Pipeline (`07_cp_async_pipeline.cu`)

对照设备侧 GMEM→SMEM 路径（需要 sm_80+）：

| mode | 含义 |
|------|------|
| `sync` | `gmem→reg→smem` 同步基线 |
| `async1` | `memcpy_async` + 立刻 wait（无多 stage overlap） |
| `pipe2` / `pipe4` | thread-local 2/4-stage software pipeline |
| `pipe2_blk` | block-shared 2-stage pipeline（对照 barrier 开销） |
| `sweep` | 扫 `fma-iters`，输出 sync/pipe2/pipe4 加速比 CSV |

#### 运行示例

```bash
./bin/02_memory_optim_07_cp_async_pipeline --mode sync   --fma-iters 8
./bin/02_memory_optim_07_cp_async_pipeline --mode pipe2  --fma-iters 8
./bin/02_memory_optim_07_cp_async_pipeline --mode pipe4  --fma-iters 8
./bin/02_memory_optim_07_cp_async_pipeline --mode sweep
```

批量跑 + 可选 NCU：

```bash
bash examples/02_memory_optim/07_profile_cp_async_pipeline.sh
DO_NCU=1 bash examples/02_memory_optim/07_profile_cp_async_pipeline.sh
```

结果回填：`docs/results/B-07_cp_async_pipeline.md`。配套文章：`article/02_memory_optim/B-07*.md`。

---

### 第 18 章 / B-08：Hopper TMA (`08_tma_intro.cu`)

对照设备侧 TMA 路径（需要 **sm_90+** / Hopper 或 Blackwell）：

| mode | 含义 |
|------|------|
| `sync` | 协作 sync load 整 tile（公平基线） |
| `bulk1d` | 1D `memcpy_async_tx` + mbarrier，立刻 wait |
| `tensor2d` | 2D `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor` |
| `pipe2` | 2-stage 1D TMA prefetch ∥ compute |
| `sweep` | 扫 `fma-iters`，输出 sync/bulk1d/tensor2d/pipe2 加速比 CSV |

#### 运行示例

```bash
# 建议：-DCMAKE_CUDA_ARCHITECTURES=120（RTX 5090）或 90（H100）
./bin/02_memory_optim_08_tma_intro --mode sync     --fma-iters 8
./bin/02_memory_optim_08_tma_intro --mode bulk1d   --fma-iters 8
./bin/02_memory_optim_08_tma_intro --mode tensor2d --fma-iters 8
./bin/02_memory_optim_08_tma_intro --mode pipe2    --fma-iters 8
./bin/02_memory_optim_08_tma_intro --mode sweep
```

批量跑 + 可选 NCU / SASS：

```bash
bash examples/02_memory_optim/08_profile_tma.sh
DO_NCU=1 bash examples/02_memory_optim/08_profile_tma.sh
bash examples/02_memory_optim/08_dump_sass.sh
python scripts/plot_b08_tma.py   # 需先写好 docs/results/B-08_*.csv
```

结果模板：`docs/results/B-08_tma.md`。配套文章：`article/02_memory_optim/B-08*.md`。

---

## 🔧 工具脚本

- `01_profile_bandwidth.sh`：性能分析脚本（Linux/WSL 专用），使用 Nsight Compute 分析 Global Memory 访问模式和带宽利用率
- `02_profile_banks.sh`：Shared Memory Bank Conflict 分析脚本（Linux/WSL 专用），使用 Nsight Compute 采集共享内存 Wavefront 等指标，验证 Naive / Padding / Swizzling 的 Bank Conflict 差异
- `05_profile_unified_memory.sh`：B-05 UM 证据链脚本（Linux/WSL 专用），使用 Nsight Systems 批量采集 fault/prefetch/advise 三组 trace
- `06_profile_pinned_dma.sh`：B-06 Pinned/DMA 批量对照（可选 `DO_NSYS=1` 采集 overlap 时间线）
- `07_profile_cp_async_pipeline.sh`：B-07 设备内 async/pipeline 批量对照（可选 `DO_NCU=1`）
- `07_dump_sass.sh`：B-07 导出 SASS，核对 `LDGSTS` / `CP.ASYNC`
- `08_profile_tma.sh`：B-08 TMA 批量对照（可选 `DO_NCU=1`）
- `08_dump_sass.sh`：B-08 导出 SASS，核对 TMA / BULK 类指令

## 📝 注意事项

### CUDA 版本要求

- **最低要求**：CUDA 12.0+（因为使用了 `cuda::pipeline` API）
- **推荐版本**：CUDA 12.3+ 或 CUDA 13.1+（支持完整特性）；B-08 TMA 推荐 12.4+
- **向后兼容性**：CUDA 13.1 完全向后兼容 CUDA 12.x 的代码和 API
- **架构要求**：
  - Async Copy Pipeline：需要计算能力 8.0+（Ampere 及以上）
  - **TMA（B-08）**：需要计算能力 **9.0+**（Hopper / Blackwell）
  - L2 Persistence：需要计算能力 8.0+（Ampere 及以上）
  - LDG.NT：需要计算能力 7.0+（Volta 及以上）

### 其他注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台

