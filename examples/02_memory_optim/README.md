# Module B: 内存优化

本目录是专栏 Module B 的配套实验代码。正文在 `article/02_memory_optim/`。  
仓库整体目录用途见 [`docs/仓库架构与现状.md`](../../docs/仓库架构与现状.md)。

**正文插图**：原理用短 ASCII；B-01～B-04、B-05～B-09 实测图由 `docs/results/*.csv` + `scripts/plot_b0N_*.py` 生成。

## 目录

| 章节 | 文件 | 核心内容 | 结果 / 重画 |
|------|------|----------|-------------|
| **B-01** | `01_global_mem_bandwidth.cu` | 合并 / 对齐 / float4（`--mode modes`） | `docs/results/B-01_*`；`python scripts/plot_b01_global_mem.py` |
| **B-02** | `02_shared_mem_bank_conflict.cu` | Bank / Padding / XOR（`--mode modes`） | `docs/results/B-02_*`；`python scripts/plot_b02_shared_mem.py` |
| **B-03** | `03_register_spill.cu` | Spilling / Occupancy / launch_bounds（`--mode modes`） | `docs/results/B-03_*`；`python scripts/plot_b03_register.py` |
| **B-04** | `04_l2_residency.cu` | L2 persisting / window / hitRatio（`--mode modes`） | `docs/results/B-04_*`；`python scripts/plot_b04_l2.py` |
| **B-05** | `05_unified_memory_pf.cu` + `05_profile_*.sh` | UM fault/prefetch/advise | `docs/results/B-05_*`；`python scripts/plot_b05_unified_memory.py` |
| **B-06** | `06_pinned_dma.cu` + `06_profile_*.sh` | Pinned / DMA / Overlap | `docs/results/B-06_*`；`python scripts/plot_b06_pinned_dma.py` |
| **B-07** | `07_cp_async_pipeline.cu` + profile/dump 脚本 | 设备内 async pipeline | `docs/results/B-07_*`；`python scripts/plot_b07_cp_async.py` |
| **B-08** | `08_tma_intro.cu` | Hopper+ TMA bulk / tensor-map（**sm_90+**） | `docs/results/B-08_*`；`python scripts/plot_b08_tma.py` |
| **B-09** | `09_layout_transform.cu` | AoS/SoA + tiled transpose（**不限 sm_90+**） | `docs/results/B-09_*`；`python scripts/plot_b09_layout.py` |
| **B-10** | —（无新 `.cu`） | Module B Checklist：症状→证据→处方 | `docs/results/B-10_checklist.md`；正文 `article/02_memory_optim/B-10*.md` |

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
# Linux: 在 build 目录下运行（主结论）
./bin/02_memory_optim_01_global_mem_bandwidth --mode modes

# Windows/CLion
.\cmake-build-debug\bin\02_memory_optim_01_global_mem_bandwidth.exe --mode modes
```

---

## 📖 各章详细说明

### 第 11 章 / B-01：Global Memory（`01_global_mem_bandwidth.cu`）

**合并 / 对齐 / float4** micro-bench（**不做** async / L2 persistence / TMA 主测）。

| mode | 含义 |
|------|------|
| `misaligned` | 读侧 `offset`（默认 1 float）错位 |
| `aligned` | 对齐连续 float copy（基线） |
| `float4` | 显式向量化（同 useful R+W） |
| `ldg_nt` | `float4` + `__ldcs`（可选；`--mode ldg_nt` 或 `--with-ldg-nt`） |
| `modes` | **主结论**：三档 median + 相对 aligned 加速比 + CSV |

#### 运行示例

```bash
# 主证据
./bin/02_memory_optim_01_global_mem_bandwidth --mode modes

# 可选 ldg_nt 行
./bin/02_memory_optim_01_global_mem_bandwidth --mode modes --with-ldg-nt

# 单 mode 调试
./bin/02_memory_optim_01_global_mem_bandwidth --mode aligned
./bin/02_memory_optim_01_global_mem_bandwidth --mode misaligned --offset 1
./bin/02_memory_optim_01_global_mem_bandwidth --mode float4

# CSV 有数据后重画
python scripts/plot_b01_global_mem.py
```

结果：`docs/results/B-01_global_mem.md`、`B-01_modes.csv`。配套文章：`article/02_memory_optim/B-01. Global Memory：合并访问、对齐与 float4——先修有效带宽再谈异步.md`。

口径：CUDA event **median**；绝对 GB/s 可能含 L2 → **主看相对 `aligned` 加速比**。

可选旁证（旧脚本，已改为传 `--mode modes`）：

```bash
bash examples/02_memory_optim/01_profile_bandwidth.sh
```

**注意**：不要把 ncu 附着时程序自打印的 ms/GB/s 当结论。

---

### 第 12 章 / B-02：Shared Memory（`02_shared_mem_bank_conflict.cu`）

同一 32×32 tile 的**列访问**三档对照（**不做** TMA / `cp.async` / `ldmatrix`）。swizzle 是 bank 处方，不是 Tensor Core 前奏。

| mode | 含义 |
|------|------|
| `naive` | `tile[32][32]` 列扫（32-way 基线） |
| `padding` | `tile[32][33]`，行跨度与 32 互质 |
| `swizzle` | 紧凑 `[32][32]`，读 `tile[row][col ^ row]` |
| `modes` | **主结论**：三档 median + 相对 naive 加速比 + CSV |

#### 运行示例

```bash
# 主证据
./bin/02_memory_optim_02_shared_mem_bank_conflict --mode modes

# 单 mode 调试
./bin/02_memory_optim_02_shared_mem_bank_conflict --mode naive
./bin/02_memory_optim_02_shared_mem_bank_conflict --mode padding
./bin/02_memory_optim_02_shared_mem_bank_conflict --mode swizzle

# CSV 有数据后重画
python scripts/plot_b02_shared_mem.py
```

结果：`docs/results/B-02_shared_mem.md`、`B-02_modes.csv`。配套文章：`article/02_memory_optim/B-02*.md`。

口径：CUDA event **median**；加速比 = `naive_median / mode_median`。默认 `grid=2048`、`block=32`、`iters=8192`。墙钟加速比常小于教科书 32×（init / barrier / 流水线掩盖），主看相对 `naive` 的形状。

**RTX 5090 参考**：`naive` 1.258 ms；`padding` **14.12×**；`swizzle` **12.41×**。

可选旁证（忽略附着时程序自打印的 ms）：

```bash
bash examples/02_memory_optim/02_profile_banks.sh
```

---

### 第 13 章 / B-03：寄存器（`03_register_spill.cu`）

同一压力核三档对照（**不做** L2 / TMA / `cp.async`）。Occupancy 不是 KPI；`__launch_bounds__` 是驻留契约。

| mode | 含义 |
|------|------|
| `baseline` | `REGS=32` 低压力对照 |
| `highreg` | `REGS=256`，推 spill / 掉 occupancy |
| `launch_bounds` | 同一 256 压力 + `__launch_bounds__(256, 2)` |
| `modes` | **主结论**：三档 median + 相对 baseline 加速比 + regs/local/occ CSV |

#### 运行示例

```bash
# 主证据
./bin/02_memory_optim_03_register_spill --mode modes

# 单 mode 调试
./bin/02_memory_optim_03_register_spill --mode baseline
./bin/02_memory_optim_03_register_spill --mode highreg
./bin/02_memory_optim_03_register_spill --mode launch_bounds

# CSV 有数据后重画
python scripts/plot_b03_register.py
```

结果：`docs/results/B-03_register.md`、`B-03_modes.csv`。配套文章：`article/02_memory_optim/B-03. 寄存器：Spilling、Occupancy 与 launch_bounds.md`。

口径：CUDA event **median**；加速比 = `baseline_median / mode_median`。同一次运行打印 `numRegs` / `localSizeBytes` / `occ_blocks`。可选旁证：编译加 `-Xptxas=-v`。

**RTX 5090 参考**：`highreg` **0.308×**；三档 `regs=19`、`occ=6`；`localB` 128→1024。墙钟跟 local 走，不是 occupancy 台阶。

---

### 第 14 章 / B-04：L2（`04_l2_residency.cu`）

同一 mixed 核上对照 policy off / persist / thrash（**不做** TMA / `cp.async` / UM）。需要 **sm_80+** 且 `persistingL2CacheMaxSize>0`。

| mode | 含义 |
|------|------|
| `streaming` | 扫过大数组，policy off |
| `hot` | 小工作集反复读，policy off |
| `mixed` | 热/冷混合，policy off（公平基线） |
| `persist` | 同一 mixed + set-aside + window + hitRatio |
| `thrash` | 同窗 `hitRatio=1.0` |
| `modes` | **主结论**：五档 median + 相对 mixed 加速比 + CSV |

#### 运行示例

```bash
# 主证据
./bin/02_memory_optim_04_l2_residency --mode modes

# 单 mode 调试（会先跑 mixed 作对照）
./bin/02_memory_optim_04_l2_residency --mode persist
./bin/02_memory_optim_04_l2_residency --mode thrash

# 有 CSV 后重画
python scripts/plot_b04_l2.py
```

结果：`docs/results/B-04_l2.md`、`B-04_modes.csv`。配套文章：`article/02_memory_optim/B-04*.md`。

口径：CUDA event **median**；加速比 = `mixed_median / mode_median`。默认 `data-mb=64`、`iters=2048`、`set-aside-mb=8`、`window-mb=32`、`hit-ratio=0.25`。跑完 reset。

**RTX 5090 参考**：`persist` **0.999×** / `thrash` **1.001×** vs `mixed`；`streaming` **19.58×**（合并扫，不是 residency）。L2=96 MB，32 MB 窗口已装得下。

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
# 主证据一条即可：
./bin/02_memory_optim_08_tma_intro --mode sweep

# 可选：单 mode 对照
./bin/02_memory_optim_08_tma_intro --mode sync   --fma-iters 8
./bin/02_memory_optim_08_tma_intro --mode pipe2  --fma-iters 8

# 有 CSV 后重画：
python scripts/plot_b08_tma.py

# 可选旁证（NCU；忽略附着时程序自打印的 ms）：
DO_NCU=1 bash examples/02_memory_optim/08_profile_tma.sh ncu-only
# SASS：bash examples/02_memory_optim/08_dump_sass.sh
```

结果：`docs/results/B-08_tma.md`。配套文章：`article/02_memory_optim/B-08*.md`。

---

### 第 19 章 / B-09：数据布局 (`09_layout_transform.cu`)

AoS vs SoA（按 `touch_fields`）+ 矩阵 copy / transpose（**不限 sm_90+**）：

| mode | 含义 |
|------|------|
| `aos` / `soa` | 宽记录 8×float，读写前 k 个字段 |
| `copy` | 方阵 copy（transpose 上限） |
| `transpose_naive` | 跨步写 |
| `transpose_tiled` / `transpose_pad` | SMEM tile 重排（±pad） |
| `sweep` | 扫 `touch_fields∈{1,2,4,8}`，SoA/AoS 加速比 CSV |
| `modes` | layout + transpose 全表（末尾带 CSV 块） |

#### 运行示例

```bash
# 主证据：
./bin/02_memory_optim_09_layout_transform --mode sweep

# transpose + touch=1 全表：
./bin/02_memory_optim_09_layout_transform --mode modes

# 有 CSV 后重画：
python scripts/plot_b09_layout.py

# 可选旁证（sectors/request；忽略附着时程序自打印的 ms）：
DO_NCU=1 bash examples/02_memory_optim/09_profile_layout.sh ncu-only
```

结果：`docs/results/B-09_layout.md`。配套文章：`article/02_memory_optim/B-09*.md`。

---

## 🔧 工具脚本

- `01_profile_bandwidth.sh`：B-01 可选 NCU（传 `--mode modes`；主证据仍是裸跑 median）
- `02_profile_banks.sh`：B-02 可选 NCU（传 `--mode modes`；主证据仍是裸跑 median）
- `05_profile_unified_memory.sh`：B-05 UM 证据链脚本（Linux/WSL 专用），使用 Nsight Systems 批量采集 fault/prefetch/advise 三组 trace
- `06_profile_pinned_dma.sh`：B-06 Pinned/DMA 批量对照（可选 `DO_NSYS=1` 采集 overlap 时间线）
- `07_profile_cp_async_pipeline.sh`：B-07 设备内 async/pipeline 批量对照（可选 `DO_NCU=1`）
- `07_dump_sass.sh`：B-07 导出 SASS，核对 `LDGSTS` / `CP.ASYNC`
- `08_profile_tma.sh`：B-08 TMA 批量对照（可选 `DO_NCU=1` / `ncu-only`）
- `08_dump_sass.sh`：B-08 导出 SASS，核对 TMA / bulk 路径
- `09_profile_layout.sh`：B-09 AoS/SoA/transpose 批量对照（可选 `DO_NCU=1` sectors/request）

## 📝 注意事项

### CUDA 版本要求

- **最低要求**：CUDA 12.0+（B-07+ 使用 `cuda::pipeline` 等）
- **推荐版本**：CUDA 12.3+ 或 CUDA 13.1+（支持完整特性）；B-08 TMA 推荐 12.4+
- **向后兼容性**：CUDA 13.1 完全向后兼容 CUDA 12.x 的代码和 API
- **架构要求**：
  - **B-01 / B-02（合并、bank）**：无 sm_90+ 硬门槛
  - Async Copy Pipeline（B-07）：需要计算能力 8.0+（Ampere 及以上）
  - **TMA（B-08）**：需要计算能力 **9.0+**（Hopper / Blackwell）
  - **Layout（B-09）**：无 sm_90+ 硬门槛（合并是全架构问题）
  - L2 Persistence（B-04）：需要计算能力 8.0+（Ampere 及以上）
  - LDG.NT（B-01 可选）：Volta+ 更常见；streaming hint 语义见 PTX

### 其他注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台

