# CUDA 专栏规划（单独文档）

本文件是从 `docs/大模型算法系列规划.md` 中**抽离并优化**得到的 **CUDA 专栏**独立规划稿，便于后续按专栏拆分与持续迭代。

---

## 0. 专栏定位与读者收益

**专栏目标**：建立从 C++/CUDA 源码 → PTX/SASS → SM 微架构 → Memory/Compute Roofline → 工业级 Benchmark/Profiling 的完整闭环能力；最终能“写得快、测得准、改得对”。

**仓库落地路径**：

- **正文文章**：`article/`
- **可运行示例（章节实验代码）**：`examples/`（主产物；CMake 自动扫描）
- **实测与绘图**：`docs/results/`（CSV/摘要）+ `scripts/plot_b0N_*.py` → `article/**/assets/`
- **目录结构**：[`docs/仓库架构与现状.md`](仓库架构与现状.md)（仓库已瘦身为 `article/` + `examples/` 主线）

**状态约定**：

- ✅ 已落地：仓库已有对应文章/代码
- 🟡 部分落地：文章/代码有其一，或为占位实现
- ⏳ 规划中：仅大纲，仓库暂无对应实现

**正文插图约定（B-05 起定稿）**：原理 / 时间线用短 **ASCII**；实测用 **matplotlib**（可复现）；封面可选。不再把信息过载的 AI 教学海报当正文原理图。

---

## 1. 目录总览（Module A–E）

- **Module A（1–10）CUDA 基础与 GPU 架构**：✅（文章/示例已落地；封面已加）
- **Module B（11–20）内存体系与访存优化**：🟡（11–17 / B-01～B-07 文章+示例+实测已落地；18–20 规划中）
- **Module C（21–30）核心编程技巧与并发原语**：⏳（规划中；`examples/03_*` 仅 README）
- **Module D（31–40）计算原语与高级算子实现**：⏳（规划中）
- **Module E（41–50）深度学习工程实战与系统集成**：⏳（规划中；仓库 Python/绑定为占位）

---

## 2. Module A：CUDA 基础与 GPU 架构（1–10）✅

> **模块目标**：建立从逻辑并行模型到硬件执行实体的“物理映射”，并形成可复现的性能分析习惯（SASS/NSYS/NCU）。

### 2.1 章节清单（文章 ↔ 示例映射）

| 篇章 | 主题 | 文章（正文） | 示例（可运行） |
|---|---|---|---|
| 1 | CUDA 核心概念总览与演进 | `article/01_cuda_basic/A-01*.md` ✅ | `examples/01_cuda_basics/01_hello_modern.cu` ✅ |
| 2 | GPU 硬件架构深度解析 | `article/01_cuda_basic/A-02*.md` ✅ | `examples/01_cuda_basics/02_hardware_query.cu` ✅ |
| 3 | 编程模型物理映射（GTE/SM/Warp） | `article/01_cuda_basic/A-03*.md` ✅ | `examples/01_cuda_basics/03_grid_mapping.cu` ✅ |
| 4 | 线程调度：SIMT / Divergence / Replay | `article/01_cuda_basic/A-04*.md` ✅ | `examples/01_cuda_basics/04_warp_divergence.cu` ✅ |
| 5 | Kernel 结构与 ABI / SASS 视角 | `article/01_cuda_basic/A-05*.md` ✅ | `examples/01_cuda_basics/05_kernel_structure.cu` ✅ |
| 6 | CUDA 工具链：NVCC / NVRTC | `article/01_cuda_basic/A-06*.md` ✅ | `examples/01_cuda_basics/06_nvrtc_jit.cpp` ✅ |
| 7 | 内存模型全景：UVA / Memory Spaces | `article/01_cuda_basic/A-07*.md` ✅ | `examples/01_cuda_basics/07_memory_spaces.cu` ✅ |
| 8 | 异步执行：Stream / Event / Pipeline | `article/01_cuda_basic/A-08*.md` ✅ | `examples/01_cuda_basics/08_async_pipeline.cu` ✅ |
| 9 | 调试与错误诊断：Compute Sanitizer | `article/01_cuda_basic/A-09*.md` ✅ | `examples/01_cuda_basics/09_debug_and_sanitizer.cu` ✅ |
| 10 | 性能建模：Roofline / SOL | `article/01_cuda_basic/A-10*.md` ✅ | `examples/01_cuda_basics/10_roofline_demo.cu` ✅ |

### 2.2 对仓库结构的优化建议（不改代码，仅改“规划表达”）

原总规划里曾写成 `examples/01_cuda_basics/01_hello_world/` 这种“按子目录组织”的结构，但仓库当前实际是**扁平单文件**组织。建议本专栏统一采用：

- **示例代码**：以 `examples/<module_dir>/<NN>_<topic>.<cu/cpp>` 为主（当前就是这样）
- **配套脚本**：放在同目录（当前如 `examples/01_cuda_basics/*_inspect_*.sh` 已具备）

这样规划页与仓库现状一致，读者不需要“脑补目录重构”。

---

## 3. Module B：内存体系与访存优化（11–20）🟡

> **模块目标**：攻克 Memory Wall，掌握从访问模式、缓存策略到异步搬运流水线的系统化方法。

### 3.1 已落地章节（11–17 / B-01～B-07）✅

| 篇章 | 文件编号 | 主题 | 文章（正文） | 示例（可运行） |
|---|---|---|---|---|
| 11 | **B-01** | Global Memory：Coalescing / 向量化 / TMA 视角 | `article/02_memory_optim/B-01*.md` ✅ | `examples/02_memory_optim/01_global_mem_bandwidth.cu` ✅ |
| 12 | **B-02** | Shared Memory：Bank / Padding / Swizzle | `article/02_memory_optim/B-02*.md` ✅ | `examples/02_memory_optim/02_shared_mem_bank_conflict.cu` ✅ |
| 13 | **B-03** | 寄存器压力与 Spilling / Occupancy | `article/02_memory_optim/B-03*.md` ✅ | `examples/02_memory_optim/03_register_spill.cu` ✅ |
| 14 | **B-04** | L2 Cache 行为与 Residency | `article/02_memory_optim/B-04*.md` ✅ | `examples/02_memory_optim/04_l2_residency.cu` ✅ |
| 15 | **B-05** | Unified Memory：Page Fault / Prefetch / Advise | `article/02_memory_optim/B-05*.md` ✅ | `examples/02_memory_optim/05_unified_memory_pf.cu` ✅ + `docs/results/B-05_*` + `scripts/plot_b05_unified_memory.py` |
| 16 | **B-06** | Pinned Memory 与 DMA：H2D/D2H 吞吐与 Overlap | `article/02_memory_optim/B-06*.md` ✅ | `examples/02_memory_optim/06_pinned_dma.cu` ✅ + `docs/results/B-06_*` + `scripts/plot_b06_pinned_dma.py` |
| 17 | **B-07** | Async Copy / Pipeline：GMEM→SMEM 藏延迟边界 | `article/02_memory_optim/B-07*.md` ✅ | `examples/02_memory_optim/07_cp_async_pipeline.cu` ✅ + `docs/results/B-07_*` + `scripts/plot_b07_cp_async.py` |

> 编号约定：规划总序号 11–20 与 Module B 内文件编号 B-01～B-10 一一对应（11↔B-01 … 17↔B-07）。  
> B-05～B-07 正文：**ASCII 讲原理**，**matplotlib 讲实测**；详见 [`仓库架构与现状.md`](仓库架构与现状.md) §4。

### 3.2 规划中章节（18–20）⏳（建议“先落地最小可复现实验”）

建议 18–20 以**工程索引型**方式落地：每篇至少一个 `examples/02_memory_optim/0N_*.cu` + `docs/results/` 指标（NCU/NSYS/SASS 三选一；设备内 async 优先 NCU）。

- **可运行 micro-bench**
- **NCU/NSYS 指标采集脚本入口**
- **SASS 证据（可选）**

#### 3.2.1 B-07～B-10（工程索引型写作清单）

| 篇章 | 工程索引型标题（建议） | 最小可复现实验（MVP） | 证据/指标（最低要求） | 代码落点 |
|---|---|---|---|---|
| 17 / **B-07** ✅ | Async Copy / Pipeline：GMEM→SMEM 何时能藏延迟，何时反而变慢 | sync load vs `memcpy_async` / `cuda::pipeline`；扫 compute intensity（**设备侧** GMEM→SMEM，不重复 B-06 Host↔Device）；对照 2/4-stage | CUDA event 加速比 vs AI 曲线；NCU：WarpStateStats（sm_120 上部分 legacy 指标可能 n/a） | `examples/02_memory_optim/07_cp_async_pipeline.cu` ✅ + `docs/results/B-07_cp_async_pipeline.md` ✅ |
| 18 / **B-08** | Hopper TMA（可选）：从 API 到吞吐瓶颈（需要硬件门槛） | 最小 TMA copy + 计算模板（若覆盖） | 以 SASS/NCU 证据为主 | `examples/02_memory_optim/08_tma_intro.cu`（可选） |
| 19 / **B-09** | 数据布局（AoS/SoA/Transpose）：一次布局调整带来的事务变化 | AoS vs SoA + transpose micro-bench | NCU：dram 吞吐 +（可选）sectors/request 类指标 | `examples/02_memory_optim/09_layout_transform.cu`（建议新增） |
| 20 / **B-10** | Module B Checklist：从“症状”到“证据”到“处方”的统一表 | 汇总 11–19 的实验结论与常见坑 | 输出 1 页 checklist + 对应 benchmark/脚本入口 | `docs/CUDA专栏规划.md`（本文件）+ `docs/results/` |

#### 3.2.2 每篇文章的固定结构（模板）

- **要解决的问题**：一句话定义瓶颈与场景边界（例如 “为什么带宽很高但算力闲置？”）
- **结论先行**：给 3–5 条工程可执行结论（What to do / What not to do）
- **最小复现实验（MVP）**：可运行代码 + 参数 + 预期现象
- **证据链**：至少一种（NCU/NSYS/SASS），并落盘到 `docs/results/`
- **优化路径**：从“诊断”到“修改”到“回归验证”的步骤
- **常见误区**：把最常踩的坑写成 checklist

#### 3.2.3 B-06 写作大纲（Pinned / DMA / Overlap）✅ 已落地

> **已交付**：
> - 正文：`article/02_memory_optim/B-06*.md`（ASCII 原理 + RTX 5090 实测表/图、NSYS CLI 旁证）
> - 封面：`article/02_memory_optim/assets/B-06-pinned-dma-cover.png`
> - 实测图：`B-06-mode-gbs-bars.png` / `B-06-overlap-median-bars.png`（`scripts/plot_b06_pinned_dma.py`）
> - 示例：`examples/02_memory_optim/06_pinned_dma.cu` + `06_profile_pinned_dma.sh`
> - 结果：`docs/results/B-06_pinned_dma_rtx5090.md` + CSV
>
> 下方保留大纲便于对照审稿；以正文为准。

**标题**：`B-06. Pinned Memory 与 DMA：H2D/D2H 吞吐上限与 Overlap 条件`

**与前后章的边界**

| 已有章节 | 已覆盖 | B-06 应深化 / 避免重复 |
|---|---|---|
| A-07 | UVA、Zero-Copy 概念警示 | 用 micro-bench 量化 mapped vs memcpy；给出“只读一次 / 不缓存”判定 |
| A-08 | Stream 流水线、Pinned 是硬前置 | 不重讲三级流水线教程；改讲 **为何伪异步、CE 数量、双向饱和、chunk 粒度** |
| B-05 | UM fault/prefetch/advise → 显式管理 | 承接“显式路径怎么做到可控”；对照表可引用 B-05 §6 |
| B-07（规划） | 设备侧 async / pipeline | 本章只到 Host↔Device；GMEM→SMEM 留给下一章 |

**TL;DR 目标结论（写作时先写死 5 条）**

1. Pageable 上的 `cudaMemcpyAsync` **不是真异步**：驱动先 stage 到临时 pinned，再 DMA；吞吐低、且易与其他流串行化。
2. Pinned（`cudaMallocHost` / `cudaHostAlloc`）是 **DMA 直达 + 真 overlap** 的物理前提；`cudaHostRegister` 可用但通常更慢、更易踩 NUMA/对齐坑。
3. Overlap **三条件同时成立**：pinned + 非默认 stream + `asyncEngineCount≥1`；H2D∥D2H 还要求足够 CE（通常看 `asyncEngineCount≥2`）且主机内存带宽跟得上。
4. 吞吐上限常不是“理论 PCIe”，而是 **min(PCIe有效带宽, DRAM/NUMA带宽, 驱动开销/小包启动)**；小传输 latency-bound，大传输才逼近链路墙。
5. Zero-Copy（mapped pinned）省的是 memcpy launch，**不省 PCIe**；离散卡上仅适合“触达少、几乎不复用”的路径，否则直接打穿 PCIe。

**建议正文结构**

1. **问题定义**：B-05 之后，显式路径仍可能“看起来 Async 但不加速”——伪异步、伪 overlap、双向打不满。
2. **物理模型**：Pageable staging → Pinned DMA；Copy Engine 与 Compute Engine 分家；`asyncEngineCount` 含义。
3. **分配与 flags**：`cudaMallocHost` vs `cudaHostAlloc`（Default / Portable / Mapped / WriteCombined）vs `cudaHostRegister`；Pinned 过量会挤占 OS 可分页内存。
4. **Overlap 决策表**：单条件失败时的 NSYS 症状（串行 Copy、Host sync、同流依赖）。
5. **吞吐实验矩阵（MVP）**：扫 size；对照 pageable / pinned /（可选）WriteCombined；单向 H2D、单向 D2H、双向并行；可选 NUMA local vs remote。
6. **Zero-Copy 分支**：mapped kernel 直读 vs 显式 memcpy；与 A-07 警示对齐，用数据判停。
7. **工程边界（2024–2026）**：小包合并；`cudaMemcpyBatchAsync`（CUDA 12.8+）摊销 launch 开销（扩展阅读）；Grace NVLink-C2C / HMM 与“传统 PCIe+pinned”对照（注明硬件门槛，不作本章必跑）。
8. **误区清单 + SOP + 下一章钩子**（→ B-07 设备内 async）。

**最小可复现实验（`06_pinned_dma.cu`）**

| 编号 | 配置 | 要回答的问题 |
|---|---|---|
| A | pageable + `cudaMemcpyAsync` | 是否退化为 sync / staging？吞吐多少？ |
| B | pinned + Async H2D | 单向 DMA 吞吐是否明显上升？ |
| C | serial（1 stream 切块 H2D→Kernel） | overlap 的公平串行基线 |
| D | overlap（多 stream 切块） | 相对 serial 端到端是否下降？NSYS 是否跨 chunk 重叠？ |
| E | pinned + 双向 H2D∥D2H | 合计是否接近 2× 单向，还是被主机内存/CE 卡住？ |
| F | mapped zero-copy kernel | 有效 host-read 带宽（勿直接对比 memcpy GB/s） |

**证据最低要求**：CUDA event / 墙钟得到 GB/s（first/median）；优先用 `serial` vs `overlap` vs `pinned` 对照判定（copy-bound 时 overlap≈pinned 即成功）；NSYS CLI/`stats` 可作旁证，有 GUI 再看时间线。可选：记录 `asyncEngineCount`、PCIe 代数、NUMA 绑定。

**参考文献池（与正文 §9 对齐）**

- 官方：CUDA Best Practices（Pinned / Async Overlap）、Programming Guide（Async Execution）、Runtime API（[API sync behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)、`cudaHostAlloc` flags）、Nsight Systems User Guide
- 经典博客：[How to Optimize Data Transfers](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)、[How to Overlap Data Transfers](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- 新 API：CUDA 12.8+ [`cudaMemcpyBatchAsync`](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)
- 近年研究/工程：Grace Hopper system memory（[ICPP’24 / arXiv:2407.07850](https://arxiv.org/abs/2407.07850)）、MultiPath H2D（[arXiv:2512.16056](https://arxiv.org/abs/2512.16056)）、PCIe Gen5/NUMA 实测（nvbandwidth 类工具链）

#### 3.2.4 B-07 写作大纲（Async Copy / Pipeline）✅ 文章+示例+实测已落地

> **已交付**：
> - 正文：`article/02_memory_optim/B-07*.md`（ASCII 原理 + RTX 5090 完整 intensity sweep + NCU WarpStateStats + SASS 旁证）
> - 示例：`examples/02_memory_optim/07_cp_async_pipeline.cu` + `07_profile_cp_async_pipeline.sh` + `07_dump_sass.sh`
> - 结果：`docs/results/B-07_cp_async_pipeline.md` + `B-07_sweep.csv` / `B-07_modes.csv`
> - 实测图：`B-07-speedup-vs-fma.png` / `B-07-mode-speedup-bars.png`（`scripts/plot_b07_cp_async.py`）
>
> 路线：**Ampere-first 多级流水线（优先 thread-local / unified）+ arithmetic intensity 扫描**；TMA 整章交给 B-08；warp specialization / CUTLASS Pipeline 仅扩展阅读。

**标题**：`B-07. Async Copy / Pipeline：GMEM→SMEM 何时能藏延迟，何时反而变慢`

**与前后章的边界**

| 已有章节 | 已覆盖 | B-07 应深化 / 避免重复 |
|---|---|---|
| A-08 | Host 侧 Stream / H2D→Compute→D2H 三级流水线 | **不重讲** CE/Stream；一句话对照「Host CE overlap ≠ SM 内 async copy」 |
| B-01 | `cp.async`→TMA 概念演进、合并访问 | 把 Ampere 路径做成 **可复现 micro-bench + 决策表**；TMA 细节不展开 |
| B-02 | SMEM bank / padding / swizzle | 强调数据 **落地之后** bank/swizzle 仍适用；本章不重做 bank conflict 教程 |
| B-06 | Host↔Device pinned / DMA / overlap | 本章只谈 **GMEM→SMEM**；不重复 pageable/pinned |
| B-08（规划） | Hopper TMA / `cp.async.bulk` | 本章只给钩子：大批量多维搬运、指令带宽墙 → 下一章 |

**TL;DR 目标结论（写作时先写死 5 条）**

1. `cp.async` / `cuda::memcpy_async` 是 **SM 内** DMA：GMEM→SMEM，旁路寄存器；与 B-06 的 Host Copy Engine **不是一层**。
2. 收益来自 **outstanding stages × 足够 compute overlap**，不是「async 指令本身比 sync load 更快」。
3. **低 arithmetic intensity / latency-bound** 才值得上；已 compute-bound 或 occupancy 已能藏 LDG 时，pipeline 同步与多 stage SMEM 常净亏损（见 Svedin 等实证）。
4. Stage 加深换延迟，但挤占 SMEM → 掉 occupancy；shared/partitioned `cuda::pipeline` 有 per-stage barrier 开销——能 **thread-local** 就不要 block shared。
5. 对齐/尺寸不满足时可能回退或走非预期路径；Hopper+ 大批量多维搬运交给 **B-08 TMA**，本章只给钩子。

**建议正文结构**

1. **问题定义**：B-06 后数据已在 HBM，kernel 仍「等 LDG」——用一行对照表区分 Host CE overlap vs 设备内 async copy。
2. **物理模型**：`LDG → RF → STS` vs `LDGSTS` / `cp.async`；MIO / async copy 路径；为何不占长 scoreboard、可旁路 L1。
3. **API 分层与同步**：sync load → 低层 `__pipeline_memcpy_async` / PTX → `cuda::memcpy_async` + `cuda::barrier` / `cuda::pipeline`；unified vs partitioned；commit 需 warp 收敛（官方 Warp Entanglement 警示）。
4. **决策表**：何时上 pipeline、几 stage、何时回退 sync（对照文献 + 本机 intensity 曲线）。
5. **MVP 实验矩阵**：见下表；主证据用 CUDA event；NCU 作旁证。
6. **工程边界**：SMEM 预算 vs occupancy；`mio_throttle`；与 B-02「落地后仍要管 bank/swizzle」。
7. **扩展阅读（2021–2026）**：CudaDMA 专用 copy warp → Ampere 硬件 async；CUTLASS multistage vs warp-specialized（不写生产级 GEMM）；Blackwell 仍保留 cp.async 路径 → 说明本章在消费级新卡仍有价值；钩子 → B-08 TMA。
8. **误区清单 + SOP + 下一章钩子**（→ B-08 Hopper TMA）。

**最小可复现实验（`07_cp_async_pipeline.cu`）**

| 编号 | 配置 | 要回答的问题 |
|---|---|---|
| A | sync：`gmem→reg→smem`（或 sync load 后直接消费） | 公平基线时延/吞吐？ |
| B | `memcpy_async` + 单缓冲 wait（无 overlap） | 仅换指令、不做流水线时有无收益/开销？ |
| C | 2-stage `cuda::pipeline` | 相对 A 是否加速？ |
| D | 4-stage `cuda::pipeline` | 更深 stage 是否继续赚，还是被 SMEM/occupancy 反噬？ |
| E | 扫 compute intensity（FMA 次数或等价 AI） | 画出「加速比 vs AI」：低 AI 受益、高 AI 持平/变慢？ |
| F（可选） | thread-local vs block shared pipeline | shared pipeline 的 barrier 开销是否可测？ |

**证据最低要求**：CUDA event 得到 median 时延或有效带宽；**intensity 扫表**写入 `docs/results/`（主结论载体）。旁证：NCU 至少一组 A vs C/D（关注 `long_scoreboard` 下降、`mio_throttle`、或 sm vs dram 吞吐）。可选：SASS 确认出现 `LDGSTS` / `CP.ASYNC`。完整对照见已落地的 `examples/02_memory_optim/07_cp_async_pipeline.cu`。

**参考文献池（与正文参考文献节对齐）**

- 官方：CUDA Programming Guide — [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)、[Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)（thread-local 优先；Warp Entanglement）；[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)（GMEM→SMEM 硬件加速）
- 工程博客：[Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)（对照 CudaDMA）
- CCCL / libcu++：[`cuda::memcpy_async`](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html)（对齐门槛与架构回退；Hopper TMA 细节不在本章展开）
- 高质量实证：Svedin et al., *Benchmarking the Nvidia GPU Lineage… with Asynchronous Memory Transfers*（PMBS@SC’21 / [arXiv:2106.04979](https://arxiv.org/abs/2106.04979)）——低 AI 约 1.07–1.35×，高 AI 可至 ~0.95×；Li et al., *Performance Implications of Async Memcpy and UVM*（IISWC’23，[PDF](https://lca.ece.utexas.edu/pubs/Li_IISWC_2023.pdf)）——GMEM→SMEM 非瓶颈时 async 无收益
- 扩展阅读：Colfax / SIGARCH [Efficient GEMM Kernel Designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)；[CUTLASS Pipeline](https://docs.nvidia.com/cutlass/media/docs/cpp/pipeline.html)；MLC.ai [Pipelining GEMM with TMA](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_async/index.html)（为 B-08 铺垫）

---

## 4. Module C：核心编程技巧与并发原语（21–30）⏳

> **模块目标**：让“正确性、可维护性、可调优性”成为 CUDA 工程默认配置（而不是靠经验主义）。

建议按仓库落地方式定义：每篇对应一个 `examples/03_compute_primitives/<NN>_*.cu`，并至少给一个可被 NCU/NSYS 验证的结论。

可覆盖主题（建议）：

- Warp primitives（ballot/shfl、warp-level reduce/scan）
- Cooperative Groups / Cluster（若覆盖 Hopper/Blackwell）
- Atomics 与 contention（global/shared、分桶/分层规约）
- Kernel fusion 的代价与边界（register pressure、occupancy）
- CUDA Graph 与 launch overhead（与 Module E 联动）

---

## 5. Module D：计算原语与高级算子实现（31–40）⏳

> **模块目标**：从“会写 kernel”走向“能写出接近库级质量的 kernel”，形成算子实现范式。

建议落地策略：

- 先做 **Reduction / Softmax / LayerNorm** 这类通用算子（易评测、易验证）
- 再做 **GEMM 相关（Tensor Core）**（需要更完整的矩阵分块与数据布局）

---

## 6. Module E：深度学习工程实战与系统集成（41–50）⏳

> **模块目标**：把“kernel 优化”放回真实工程链路：Python → C++ 扩展 → profiler → benchmark → 部署形态。

### 6.1 与仓库现状的对齐（重要）

总规划中写了 `examples/05_dl_engineering/...` / Python 绑定等完整树；**当前仓库已删除占位 `python/`、`include/`、`src/`**，Module E 仅保留规划文档。

对 41–50 明确标注：

- **“规划中：目录/代码尚未落地”**
- 每落地一篇，再新建对应路径（文章/示例/脚本）

---

## 7. 本仓库的 CUDA Bench & Profiling 闭环（建议读者必跑）

### 7.1 章节实验（主入口）✅

- `examples/01_cuda_basics/*.cu`：Module A
- `examples/02_memory_optim/01_*.cu` … `07_*.cu`：Module B（含 B-07 intensity sweep）
- 实测摘要：`docs/results/B-05_*` / `B-06_*` / `B-07_*`

### 7.2 脚本与结果目录 ✅

- `scripts/plot_b05_unified_memory.py` / `plot_b06_pinned_dma.py` / `plot_b07_cp_async.py`：正文实测图
- `scripts/dump_sass.sh` → `docs/sass/`（可选；章节也可用 `examples/**/0N_dump_sass.sh`）
- `scripts/profile_ncu.sh` → `docs/results/ncu/`（按需）
- `scripts/parse_roofline.py` / `plot_roofline.py`：Roofline 辅助
- `docs/results/perf_table.md`：对比表占位

---

## 8. 后续维护建议（让规划“长期不漂移”）

- **规划文档只写“真实路径”**：落地后再补链接，不预设不存在的目录树
- **每章三件套**（建议强制）：文章 + 可运行代码 + 可复现指标（NSYS/NCU/SASS 任一）
- **将“规划 vs 已实现”显式标记**：避免读者/协作者误判仓库完成度

