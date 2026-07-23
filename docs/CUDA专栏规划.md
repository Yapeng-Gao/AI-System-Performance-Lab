# CUDA 专栏规划（单独文档）

本文件是从 `docs/大模型算法系列规划.md` 中**抽离并优化**得到的 **CUDA 专栏**独立规划稿，便于后续按专栏拆分与持续迭代。

---

## 0. 专栏定位与读者收益

**专栏目标**：建立从 C++/CUDA 源码 → PTX/SASS → SM 微架构 → Memory/Compute Roofline → 工业级 Benchmark/Profiling 的完整闭环能力；最终能“写得快、测得准、改得对”。

**仓库落地路径**：

- **正文文章**：`article/`
- **可运行示例（章节实验代码）**：`examples/`
- **微基准与数据采集链路**：`benchmarks/` + `scripts/` + `docs/results/`

**状态约定**：

- ✅ 已落地：仓库已有对应文章/代码
- 🟡 部分落地：文章/代码有其一，或为占位实现
- ⏳ 规划中：仅大纲，仓库暂无对应实现

---

## 1. 目录总览（Module A–E）

- **Module A（1–10）CUDA 基础与 GPU 架构**：✅（文章/示例已落地）
- **Module B（11–20）内存体系与访存优化**：🟡（11–16 / B-01～B-06 已落地；17–20 规划中）
- **Module C（21–30）核心编程技巧与并发原语**：⏳（规划中）
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

### 3.1 已落地章节（11–16 / B-01～B-06）✅

| 篇章 | 文件编号 | 主题 | 文章（正文） | 示例（可运行） |
|---|---|---|---|---|
| 11 | **B-01** | Global Memory：Coalescing / 向量化 / TMA 视角 | `article/02_memory_optim/B-01*.md` ✅ | `examples/02_memory_optim/01_global_mem_bandwidth.cu` ✅ |
| 12 | **B-02** | Shared Memory：Bank / Padding / Swizzle | `article/02_memory_optim/B-02*.md` ✅ | `examples/02_memory_optim/02_shared_mem_bank_conflict.cu` ✅ |
| 13 | **B-03** | 寄存器压力与 Spilling / Occupancy | `article/02_memory_optim/B-03*.md` ✅ | `examples/02_memory_optim/03_register_spill.cu` ✅ |
| 14 | **B-04** | L2 Cache 行为与 Residency | `article/02_memory_optim/B-04*.md` ✅ | `examples/02_memory_optim/04_l2_residency.cu` ✅ |
| 15 | **B-05** | Unified Memory：Page Fault / Prefetch / Advise | `article/02_memory_optim/B-05*.md` ✅ | `examples/02_memory_optim/05_unified_memory_pf.cu` ✅（含 `05_profile_unified_memory.sh`） |
| 16 | **B-06** | Pinned Memory 与 DMA：H2D/D2H 吞吐与 Overlap | `article/02_memory_optim/B-06*.md` ✅ | `examples/02_memory_optim/06_pinned_dma.cu` ✅ |

> 编号约定：规划总序号 11–20 与 Module B 内文件编号 B-01～B-10 一一对应（11↔B-01 … 16↔B-06）。

### 3.2 规划中章节（17–20）⏳（建议“先落地最小可复现实验”）

为了与仓库现有 `benchmarks/` + `scripts/` 的数据链路形成闭环，建议 17–20 以**工程索引型**方式落地：每篇至少给一个可运行 micro-bench + 可复现指标（NCU/NSYS/SASS 三选一；设备内 async 优先 NCU）。

- **可运行 micro-bench**
- **NCU/NSYS 指标采集脚本入口**
- **SASS 证据（可选）**

#### 3.2.1 B-07～B-10（工程索引型写作清单）

| 篇章 | 工程索引型标题（建议） | 最小可复现实验（MVP） | 证据/指标（最低要求） | 代码落点 |
|---|---|---|---|---|
| 17 / **B-07** | Async Copy / Pipeline：何时能隐藏延迟，何时反而变慢 | `cuda::pipeline`/`memcpy_async` 对比同步 load；扫不同 compute intensity（**设备侧** GMEM→SMEM，不重复 B-06 的 Host↔Device） | NCU：sm 吞吐 vs dram 吞吐（或简单吞吐对比表） | `benchmarks/cp_async_pipeline.cu`（已存在，可拆成 example） |
| 18 / **B-08** | Hopper TMA（可选）：从 API 到吞吐瓶颈（需要硬件门槛） | 最小 TMA copy + 计算模板（若覆盖） | 以 SASS/NCU 证据为主 | `examples/02_memory_optim/07_tma_intro.cu`（可选） |
| 19 / **B-09** | 数据布局（AoS/SoA/Transpose）：一次布局调整带来的事务变化 | AoS vs SoA + transpose micro-bench | NCU：dram 吞吐 +（可选）sectors/request 类指标 | `examples/02_memory_optim/08_layout_transform.cu`（建议新增） |
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
> - 正文：`article/02_memory_optim/B-06*.md`（含原理图、RTX 5090 实测、NSYS CLI 旁证）
> - 封面：`article/02_memory_optim/assets/B-06-pinned-dma-cover.png`
> - 示例：`examples/02_memory_optim/06_pinned_dma.cu` + `06_profile_pinned_dma.sh`
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

总规划中写了 `examples/05_dl_engineering/41_fusion_template/ ...` 的完整树，但仓库目前尚未落地该目录；同时 `python/csrc/module.cpp` 与 `include/aspl/ops/ops.h` 仍是占位。

因此建议在本专栏中对 41–50 明确标注：

- **“规划中：目录/代码尚未落地”**
- 每落地一篇，再补上对应路径（文章/示例/脚本）

---

## 7. 本仓库的 CUDA Bench & Profiling 闭环（建议读者必跑）

### 7.1 基准（当前已存在）✅/🟡

- `benchmarks/hbm_pointer_chasing.cu`：HBM pointer chasing 下限（带宽墙）
- `benchmarks/cp_async_pipeline.cu`：异步搬运流水线（`cuda::pipeline` / `memcpy_async`）
- `benchmarks/attention_memory_bound.cu`：attention 形态的 memory/compute 交织示例
- `benchmarks/bench_flash_attn.cu`：NVBench 流程样例（当前为 mock 占位实现）🟡

### 7.2 脚本与结果目录 ✅

- `scripts/dump_sass.sh` → `docs/sass/`：导出不同 SM 架构下的 SASS
- `scripts/profile_ncu.sh` → `docs/results/ncu/`：采集 NCU CSV
- `scripts/parse_roofline.py`：从 CSV 推导 BW/TFLOPs/OI（Roofline 打点基础）
- `docs/results/perf_table.md`：对比表（可作为后续自动汇总输出）

---

## 8. 后续维护建议（让规划“长期不漂移”）

- **规划文档只写“真实路径”**：落地后再补链接，不预设不存在的目录树
- **每章三件套**（建议强制）：文章 + 可运行代码 + 可复现指标（NSYS/NCU/SASS 任一）
- **将“规划 vs 已实现”显式标记**：避免读者/协作者误判仓库完成度

