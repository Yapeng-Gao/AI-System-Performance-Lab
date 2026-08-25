# CUDA 专栏规划（导航）

本文件是 **CUDA 专栏的进度导航与约定**。长写作大纲按章放在 [`CUDA专栏大纲/`](CUDA专栏大纲/README.md)，不要把全文大纲堆回本页。

源自 `docs/大模型算法系列规划.md` 的 CUDA 部分抽离；仓库落地以本文 + 分册大纲为准。

---

## 0. 文档地图

| 文档 | 职责 |
|---|---|
| **本文件** | 定位、状态约定、Module A–E 总表、每章一行索引、写作模板、维护约定 |
| [`CUDA专栏大纲/`](CUDA专栏大纲/README.md) | 按章/按模块写作大纲（边界 / TL;DR / MVP / 文献池） |
| [`仓库架构与现状.md`](仓库架构与现状.md) | 目录树、已删占位、插图约定 |
| `docs/results/` | 实测摘要与 CSV |
| `article/` / `examples/` | 正文与可运行示例 |
| [专栏导读](../article/01_cuda_basic/00.%20专栏导读：怎么读、怎么跑、本机数字从哪来.md) | 外发入口：怎么读 / 怎么跑 / 已交付 vs 规划 |

---

## 1. 专栏定位与约定

**目标**：建立从 C++/CUDA 源码 → PTX/SASS → SM 微架构 → Memory/Compute Roofline → 工业级 Benchmark/Profiling 的闭环；做到「写得快、测得准、改得对」。

**仓库落地路径**：

- **正文**：`article/`
- **示例**：`examples/`（CMake 自动扫描）
- **实测与绘图**：`docs/results/` + `scripts/plot_b0N_*.py` → `article/**/assets/`

**状态约定**：

| 标记 | 含义 |
|---|---|
| ✅ | 文章 + 示例 +（需要时）实测已落地 |
| 🟡 | 部分落地（缺实测 / 或文章与代码缺一） |
| ⏳ | 仅大纲或未开工 |

**正文插图（B-05 起）**：原理 / 时间线用短 **ASCII**；实测用 **matplotlib**；封面可选。不用信息过载的 AI 教学海报当原理图。

**编号**：总序号 11–20 ↔ Module B 文件编号 B-01～B-10（11↔B-01 … 20↔B-10）。

---

## 2. Module A–E 总览

| 模块 | 范围 | 状态 | 大纲 |
|---|---|---|---|
| **A** CUDA 基础与 GPU 架构 | 1–10 / A-01～A-10 | 🟡 A-10 仍为旧稿 | A-01～A-09 ✅；[A-09 大纲](CUDA专栏大纲/A-09_sanitizer.md) |
| **B** 内存体系与访存优化 | 11–20 / B-01～B-10 | ✅ B-01～B-10 | [`CUDA专栏大纲/`](CUDA专栏大纲/README.md) |
| **C** 核心编程技巧与并发原语 | 21–30 | 🟡 C-01～C-06 ✅ | [`CUDA专栏大纲/Module-C.md`](CUDA专栏大纲/Module-C.md) |
| **D** 计算原语与高级算子 | 31–40 | ⏳ | [`CUDA专栏大纲/Module-D.md`](CUDA专栏大纲/Module-D.md) |
| **E** DL 工程实战与系统集成 | 41–50 | ⏳ | [`CUDA专栏大纲/Module-E.md`](CUDA专栏大纲/Module-E.md) |

---

## 3. Module A：CUDA 基础与 GPU 架构 🟡

> **目标**：逻辑并行模型 → 硬件执行实体的物理映射；养成可复现性能分析习惯（SASS/NSYS/NCU）。

| 篇章 | 主题 | 文章 | 示例 |
|---|---|---|---|
| 0 / 导读 | 怎么读、怎么跑、本机数字 | `article/01_cuda_basic/00. 专栏导读*` ✅ | — |
| 1 / A-01 | CUDA 核心概念总览与演进 | `article/01_cuda_basic/A-01*.md` ✅ | `examples/01_cuda_basics/01_hello_modern.cu` ✅ |
| 2 / A-02 | GPU 硬件架构深度解析 | `article/01_cuda_basic/A-02*.md` ✅ | `examples/01_cuda_basics/02_hardware_query.cu` ✅ |
| 3 / A-03 | 编程模型物理映射（GTE/SM/Warp） | `article/01_cuda_basic/A-03*.md` ✅ | `examples/01_cuda_basics/03_grid_mapping.cu` ✅ |
| 4 / A-04 | 线程调度：SIMT / Divergence / Replay | `article/01_cuda_basic/A-04*.md` ✅ | `examples/01_cuda_basics/04_warp_divergence.cu` ✅ |
| 5 / A-05 | Kernel 结构与 ABI / SASS 视角 | `article/01_cuda_basic/A-05*.md` ✅ | `examples/01_cuda_basics/05_kernel_structure.cu` ✅ |
| 6 / A-06 | CUDA 工具链：NVCC / NVRTC | `article/01_cuda_basic/A-06*.md` ✅ | `examples/01_cuda_basics/06_nvrtc_jit.cpp` ✅ |
| 7 / A-07 | 内存模型全景：UVA / Memory Spaces | `article/01_cuda_basic/A-07*.md` ✅（[大纲](CUDA专栏大纲/A-07_memory_spaces.md)） | `examples/01_cuda_basics/07_memory_spaces.cu` ✅ |
| 8 / A-08 | 异步执行：Stream / Event / Pipeline | `article/01_cuda_basic/A-08*.md` ✅（[大纲](CUDA专栏大纲/A-08_async_stream.md)） | `examples/01_cuda_basics/08_async_pipeline.cu` ✅ |
| 9 / A-09 | 调试与错误诊断：Compute Sanitizer | `article/01_cuda_basic/A-09*.md` ✅（[大纲](CUDA专栏大纲/A-09_sanitizer.md)） | `examples/01_cuda_basics/09_debug_and_sanitizer.cu` ✅ |
| 10 / A-10 | 性能建模：Roofline / SOL | `article/01_cuda_basic/A-10*.md` ✅ | `examples/01_cuda_basics/10_roofline_demo.cu` ✅ |

示例组织：扁平单文件 `examples/<module>/<NN>_<topic>.cu`（配套脚本同目录）。

---

## 4. Module B：内存体系与访存优化 ✅

> **目标**：攻克 Memory Wall；从访问模式、缓存策略到异步搬运流水线的系统化方法。

### 4.1 章节索引

| 篇章 | 主题 | 状态 | 文章 / 示例 | 大纲 |
|---|---|---|---|---|
| 11 / B-01 | Global Memory：Coalescing / 向量化 / TMA 视角 | ✅ | `B-01*.md` / `01_global_mem_bandwidth.cu` | —（早期章，以正文为准） |
| 12 / B-02 | Shared Memory：Bank / Padding / Swizzle | ✅ | `B-02*.md` / `02_shared_mem_bank_conflict.cu` | — |
| 13 / B-03 | 寄存器压力与 Spilling / Occupancy | ✅ | `B-03*.md` / `03_register_spill.cu` | — |
| 14 / B-04 | L2 Cache 行为与 Residency | ✅ | `B-04*.md` / `04_l2_residency.cu` | — |
| 15 / B-05 | Unified Memory：Fault / Prefetch / Advise | ✅ | `B-05*.md` / `05_unified_memory_pf.cu` + results/plot | — |
| 16 / B-06 | Pinned Memory 与 DMA / Overlap | ✅ | `B-06*.md` / `06_pinned_dma.cu` + results/plot | [归档](CUDA专栏大纲/archive/B-06_pinned_dma.md) |
| 17 / B-07 | Async Copy / Pipeline（GMEM→SMEM） | ✅ | `B-07*.md` / `07_cp_async_pipeline.cu` + results/plot | [大纲](CUDA专栏大纲/B-07_cp_async.md) |
| 18 / B-08 | Hopper TMA：Bulk Copy 与吞吐墙 | ✅ | `B-08*.md` / `08_tma_intro.cu` + results/plot | [大纲](CUDA专栏大纲/B-08_tma.md) |
| 19 / B-09 | 数据布局（AoS / SoA / Transpose） | ✅ | `B-09*.md` / `09_layout_transform.cu` + results/plot | [大纲](CUDA专栏大纲/B-09_layout.md) |
| 20 / B-10 | Module B Checklist（症状→证据→处方） | ✅ | `B-10*.md` + results 索引（无新 `.cu`） | [归档](CUDA专栏大纲/archive/B-10_checklist.md) |

路径前缀：文章 `article/02_memory_optim/`，示例 `examples/02_memory_optim/`。

### 4.2 当前焦点

1. **当前写作：下一章 A-10**（Roofline；A-09 三 tool 本机 PASS）
2. **Module B 已收束**（B-10 Checklist：正文 + `docs/results/B-10_checklist.md`）
3. **Module C**：C-01～C-06 ✅；其后 C-07/Checklist（[`CUDA专栏大纲/Module-C.md`](CUDA专栏大纲/Module-C.md)）

### 4.3 工程索引型最低交付（新章）

- 可运行 micro-bench（`examples/.../0N_*.cu`）
- 主证据：CUDA event median → `docs/results/`
- NCU/NSYS/SASS 为旁证（按章大纲裁决）

---

## 5. 每篇文章的固定结构（模板）

新章大纲必须覆盖下列块（完整示例见 B-07/B-08/B-09 样板分册；B-06/B-10 大纲已进 `CUDA专栏大纲/archive/`）：

1. **要解决的问题**：一句话定义瓶颈与场景边界
2. **结论先行**：3–5 条工程可执行结论（What to do / What not to do）
3. **最小复现实验（MVP）**：可运行代码 + 参数 + 预期现象；可行性表标清必做/可选/不做
4. **证据链**：主证据落盘 `docs/results/`；旁证按需
5. **优化路径**：诊断 → 修改 → 回归验证
6. **常见误区** + SOP + 下一章钩子
7. **参考文献池**：官方 / 工程 / 高质量实证 / 前沿（分层）

写稿流程见仓库 skill：`.cursor/skills/aspl-cuda-article/SKILL.md`。

---

## 6. Module C / D / E

| 模块 | 一句话 | 大纲 |
|---|---|---|
| C | Warp / CG / Atomics / Sync / Fusion / Graph / … / Checklist | [Module-C.md](CUDA专栏大纲/Module-C.md) |
| D | Reduce → Softmax/LN → GEMM/Tensor Core | [Module-D.md](CUDA专栏大纲/Module-D.md) |
| E | Python 扩展 → profiler → 部署；目录未建 | [Module-E.md](CUDA专栏大纲/Module-E.md) |

### 6.1 Module C 章节索引（21–30）

| 篇章 | 主题 | 状态 | 大纲 |
|---|---|---|---|
| 21 / C-01 | Warp primitives（ballot / shfl / warp reduce） | ✅ | [C-01_warp_primitives.md](CUDA专栏大纲/C-01_warp_primitives.md) |
| 22 / C-02 | Cooperative Groups（tile / coalesced；Cluster 可选） | ✅ | [C-02_cooperative_groups.md](CUDA专栏大纲/C-02_cooperative_groups.md) |
| 23 / C-03 | Atomics 与 contention | ✅ | [C-03_atomics_contention.md](CUDA专栏大纲/C-03_atomics_contention.md) |
| 24 / C-04 | 同步分层（warp / block / grid） | ✅ | [C-04_sync_layers.md](CUDA专栏大纲/C-04_sync_layers.md) |
| 25 / C-05 | Kernel fusion 代价边界 | ✅ | [C-05_kernel_fusion.md](CUDA专栏大纲/C-05_kernel_fusion.md) |
| 26 / C-06 | CUDA Graph 与 launch overhead | ✅ | [C-06_cuda_graph.md](CUDA专栏大纲/C-06_cuda_graph.md) |
| 27 / C-07 | Persistent **或** Warp specialization（候选） | ⏳ 候选 | 同上 |
| 28 / C-08 | 多 kernel 重叠（候选，可并入 C-06） | ⏳ 候选 | 同上 |
| 29 / C-09 | 预留（divergence 处方 / named barrier 等） | ⏳ 候选 | 同上 |
| 30 / C-10 | Module C Checklist | ⏳ | 同上 |

路径前缀（落地时建）：文章 `article/03_compute_primitives/`，示例 `examples/03_compute_primitives/`。

---

## 7. Bench & Profiling 闭环

### 7.1 章节实验（主入口）

- Module A：`examples/01_cuda_basics/*.cu`
- Module B：`examples/02_memory_optim/01_*.cu` … `09_*.cu`
- 实测摘要：`docs/results/B-05_*` … `B-09_*`；Checklist 索引 `B-10_checklist.md`

### 7.2 脚本与结果

- `scripts/plot_b05_*.py` … `plot_b09_layout.py`：正文实测图
- `scripts/dump_sass.sh` → `docs/sass/`（可选；章节亦可用 `examples/**/0N_dump_sass.sh`）
- `scripts/profile_ncu.sh` → `docs/results/ncu/`（按需）
- `scripts/parse_roofline.py` / `plot_roofline.py`：Roofline 辅助

---

## 8. 维护约定（防漂移）

1. **导航本文件只写真实路径与状态**；长大纲进 `CUDA专栏大纲/`。
2. **新章**：新建 `CUDA专栏大纲/B-0N_<topic>.md` 或 `C-0N_<topic>.md` → 本表加一行 → 用户确认后再写代码/正文。
3. **每章三件套**：文章 + 可运行代码 + 可复现指标（需要实测的章）。
4. **已落地章**：以正文与 `docs/results/` 为准；样板大纲保留 B-07～B-09，其余可归档到 `CUDA专栏大纲/archive/`。
5. **规划 vs 已实现**必须显式标记（✅ / 🟡 / ⏳）。
