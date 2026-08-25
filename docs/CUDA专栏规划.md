# CUDA 专栏规划（导航）

本文件是 **CUDA 专栏的进度导航与约定**。先锁全景与模块，再写单章。长写作大纲按章放在 [`CUDA专栏大纲/`](CUDA专栏大纲/README.md)，不要把全文大纲堆回本页。

源自 `docs/大模型算法系列规划.md` 的 CUDA 部分抽离；仓库落地以本文 + 分册大纲为准。

---

## 0. 文档地图

| 文档 | 职责 |
|---|---|
| **本文件** | L0 全景、整体布局、Module A–E 边界、章模板、明确不做、落地焦点 |
| [`CUDA专栏大纲/`](CUDA专栏大纲/README.md) | L1 模块地图 + 按章大纲（边界 / TL;DR / MVP / 文献池） |
| [`仓库架构与现状.md`](仓库架构与现状.md) | 目录树、已删占位、插图约定 |
| `docs/results/` | 实测摘要与 CSV |
| `article/` / `examples/` | 正文与可运行示例 |
| [专栏导读](../article/01_cuda_basic/00.%20专栏导读：怎么读、怎么跑、本机数字从哪来.md) | 外发入口：怎么读 / 怎么跑 / 已交付 vs 规划 |

---

## 1. 专栏定位与「全面」定义

**一句话目标**：从 C++/CUDA 源码 → PTX/SASS → SM 微架构 → Memory/Compute Roofline → 可复现调优 →（规划中）算子与工程集成；做到「写得快、测得准、改得对」。

**「全面」指什么**

- ✅ 性能工程师日常会撞到的 CUDA **能力面**都有 Module/章可回（展开权见表 §2.2）。
- ✅ 教法完整：顶层全景 → 模块地图 → 章内「原理(+图) / 短演进 / 实验 / 钩子」。
- ❌ 不是 Runtime/Driver API 百科，不是第二套官方文档，不是生产推理栈教程。

---

## 2. L0 全景（已锁定）

教法自下而上写；排障按症状进 Module。全景只回答：**问题从哪一层来、进哪个 Module、细节在哪一章展开**。

```text
        卡在哪一层？                         进哪个 Module
  ┌─────────────────────────────────┐
  │ 产品：扩展 / 打包 / 端到端口径 / 多卡入口 │  →  E  会接进产品
  ├─────────────────────────────────┤
  │ 算子：Reduce / Softmax / LN / GEMM / TC │  →  D  会写接近库级
  ├─────────────────────────────────┤
  │ 协作：warp / CG / atomic / sync / fuse / graph │  →  C  会协作与控 launch
  ├─────────────────────────────────┤
  │ 搬运：形状 → on-chip → 跨空间 → 引擎 → 布局 │  →  B  会搬数据
  ├─────────────────────────────────┤
  │ 映射与度量：模型 / 硬件 / 工具 / Roofline   │  →  A  会看会测
  └─────────────────────────────────┘
```

读者带走的路径（与上图同构，勿另起一套叙事）：

```text
A 会看会测 → B 会搬数据 → C 会协作与控 launch → D 会写接近库级算子 → E 会接进产品
```

**像素全景图**：构图以本节 ASCII 为权威；画的时候一张图、五层、每层一个标签，挂导读与本文。禁止徽章堆、双语脚注、周期表。未画之前以 ASCII 为准，不挡写稿。

### 2.1 三层导航

| 层 | 产物 | 职责 |
|---|---|---|
| L0 专栏全景 | 本节 ASCII（+ 日后一张总图）；导读 §3 | 问题从哪一层来、进哪个 Module |
| L1 模块地图 | [`Module-A`](CUDA专栏大纲/Module-A.md) … [`Module-E`](CUDA专栏大纲/Module-E.md)；B-10 / C-10 | 章怎么排、细节在哪一章 |
| L2 章正文 | `article/**` | 只展开本章边界内的原理 / 演进 / 实验 |

### 2.2 展开权（硬规则）

全景或模块地图里出现的词，**展开权在指定章**；其它章只留钩子，不重开教程。

| 词 | 展开章 | 其它章 |
|---|---|---|
| Grid / Block / Warp 映射 | A-03 | 钩子 |
| SIMT / Divergence / Replay | A-04 | 钩子 |
| ABI / SASS | A-05 | 钩子 |
| NVCC / NVRTC | A-06 | 钩子 |
| 内存空间 / UVA | A-07 | 钩子 |
| Host Stream / Event / CE | A-08 | 钩子 |
| Sanitizer | A-09 | 钩子 |
| Roofline / SOL | A-10 | 钩子 |
| GMEM 合并 / 对齐 / 向量化 | B-01 | 钩子 |
| SMEM bank / padding / swizzle | B-02 | 钩子 |
| 寄存器 / spilling / occupancy | B-03 | 钩子 |
| L2 residency | B-04 | 钩子 |
| Unified Memory | B-05 | 钩子 |
| Pinned / H2D DMA | B-06 | 钩子 |
| `cp.async` / 设备 pipeline | B-07 | 钩子 |
| TMA / bulk | B-08 | 钩子 |
| AoS / SoA / transpose | B-09 | 钩子 |
| Warp primitives | C-01 | 钩子 |
| Cooperative Groups | C-02 | 钩子 |
| Atomics / contention | C-03 | 钩子 |
| 同步分层 | C-04 | 钩子 |
| Kernel fusion 代价 | C-05 | 钩子 |
| CUDA Graph / launch 墙 | C-06 | 钩子 |
| Device Reduce | D-01 | C 不写算子正文 |
| Softmax | D-02 | 钩子 |
| LayerNorm / RMSNorm | D-03 | 钩子 |
| GEMM tiling | D-04 | 钩子 |
| Tensor Core GEMM | D-05 | 不复刻 CUTLASS |
| Epilogue 融合 | D-06 | 钩子 |
| pybind / 扩展闭环 | E-01 | 钩子 |
| 端到端 bench 口径 | E-03 | 不替代章内 median |
| 框架集成 | E-05 | 不写框架源码课 |
| Multi-GPU / NCCL | E-07 | 单卡主链之后 |

Cluster、PDL、Blackwell Tensor Memory、named barrier：需要时在邻章 **一句钩子**，不单开主线（C-07～C-09 候选槽可收，见 §3）。

---

## 3. 整体布局（已锁定）

五层 Module，每层 10 个槽；导读是第 0 篇，不占槽。

| 模块 | 总序 | 路径（落地时） | 收束章 |
|---|---|---|---|
| **A** | 1–10 | `article/01_cuda_basic/` · `examples/01_cuda_basics/` | A-10 = Roofline（度量语言终点，不是 Checklist） |
| **B** | 11–20 | `article/02_memory_optim/` · `examples/02_memory_optim/` | B-10 = Checklist |
| **C** | 21–30 | `article/03_compute_primitives/` · `examples/03_compute_primitives/` | C-10 = Checklist |
| **D** | 31–40 | 开写再建 `article/04_*` / `examples/04_*` | D-10 = Checklist |
| **E** | 41–50 | 开写再建；不提前建空 `python/` | E-10 = Checklist / 专栏收束索引 |

**槽位纪律**

| 规则 | 说明 |
|---|---|
| 前半锁定 | A 全 10 章；B 全 10 章主题；C-01～C-06；D-01～D-06；E-01～E-07 |
| 后半可裁 | C-07～C-09、D-07～D-09、E-08～E-09 允许合并/改名；**禁止为空凑序** |
| 新 Module 目录 | 随**首章落地**再建；禁止空 README 占坑 |
| 编号 | 总序 1–10↔A、11–20↔B、21–30↔C、31–40↔D、41–50↔E |

**状态标记**（只这三种）

| 标记 | 含义 |
|---|---|
| ✅ | 文章 + 示例 +（需要时）实测已落地，且不抢邻章展开权 |
| 🟡 | 部分落地（旧稿待对齐 / 缺图 / 或缺实测） |
| ⏳ | 仅大纲或未开工 |

**工程索引对齐**（B-02～B-04 适用）四条都到才从 🟡 改 ✅：边界表不抢邻章、一条 `--mode` 主命令 + median、本地 assets、文首文末 GitHub 绝对链。

---

## 4. 各模块

章表细节与文献池在 L1：[`Module-A`](CUDA专栏大纲/Module-A.md) … [`Module-E`](CUDA专栏大纲/Module-E.md)。本节只锁**教法弧**和**硬边界**。

### 4.1 Module A：映射与度量 ✅

> **一句话**：代码落在哪块硅、怎么测、怎么建模。  
> **教法弧**：模型 → 硬件 → 映射 → SIMT → ABI/SASS → 工具链 → 空间 → Host 异步 → 正确性 → Roofline。

| 篇章 | 主题 | 状态 |
|---|---|---|
| 0 / 导读 | 怎么读、怎么跑、本机数字 | ✅ |
| 1–10 / A-01～A-10 | 概念总览 → … → Roofline / SOL | ✅ |

硬边界：不讲合并处方（→ B-01）、不讲 bank（→ B-02）、不讲设备 `cp.async`/TMA（→ B-07/B-08）。Host Stream 只在 A-08。

### 4.2 Module B：数据怎么搬 🟡

> **一句话**：攻克 Memory Wall。  
> **教法弧**（锁定，按这个顺序讲，也按这个顺序对齐）：

```text
形状 B-01  →  on-chip B-02/B-03/B-04  →  跨空间 B-05/B-06  →  引擎 B-07/B-08  →  布局 B-09  →  索引 B-10
```

| 篇章 | 弧上位置 | 主题 | 状态 |
|---|---|---|---|
| 11 / B-01 | 形状 | GMEM：合并 / 对齐 / float4 | ✅ |
| 12 / B-02 | on-chip | SMEM：Bank / Padding / Swizzle | ✅ |
| 13 / B-03 | on-chip | 寄存器 / Spilling / Occupancy | 🟡 |
| 14 / B-04 | on-chip | L2 residency | 🟡 |
| 15 / B-05 | 跨空间 | Unified Memory | ✅ |
| 16 / B-06 | 跨空间 | Pinned / DMA overlap | ✅ |
| 17 / B-07 | 引擎 | `cp.async` / pipeline | ✅ |
| 18 / B-08 | 引擎 | Hopper TMA | ✅ |
| 19 / B-09 | 布局 | AoS / SoA / Transpose | ✅ |
| 20 / B-10 | 索引 | 症状 → 证据 → 处方 | ✅ |

硬边界：先形状后引擎；TMA 叙事不得出现在 B-01/B-02 开篇当主线。Host CE ≠ 设备 async ≠ TMA bulk。大纲：B-07～B-09 为样板；B-06/B-10 已归档。

**章序 vs 物理**：布局（B-09）和形状（B-01）是一层，现序排在引擎之后是迁就已发布顺序，不是「先 TMA 再改 AoS」。B-01 出口同时挂 B-09。对齐 B-02 时 swizzle 只讲 bank，**不要**写成「为 TMA 服务的下一站」。

### 4.3 Module C：线程怎么协作 🟡

> **一句话**：正确性 + 并发税 + launch/fuse。  
> **教法弧**：通信 C-01/C-02 → 争用与同步 C-03/C-04 → 摊销 launch C-05/C-06 →（候选）→ 索引 C-10。

| 篇章 | 主题 | 状态 |
|---|---|---|
| 21–26 / C-01～C-06 | Warp / CG / Atomics / Sync / Fusion / Graph | ✅ |
| 27–29 / C-07～C-09 | Persistent · 设备侧重叠 · named barrier 等 | ⏳ **候选可裁** |
| 30 / C-10 | Module C Checklist | ⏳ |

硬边界：不重开访存教程；不算子数值正文（→ D）；Graph 生产叠法 / Torch capture → E。详见 [Module-C.md](CUDA专栏大纲/Module-C.md)。

### 4.4 Module D：算子怎么写到接近库级 ⏳

> **一句话**：可验证的算子实现（正确性 + Roofline 形状 + 决策表）。  
> **教法弧**：规约族 D-01～D-03 → 矩阵 D-04/D-05 → epilogue D-06 →（候选）→ 索引 D-10。

| 总序 | 章 | 主题 |
|---|---|---|
| 31–33 | D-01～D-03 | Reduction / Softmax / LN·RMSNorm |
| 34–36 | D-04～D-06 | Naive GEMM→tiling / TC GEMM 入门 / Epilogue |
| 37–39 | D-07～D-09 | Attention 微基准 / 量化钩子 / 预留（**可裁**） |
| 40 | D-10 | Checklist |

硬边界：CUTLASS/CuTe 只对照，不复刻库文档；生产 FA / vLLM 不做主线。详见 [Module-D.md](CUDA专栏大纲/Module-D.md)。

### 4.5 Module E：怎么嵌进产品 ⏳

> **一句话**：kernel 回到可调用、可测量、可交付的链路。  
> **教法弧**：可调用 E-01/E-02 → 可测量 E-03/E-04 → 可交付 E-05/E-06 → 多卡钩子 E-07 →（候选）→ 收束 E-10。

| 总序 | 章 | 主题 |
|---|---|---|
| 41–42 | E-01～E-02 | pybind11 闭环 / 打包与构建 |
| 43–44 | E-03～E-04 | 端到端 bench 规范 / NSYS·NCU 工作流 |
| 45–46 | E-05～E-06 | 框架集成钩子 / 部署形态 |
| 47 | E-07 | Multi-GPU 钩子（P2P / NCCL 入口） |
| 48–49 | E-08～E-09 | MPS/MIG、回归门禁（**可裁**） |
| 50 | E-10 | Checklist / 专栏收束索引 |

硬边界：单卡主链（A–D）完整之前，E-07 不阻塞。不提前建空 `python/`。详见 [Module-E.md](CUDA专栏大纲/Module-E.md)。

---

## 5. 每篇文章的固定结构（模板）

新章大纲必须覆盖下列块（完整示例见 B-07/B-08/B-09）。

**教法四件套（读者感知）**

1. **原理**（+ 本地配图或 ASCII）
2. **短演进 / 钩子**（不抢后章；地图级）
3. **实验**（`--mode` + median；概念章可为可复现现象）
4. **边界与出口**（细节在哪一章）

**工程骨架（写作必填）**

1. 要解决的问题（边界表）
2. 结论先行（TL;DR 3–5 条；有数钉数）
3. 最小复现实验（MVP 可行性表）
4. 证据链（`docs/results/`）
5. 优化路径 / SOP / 误区
6. 下一章钩子（GitHub 目录绝对链）
7. 参考文献池（官方 / 工程 / 实证 / 前沿）

写稿流程：`.cursor/skills/aspl-cuda-article/SKILL.md`。Checklist 章（B-10 / C-10 / D-10 / E-10）无强制新 `.cu`。

**插图**

| 类型 | 做法 |
|---|---|
| L0 全景 | 以 §2 ASCII 为权威构图；日后一张本地总图 |
| L1 模块弧 | 各 Module 大纲里的教法弧；可选一张模块图 |
| 章内原理 | 本地原理图 1～2 张或短 ASCII；禁止 csdn 外链当主图 |
| 实测 | `matplotlib` → `assets/` |
| 不做 | 信息过载的 AI 教学海报冒充原理图 |

---

## 6. 明确不做 / 后置（防范围爆炸）

| 主题 | 裁决 | 说明 |
|---|---|---|
| CUDA Driver API 深挖 | **不做主章** | 需要时 A-06 一句钩子 |
| Dynamic Parallelism | **不做主章** | 现代性能路径少用；附录级即可 |
| CUDA Fortran / OpenACC | **不做** | 范围爆炸 |
| Thrust / CUB / CUTLASS 用法教程 | **不做教程** | D 可作对照，不复刻库文档 |
| 完整 FA / vLLM / TensorRT 生产栈 | **不做主线** | 导读已声明；D 扩展或后置 |
| Multi-GPU / P2P / NCCL / NVLink | **后置 E-07** | 单卡主链先完整 |
| MPS / MIG / Green Context | **E 可选** | 部署隔离，非主线 |
| Occupancy 单独成章 | **并入** B-03 + A-03 | 不单开 |

---

## 7. 落地状态与当前焦点

| 模块 | 状态 | 说明 |
|---|---|---|
| A | ✅ | 导读 + A-01～A-10 已收束 |
| B | 🟡 | B-01、B-02、B-05～B-10 ✅；B-03～B-04 按 §4.2 弧对齐 |
| C | 🟡 | C-01～C-06 ✅；C-07～C-09 候选；C-10 未开 |
| D / E | ⏳ | 章表已锁；无代码目录 |

**当前焦点（只一条主线）**：按弧对齐 on-chip，下一章 **B-03（寄存器 / spilling / occupancy）**。C 后半与 C-10 **不阻塞**。D/E 不开写。

---

## 8. Bench & Profiling 闭环

### 8.1 章节实验（主入口）

- Module A：`examples/01_cuda_basics/*.cu`
- Module B：`examples/02_memory_optim/01_*.cu` … `09_*.cu`（B-10 无新 `.cu`）
- Module C：`examples/03_compute_primitives/*.cu`
- 实测摘要：`docs/results/`；Checklist：`B-10_checklist.md`（日后 `C-10`）

### 8.2 脚本与结果

- `scripts/plot_b01_*.py` … `plot_b09_*.py` / `plot_c0N_*.py`：正文实测图
- `scripts/dump_sass.sh` → `docs/sass/`（可选）
- `scripts/profile_ncu.sh` → `docs/results/ncu/`（按需）
- Roofline 辅助：`scripts/parse_roofline.py` / `plot_roofline.py` / `plot_a10_roofline.py`

正文：`article/` · 示例：`examples/`（CMake 扫描）· 实测：`docs/results/` + `scripts/plot_*.py` → `assets/`。

---

## 9. 维护约定（防漂移）

1. **L0/L1 已锁定。** 改单章、大纲、示例、导航都不得偏离 §2 全景、§2.2 展开权、§4 教法弧。Agent 硬约束见 `.cursor/rules/aspl-cuda-column.mdc` 第 0 条与 `aspl-cuda-article` skill。**只有**明确「优化全景 / 改全景图 / 重锁 L0」才许改本节坐标系。导航本文件只写真实路径与状态；长大纲进 `CUDA专栏大纲/`。
2. **新章**：新建大纲分册 → 本表加一行 → 用户确认后再写代码/正文。
3. **每章交付**：文章 +（通常）可运行代码 + 可复现指标或可复现现象；原理图本地化。
4. **已落地章**：以正文与 `docs/results/` 为准；样板大纲保留 B-07～B-09，其余可归档。
5. **规划 vs 已实现**必须显式标记（✅ / 🟡 / ⏳）；候选章允许合并/改名，禁止为空凑序。
6. **当前焦点只写一条主线**；邻模块进度用状态表，不并列成「下一个」。
7. **全景图**画完后更新导读与本文 §2 链接；未画则以 §2 ASCII 为权威。
