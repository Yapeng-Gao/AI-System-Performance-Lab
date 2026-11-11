# C-10 写作大纲：Module C Checklist

> 状态：✅ 文章+索引已落地（无新 `.cu`）。
>
> **已交付**：正文 `article/03_compute_primitives/C-10*.md`；封面/分层图；`docs/results/C-10_checklist.md`。
> **不做**：新 micro-bench、入口打印脚本、新 NCU/SASS、重跑 C-01～C-09。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)
>
> **路线**：Checklist-first。主题在规划 §4.3 已锁为「症状 → 证据 → 处方」，无新 `.cu`。开写时无路线分叉。

> **硬件门槛**：**不限**（汇总导航）。各行处方继承对应章门槛：C-08 PDL 为 sm_90+；C-09 unaligned named barrier 为 sm_70+。本机锚点均为 RTX 5090 / `sm_120`。

> **相对骨架的收紧**：
> 1. 主表按 **症状层**分行（访存回退 / 通信 / 争用 / 同步层 / 融合 / 提交 / 常驻 / 依赖核重叠 / CTA 内子集），禁止按章号流水账。
> 2. 每行必须挂 **证据入口**（binary `--mode` + `docs/results/C-0N_*`）。
> 3. 「少发 kernel」三杠杆分清：融合写回 ≠ Graph 提交 ≠ persistent 一任务一发。倍数禁止跨行引用。
> 4. 「重叠」三处分清：异流（A-08）≠ 同流依赖核（C-08）≠ CTA 内两角色（C-09）≠ mbarrier（B-07/B-08）。
> 5. 症状其实是带宽/布局时，**先回 B-10**，不在本章开刀。

**标题**：`C-10. Module C Checklist：从症状到证据到处方`

## 与前后章的边界

| 已有章节 | 已覆盖 | C-10 应深化 / 避免重复 |
|---|---|---|
| C-01～C-09 | 各章机制、决策表、5090 曲线 | **汇总成可检索 triage**；不重讲机制；数字只引用已落盘形状 |
| B-10 | 访存症状 → 证据 → 处方 | 本章只留「先回 B-10」一行；不重开 coalescing / TMA / 布局 |
| A-08 | Host stream / CE overlap | 防混层表里一行对照；不重测 |
| A-04 / A-09 | divergence、sanitizer | 正确性错数先回那里；本章不讲机制 |
| Module D / E | 算子、Torch capture、部署 | 钩子一句。不写 GEMM / pybind |

## TL;DR 目标结论

1. **先定症状层，再打开对应 binary**。访存/布局回 B-10。层错了，换 API 没有意义。
2. **每条处方挂证据入口**。主证据仍是裸跑 CUDA event **median**。禁止 ncu 附着墙钟。
3. **少发 kernel 的三把尺子不能互换**：瘦融合 **3.87×→9.81×**（fat 相对瘦融合慢 **8.01×**）；Graph 短链 **~3.3～4.1×**，`work=4096` → **1.01×**；persistent 一任务一发 **32×→3740×**，`work=4096` 仍 **314×**。
4. **重叠分三处**：同流依赖核 C-08 贴 **1×**（定点 **1.13×**）；CTA 内两角色 C-09 两边相等 **1.97×**，活在 arrive 前 **1.007×**。异流回 A-08。mbarrier 回 B-07/B-08。
5. **Checklist 是导航**。判停看能否复现该章形状。不要把 C-07 的 **3740×** 写进 Graph 或 GEMM。

## 建议正文结构

1. 问题：Module C 读完仍不知道该开哪一层。
2. 自问表（第一个「是」锁一行；访存排第一）。
3. 症状 → 证据 → 处方总表。
4. 防混层：少发 kernel / 重叠 / 同步，三张小表。
5. 证据索引（正文短，明细在 `docs/results/C-10_checklist.md`）。
6. 边界、SOP、误区。
7. 钩子：Module C 收束；D 未开，不写算子。

## MVP 可行性评估

| 编号 | 产物 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | 症状→证据→处方总表（覆盖 C-01～C-09 + 回 B-10） | ✅ | **必做** |
| B | 三杠杆 + 三处重叠 + 同步层对照 | ✅ | **必做** |
| C | `docs/results/C-10_checklist.md` 锚点与 binary 表 | ✅ | **必做** |
| D | 封面 + 1 张分层图 | ✅ | **必做** |
| E | SOP / 误区 / D 钩子 | ✅ | **必做** |
| F | 新 `.cu` / 新 CSV | ❌ | **不做** |
| G | `10_print_checklist` 入口脚本 | ❌ | **不做**（对齐 B-10） |
| H | 新 NCU/SASS | ❌ | **不做** |

## 参考文献池

本章不声称新曲线。文献只支撑「怎么分流」和「哪些对象不要写成同一个」。

**官方**

1. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：先分层再改代码。
2. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)：Compute / Memory SOL。Memory 高则离开本章，回 B-10。
3. [PTX ISA — `bar` / `barrier`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier) 与 [CUDA PG — Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)：named barrier、`__syncthreads`、`cuda::barrier` 不是同一对象。机制留在 C-09 与 B-07/B-08，本章只防混层。

**结构参考（数字以本仓库为准）**

4. GPU MODE Lecture 8 笔记：[CUDA Performance Checklist](https://christianjmills.com/posts/cuda-mode-notes/lecture-008/)。只借「症状→检查」的排法。加速比只用 `docs/results/C-01`～`C-09`。

**明确不引入**

- 不把 IPDPS’17、Kepler warp-aggregated atomics、CudaDMA 的倍数写成本章达标线（各章正文已拒绝）。
- 不新开算子论文。Device reduce / GEMM 留给 Module D。
