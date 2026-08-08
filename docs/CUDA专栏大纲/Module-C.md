# Module C 大纲：核心编程技巧与并发原语（21–30）

> 状态：🟡 进行中（**C-01～C-04 ✅**；下一章 C-05）。目录已随首章落地。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · C-01：[`C-01_warp_primitives.md`](C-01_warp_primitives.md) · C-02：[`C-02_cooperative_groups.md`](C-02_cooperative_groups.md) · C-03：[`C-03_atomics_contention.md`](C-03_atomics_contention.md) · C-04：[`C-04_sync_layers.md`](C-04_sync_layers.md)

## 模块目标

让「正确性、可维护性、可调优性」成为 CUDA 工程默认配置：用 **可复现 micro-bench + 决策表** 固定并发原语的用法与代价，而不是靠经验主义。

## 落地约定

- 每篇对应一个 `examples/03_compute_primitives/<NN>_*.cu`，主证据 = CUDA event **median**；NCU/NSYS/SASS 按该章大纲裁决。
- 文章落 `article/03_compute_primitives/`。
- 写稿流程：`.cursor/skills/aspl-cuda-article/SKILL.md`（与 B-06～B-10 同构）。

## 章节总表（21–30 ↔ C-01～C-10）

| 总序 | 章 | 主题 | 一句话边界 | 状态 |
|---|---|---|---|---|
| 21 | **C-01** | Warp primitives（ballot / shfl / warp reduce·scan） | 寄存器级通信与 mask 正确性；**不做** DeviceReduce / Softmax | ✅ [大纲](C-01_warp_primitives.md) |
| 22 | **C-02** | Cooperative Groups（tile / coalesced） | 安全分组 API；Cluster 标可选支线 | ✅ [大纲](C-02_cooperative_groups.md) |
| 23 | **C-03** | Atomics 与 contention | global vs shared；分层规约 / 聚合；**深挖** warp-aggregated | ✅ [大纲](C-03_atomics_contention.md) |
| 24 | **C-04** | 同步分层（warp / block / grid） | `__syncwarp`、block barrier、grid sync；与 C-02 去重 | ✅ [大纲](C-04_sync_layers.md) |
| 25 | C-05 | Kernel fusion 代价边界 | fusion vs 多 kernel：寄存器 / occupancy / 可维护性 | ⏳ |
| 26 | C-06 | CUDA Graph 与 launch overhead | 先测 launch 墙；生产 capture / PyTorch 留给 Module E | ⏳ |
| 27 | C-07 | Persistent / grid-wide **或** Warp specialization（候选） | 与 C-06/C-04/B-07 易重叠；**可裁可并** | ⏳ 候选 |
| 28 | C-08 | 计算侧多 kernel 重叠（候选） | 可并入 C-06 一节；防滑回 A-08 | ⏳ 候选 |
| 29 | C-09 | （预留：divergence 工程处方 / named barrier 等） | 开写 C-06 后再定；勿锁死空壳主题 | ⏳ 候选 |
| 30 | C-10 | Module C Checklist | 症状→证据→处方；无强制新 `.cu`（对齐 B-10） | ⏳ |

> **开写纪律**：
> - **前半锁定**：C-01～C-06 + C-10 主题按表推进；后章开工前再拆 `C-0N_*.md`。
> - **后半可裁**：C-07～C-09 为候选槽，允许合并/改名；不得为凑满 10 章写空教程。
> - **硬边界**：不得抢 Module D 算子正文 / Module E 部署正文。

## 相对初稿的收紧（本次优化）

1. **补齐 26–30**：原先只列到 Graph，后半空缺；现给出可执行骨架，避免写到一半漂移。
2. **C-02 / C-04 拆分**：CG 负责「分组与集体 API」；同步分层单独成章，避免 CG 章又塞满 `__syncthreads` 全家桶。
3. **C-01 ↔ Module D**：C 只交付 **warp/block 级原语与模式**；完整 Reduce / Softmax / LN / GEMM 留给 D。
4. **C-03 ↔ 前沿**：ARC / warp-aggregated atomics 的深度实证放 C-03；C-01 只给「elect + ballot → 聚合」钩子。
5. **C-06 ↔ Module E**：C 测 launch / Graph 基础收益；Torch capture、部署图留给 E。
6. **后半可裁**：C-07～C-09 不锁死；优先保证 C-01～C-06 + Checklist 有独立可测命题。

## 与前后 Module 的硬边界

| 对象 | 已覆盖 / 将覆盖 | Module C 原则 |
|---|---|---|
| Module B / B-10 | 访存、async、TMA、布局 Checklist | **不重开** coalescing / TMA / pinned 教程；只挂钩「原语之上仍要管访存」 |
| A-04 | SIMT / Divergence / Replay | **不重讲** divergence 机制课；C-01 只固化 `*_sync` mask 正确性 |
| A-08 | Host Stream / CE overlap | C-08 只谈 **device 侧** 多 kernel 重叠，一句对照即可 |
| Module D | Reduce → Softmax/LN → GEMM/TC | C 停在原语与融合边界；算子数值与库级实现归 D |
| Module E | Python 扩展 / profiler / 部署 | Graph 生产叠法、框架集成归 E |

## 开写交接

- **已收束**：C-01～C-04（正文 + `docs/results/C-0N_*`）。
- **当前焦点**：拆 C-05 Kernel fusion 大纲后再写 `.cu` / 正文。
- **访存回退**：问题其实是带宽/布局时，先回 [`../results/B-10_checklist.md`](../results/B-10_checklist.md)。
