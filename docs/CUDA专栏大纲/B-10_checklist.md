# B-10 写作大纲：Module B Checklist

> 状态：⏳ 规划中（骨架大纲；落地前再补文献池与 MVP）。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

**标题**：`B-10. Module B Checklist：从「症状」到「证据」到「处方」的统一表`

## 与前后章的边界

| 已有章节 | 已覆盖 | B-10 应深化 / 避免重复 |
|---|---|---|
| B-01～B-09 | 各章决策表、误区、SOP、实测结论 | **汇总成一张可检索表**；不重开 micro-bench 全家桶 |
| Module C | Warp / atomics / Graph 等 | 本章只收束 **访存 / 内存体系**；并发原语留给 C |

## TL;DR 目标结论（草稿，落地前改写）

1. 先定 **症状层**（带宽墙 / latency / Host↔Device / 设备内 copy / 布局），再选证据，勿一上来换 API。
2. 每条处方必须挂 **证据入口**（对应 `examples/02_*` mode + `docs/results/B-0N_*`）。
3. Host CE overlap、SM 内 `cp.async`、TMA bulk **三层分清**，禁止混成「async 就快」。
4. 布局问题优先于引擎升级：sectors/request / useful GB/s 异常时先修 AoS/SoA/transpose。
5. Checklist 是导航，不是新理论；细节回链各章正文。

## 建议正文结构

1. 问题定义：Module B 读完仍不知从哪开刀 → 统一 triage 表。
2. 症状 → 证据 → 处方总表（主产物）。
3. 按场景速查：H2D/D2H、UM、GMEM coalescing、SMEM bank、寄存器 spill、L2、async/TMA、布局。
4. 证据落点索引（binary / CSV / plot 脚本）。
5. 常见误区合并清单（去重后的 Top N）。
6. 钩子 → Module C。

## MVP（规划）

| 产物 | 说明 | 裁决 |
|---|---|---|
| 正文 checklist 表 | 症状 / 指标 / 处方 / 章节链接 | **必做** |
| `docs/results/` 索引段 | 链到 B-05～B-09 已有摘要 | **必做** |
| 新 `.cu` micro-bench | — | **默认不做**（除非缺口实验） |

## 证据最低要求

- 不新增主结论曲线时，以各章已落地 CSV/摘要为旁证链接即可。
- 若补「一键打印入口表」脚本，须极短、可选。

## 参考文献池

待写章时按 [`../CUDA专栏规划.md`](../CUDA专栏规划.md) 固定结构 + skill `literature.md` 补齐；优先复用 B-01～B-09 文献池中已进决策表的条目。
