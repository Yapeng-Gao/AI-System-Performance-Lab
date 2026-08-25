# B-01 写作大纲：Global Memory（合并 / 对齐 / 向量化）

> 状态：✅ **已收口**（正文 + `.cu` + 5090 `--mode modes` + plot；旧「TMA 演进」稿已删）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §4.2 · [Module-B.md](Module-B.md)

**写作机（2026-08-25）**：仅 GTX 1050；数字以用户机 RTX 5090 / `sm_120` 回填为准。

**路线**：**Microbench-first**——`misaligned` / `aligned` / `float4` 对照 + 相对加速比；`ldg_nt` 可选；async / L2 / TMA **只钩子**。

**硬件门槛**：**不限 sm_90+**。  
**证据口径**：裸跑 CUDA event **median**；绝对 GB/s 可能含 L2 → **主看相对 `aligned` 加速比**。

**标题**：`B-01. Global Memory：合并访问、对齐与 float4——先修有效带宽再谈异步`

**本机要点（5090，`offset=1`，n=16M）**：`misaligned` **0.988×** / `float4` **1.038×**（相对 `aligned`）。小 offset ≠ 合并崩盘。

**边界（锁定）**

| 已有章节 | 本章 |
|---|---|
| A-10 | 不重画 Roofline |
| B-02 | 不讲 bank；下一章按弧 |
| B-04 / B-07 / B-08 | 不做主测；§3.1 地图 |
| B-09 | 不重开布局课；字段跨步是形状近亲，出口挂 B-09 |

**刻意不做**：async / L2 persistence / TMA 主测；演进叙事当主线。

**交付 checklist**

- [x] 大纲确认（边界 + MVP modes + TL;DR）
- [x] `01_global_mem_bandwidth.cu`（`--mode modes` + median；无 async/L2 主路径）
- [x] 5090 实测 → `docs/results/B-01_*` + plot
- [x] 正文（本地 assets；文首文末绝对链；B-09 近亲出口）
- [x] 旧稿 `B-01.Global Memory 极致优化*` 删除
- [x] 规划 / Module-B / 大纲 README / examples README 回填
- [x] 收口评估（NCU sectors 为 P1，不挡）
