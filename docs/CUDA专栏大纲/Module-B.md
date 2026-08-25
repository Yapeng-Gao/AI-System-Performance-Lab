# Module B 大纲：数据怎么搬（11–20）

> 状态：🟡 工程索引对齐中（B-01、B-02、B-05～B-10 ✅；B-03～B-04 按弧对齐）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §2 全景 · §4.2

## 模块目标

攻克 Memory Wall。从访问形状、on-chip 缓存、跨空间、异步引擎到布局，给出可复现处方；出口是 B-10 Checklist。

## 教法弧（锁定）

```text
形状 B-01  →  on-chip B-02 / B-03 / B-04  →  跨空间 B-05 / B-06
  →  引擎 B-07 / B-08  →  布局 B-09  →  索引 B-10
```

先形状后引擎。TMA / `cp.async` 不得写进 B-01/B-02 开篇当主线。

布局（B-09）和形状（B-01）是一层；现序在引擎之后是迁就已发布顺序。B-01 出口同时挂 B-09。对齐 B-02 时 swizzle 只讲 bank，不要写成「为 TMA 服务的下一站」。

## 章节总表（11–20 ↔ B-01～B-10）

| 总序 | 章 | 弧 | 主题 | 状态 | 大纲 |
|---|---|---|---|---|---|
| 11 | B-01 | 形状 | GMEM：合并 / 对齐 / float4 | ✅ | [B-01](B-01_global_mem.md) |
| 12 | B-02 | on-chip | SMEM：Bank / Padding / Swizzle | ✅ | [B-02](B-02_shared_mem.md) |
| 13 | B-03 | on-chip | 寄存器 / Spilling / Occupancy | 🟡 | 待建 |
| 14 | B-04 | on-chip | L2 residency | 🟡 | 待建 |
| 15 | B-05 | 跨空间 | Unified Memory | ✅ | — |
| 16 | B-06 | 跨空间 | Pinned / DMA overlap | ✅ | [归档](archive/B-06_pinned_dma.md) |
| 17 | B-07 | 引擎 | `cp.async` / pipeline | ✅ | [B-07](B-07_cp_async.md) |
| 18 | B-08 | 引擎 | Hopper TMA | ✅ | [B-08](B-08_tma.md) |
| 19 | B-09 | 布局 | AoS / SoA / Transpose | ✅ | [B-09](B-09_layout.md) |
| 20 | B-10 | 索引 | 症状 → 证据 → 处方 | ✅ | [归档](archive/B-10_checklist.md) |

路径：`article/02_memory_optim/` · `examples/02_memory_optim/`。样板：B-07～B-09。

## 对齐验收（🟡 → ✅）

四条都到才改标记：边界表不抢邻章；一条 `--mode` + median；本地 assets；文首文末 GitHub 绝对链。

## 硬边界

| 对象 | Module B 原则 |
|---|---|
| A-10 | 不重画 Roofline；承接「带宽墙先修访存」 |
| A-08 | Host CE ≠ 设备 async ≠ TMA bulk |
| C / D | 不讲 warp 原语、不算子数值 |
| 展开权 | 见规划 §2.2；TMA 只在 B-08 |

## 开写交接

- **已收口**：B-01（形状）、B-02（SMEM bank）。
- **主线**：按弧对齐 on-chip，下一章 **B-03**。
- **C 后半**：不阻塞本弧。
