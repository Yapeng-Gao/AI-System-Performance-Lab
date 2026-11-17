# D-03 写作大纲：Elementwise Add

> 状态：✅ 已收口（新主机 sm_120 复测）。5090：大数组有效带宽约 **1.55 TB/s**，`scalar/vec4`≈**1.03×**。  
> 导航：[`Module-D.md`](Module-D.md) · 上章：[`D-02_dot.md`](D-02_dot.md)  
> D-04～D-10 短章共用本章网格 / block / 字节口径。  
> **路线**：手写算子（「达到」三连收口）。达标=掉缓存后带宽不再涨。配图：封面=达到~1.55 TB/s；原理图=2R1W；实测图=加速比+带宽。

**标题**：`D-03. 逐元素加达到带宽顶`（只测 add；sub 同访存不另开曲线）

## 边界

| 章 | 已有 | 本章 |
|---|---|---|
| D-01 / D-02 | 归约、点积 | 无归约。写出整段 `c` |
| D-04～D-10 | — | 只换函数，不重讲向量化 |
| B-01 | 合并带宽 | 本章是算子形的三流带宽（2 读 1 写） |

## MVP

- `scalar` / `vec4`（float add）。`half2` 不做（短章激活再用 half 时另说）。
- sweep `n = 2^16 … 2^28`，`n % 256 == 0`。
- 有效字节 = `3 * n * 4`。含 L2。
- 正确性：抽检或全量 max abs 对闭式 `a+b`（值在 0…510，float 精确）。
- 不做：官方库对照、NCU、减/乘单独曲线（减在正文一句带过）。

主命令：`./bin/04_operators_03_elementwise --mode sweep`

## 文献

| 层 | 条目 | 用途 |
|---|---|---|
| A | [CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) | LDG.128 / STG.128 |
| B | D-01 / D-02 本机 | 一元 / 双流后的加载收益对照 |
| C | 有效带宽公式 3N×4 | 本机曲线 |
