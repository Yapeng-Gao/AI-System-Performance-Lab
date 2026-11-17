# D-06 写作大纲：ELU（短章）

> 状态：✅ 已落地。短章。共用 D-03 harness；对 cuDNN Activation。  
> 导航：[`Module-D.md`](Module-D.md) · 上章：[`D-05_relu.md`](D-05_relu.md)  
> **定位**：负半轴带 `exp`。本机：`vec4/cudnn`≈**1.01**，**未掉出**带宽。

**标题**：`D-06. ELU 达到 cuDNN`

## 边界

| 章 | 已有 | 本章 |
|---|---|---|
| D-04 / D-05 | sigmoid / ReLU 均 `vec4/cudnn`≈1.01、~1.50 TB/s | ELU（α=1）；不重讲 float4 |
| D-07 | — | GELU |

## MVP

- 前向：`y = x (x≥0)；α(e^x−1) (x<0)`，α=1。`scalar` / `vec4` / `cudnn`。
- 反传：`dx = dy (y>0)；dy(y+α) (y≤0)`。
- 有效字节同前；sweep + modes。
- 主看右半段 `vec4/cudnn` 与 GB/s。
- 正文须有「实现思路与代码」（负半轴 `exp` 方案 + `.cu`）；贴顶可瘦数字，不可省方案。

主命令：`./bin/04_operators_06_elu --mode sweep`

## 本机结论（已测）

- 正文：[`article/04_operators/D-06. ELU 达到 cuDNN.md`](../../article/04_operators/D-06.%20ELU%20达到%20cuDNN.md)
- 结果：[`docs/results/D-06_elu.md`](../results/D-06_elu.md)
- `vec4/cudnn`≈**1.01**；~**1.50 TB/s**
