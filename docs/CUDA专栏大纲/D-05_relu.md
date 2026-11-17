# D-05 写作大纲：ReLU（短章）

> 状态：✅ 已落地。短章。共用 D-03 网格 / block / median；只换函数。  
> 导航：[`Module-D.md`](Module-D.md) · 上章：[`D-04_sigmoid.md`](D-04_sigmoid.md)  
> **定位**：手写一元激活。官方对照 **cuDNN Activation**。本机：`vec4/cudnn`≈**1.01**；与 D-04 同墙。须有「实现思路与代码」。

**标题**：`D-05. ReLU 达到 cuDNN`

## 边界

| 章 | 已有 | 本章 |
|---|---|---|
| D-03 | 三流带宽顶 ~1.55 TB/s | 尺子 |
| D-04 | sigmoid + `expf`，仍贴带宽 | 对照：去掉特殊函数后是否更贴 |
| D-06～D-10 | — | 同 harness |

## MVP

- 前向：`y = max(0, x)`。`scalar` / `vec4`。
- 反传一档：`dx = dy * (y > 0)`（输入 y 与 dy；等价于 x>0，零点不计）。
- 有效字节：前向 `2*n*4`；反传 `3*n*4`。含 L2。
- sweep `n=2^16…2^28`。正确性对 host。
- 主看右半段；两流 GB/s 对 D-03 虚线只作量级；同尺锁反传三流。
- 官方：`cudnnActivation*`（RELU）。
- 不做：half、NCU、重讲 float4。贴顶可瘦数字，不可省方案与 `.cu` 挂钩。

主命令：`./bin/04_operators_05_relu --mode sweep`

## 文献

| 层 | 条目 | 用途 |
|---|---|---|
| B | D-03 / D-04 本机带宽 | 是否更贴顶 |
| C | 本机 ReLU GB/s | 判停 |

## 本机结论（已测）

- 正文：[`article/04_operators/D-05. ReLU 达到 cuDNN.md`](../../article/04_operators/D-05.%20ReLU%20达到%20cuDNN.md)
- 结果：[`docs/results/D-05_relu.md`](../results/D-05_relu.md)
- `vec4/cudnn`≈**1.01**；前向 ~**1.50 TB/s** → 与 D-04 **同墙**
