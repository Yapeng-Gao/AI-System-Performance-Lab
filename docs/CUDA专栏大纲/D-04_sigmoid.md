# D-04 写作大纲：Sigmoid（短章）

> 状态：✅ 已落地。短章。共用 D-03 网格 / block / median；只换函数。  
> 导航：[`Module-D.md`](Module-D.md) · 上章：[`D-03_elementwise.md`](D-03_elementwise.md)  
> **定位**：手写一元激活。官方对照 **cuDNN Activation**；兼对照带宽。本机：`vec4/cudnn`≈**1.01**，~1.50 TB/s。

**标题**：`D-04. Sigmoid 达到 cuDNN`

## 边界

| 章 | 已有 | 本章 |
|---|---|---|
| D-03 | 两读一写 add，~1.55 TB/s | 一元 `sigmoid`；不重讲 float4 |
| D-05～D-10 | — | 同 harness，只换公式 |
| C-05 | 融合 | 本章不融 bias/dropout |

## MVP

- 前向：`y = 1/(1+exp(-x))`。`scalar` / `vec4`。
- 反传一档：`dx = dy * y * (1-y)`（输入 y 与 dy）。
- 有效字节：前向 `2*n*4`；反传 `3*n*4`。含 L2。
- sweep `n=2^16…2^28`。正确性对 host double（rtol 1e-5 量级）。
- 主看右半段：vec4 有效 GB/s 是否接近 D-03 带宽顶；若明显更低 → 算力墙。
- 官方：`cudnnActivationForward/Backward`（SIGMOID）。handle/desc 在计时外。
- 正文须有「实现思路与代码」：`act_f` + fwd/bwd + `cudnnActivation*` 计时边界。
- 不做：half、NCU、与 add 同进程内联对比。

主命令：`./bin/04_operators_04_sigmoid --mode sweep`

## 文献

| 层 | 条目 | 用途 |
|---|---|---|
| A | CUDA math `expf` / special functions | 特殊函数成本 |
| B | D-03 本机带宽顶 | 是否掉出带宽墙 |
| C | 本机 sigmoid GB/s | 判停 |

## 本机结论（已测）

- 正文：[`article/04_operators/D-04. Sigmoid 达到 cuDNN.md`](../../article/04_operators/D-04.%20Sigmoid%20达到%20cuDNN.md)
- 结果：[`docs/results/D-04_sigmoid.md`](../results/D-04_sigmoid.md)
- 大数组 `vec4/cudnn`≈**1.01**（达到）；前向 ~**1.50 TB/s**
- 反传 `n=2^26`：`vec4/cudnn`≈**1.014**
