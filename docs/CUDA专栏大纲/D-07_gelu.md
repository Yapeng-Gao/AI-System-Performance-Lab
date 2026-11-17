# D-07 写作大纲：GELU（短章）

> 状态：✅ 已落地（RTX 5090）。短章。共用 D-03 网格 / float4 / median；只换函数。  
> 导航：[`Module-D.md`](Module-D.md) · 上章：[`D-06_elu.md`](D-06_elu.md)  
> **定位**：手写一元 GELU。主尺子 = 有效带宽 + host 真值；写清 **erf 精确式** 与 **tanh 近似** 的数值差。legacy `cudnnActivation*` **无 GELU**。

**标题**：`D-07. GELU 贴带宽`

## 边界

| 章 | 已有 | 本章 |
|---|---|---|
| D-03 | 网格 / float4 / median / 字节口径 | 复用；不重讲 |
| D-04～D-06 | sigmoid / ReLU / ELU 均 `vec4/cudnn`≈1.01、~1.50 TB/s | 对照：更重的特殊函数是否掉出 |
| D-08 | — | Swish；本章不做 Swish |
| C-05 | 融合 | 本章不融 bias / Linear |

**严禁**：把 Graph API 教程写成主线；重讲 float4；另开「GELU 数学课」抢注意力。

## TL;DR（本机）

1. **公式两档**：精确 erf；默认 tanh 近似（`--formula`）。
2. **正确性**：host 自洽；探针 max abs≈5e-4（rel 近零虚高）。
3. **路径**：`scalar`/`vec4` + `--formula`；`sweep` 两档一起记。
4. **判停**：右半段 ~**1.50 TB/s**；`tanh_vec4/erf_vec4`≈1 → 未掉出。
5. **官方**：legacy Activation **无 GELU**；Graph 默认不做。

## 正文结构（学习链）

1. GELU：erf 与 tanh 近似  
2. 两档公式  
3. 手写主路径  
4. 复现与正确性  
5. 贴带宽  
6. 生产建议 → 7. 下一算子（D-08 Swish）  


## MVP

| 项 | 裁决 |
|---|---|
| 前向默认 | **tanh 近似**；另跑 **erf** |
| 反传 | `--mode bwd` / `modes`；`--formula` 可切 |
| 有效字节 | 前向 `2*n*4`；反传 `3*n*4`；含 L2 |
| 网格 | 同 D-03 |
| sweep | `n=2^16…2^28`；主看 median > 0.05 ms |
| 官方列 | **默认无**（host+带宽） |
| 不做 | half、NCU、Graph 教程 |

主命令：`./bin/04_operators_07_gelu --mode sweep`

## 落地产物

- 正文：[`article/04_operators/D-07. GELU 贴带宽.md`](../../article/04_operators/D-07.%20GELU%20贴带宽.md)
- 结果：[`docs/results/D-07_gelu.md`](../results/D-07_gelu.md) · [`D-07_sweep.csv`](../results/D-07_sweep.csv)
- 示例：`examples/04_operators/07_gelu.cu`

## 参考文献池

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | cuDNN Graph / Pointwise GELU | legacy Activation 无 GELU | 边界 |
| A 公式 | ONNX Gelu | erf 默认；`approximate=tanh` | 两档对齐 |
| B 工程 | Hendrycks & Gimpel, GELU | 定义出处 | 一句 |
| B 工程 | 本专栏 D-03～D-06 | ~1.50–1.55 TB/s | 是否掉出 |
| C 实证 | 本章 5090 median | tanh≈erf≈1.50 TB/s | 判停 |
