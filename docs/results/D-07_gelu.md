# D-07 GELU — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 示例：`examples/04_operators/07_gelu.cu`  
> CSV：[D-07_sweep.csv](D-07_sweep.csv)  
> 官方对照：legacy `cudnnActivation*` **无 GELU**；本章尺子 = host 真值 + 有效带宽（默认不做 Graph Pointwise）

## 平台

| 项 | 值 |
|---|---|
| GPU | RTX 5090 / `sm_120` / 170 SM |
| 算子 | 前向 GELU（默认 tanh 近似；对照 erf）；反传解析导 |
| 有效字节 | 前向 `2*n*4`；反传 `3*n*4`（含 L2） |
| 编译 | `CMAKE_CUDA_ARCHITECTURES=120` |

## 前向主结论（右半段）

| n | tanh_vec4_ms | erf_vec4_ms | tanh/erf | tanh GB/s | erf GB/s |
|---:|---:|---:|---:|---:|---:|
| 2^26 | 0.3602 | 0.3611 | 0.998 | 1490 | 1487 |
| 2^27 | 0.7169 | 0.7197 | 0.996 | 1498 | 1492 |
| 2^28 | 1.4348 | 1.4364 | **0.999** | **1497** | **1495** |

定点 `n=2^26`（modes）：tanh `scalar/vec4`≈1.05；反传 tanh vec4 **0.518 ms** ≈ **1.55 TB/s**。

## 数值

- host 自洽：device 输出对各自公式（tanh / erf）通过探针。  
- 近似差：探针 **max abs≈5e-4**；max rel≈0.45 出在近零点（虚高）。看 abs，不把 rel 当实现误差。

## 怎么读

- 大数组 tanh ≈ erf ≈ **1.50 TB/s**；`tanh_vec4/erf_vec4`≈1 → **没有掉出带宽**。  
- 无 legacy Activation 可对；判停靠 host + 与 D-03～D-06 同墙。  
- 左半段 L2 峰值不当总线结论。
