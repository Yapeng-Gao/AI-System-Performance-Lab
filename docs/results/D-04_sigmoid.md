# D-04 Sigmoid — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 示例：`examples/04_operators/04_sigmoid.cu`  
> CSV：[D-04_sweep.csv](D-04_sweep.csv)  
> 官方对照：cuDNN `cudnnActivationForward` / `Backward`（`CUDNN_ACTIVATION_SIGMOID`）

## 平台

| 项 | 值 |
|---|---|
| GPU | RTX 5090 / `sm_120` / 170 SM |
| 算子 | 前向 `sigmoid`；反传 `dx=dy*y*(1-y)` |
| 有效字节 | 前向 `2*n*4`；反传 `3*n*4`（含 L2） |
| 编译 | `CMAKE_CUDA_ARCHITECTURES=120`；`CUDNN_ROOT`（本机 `~/cudnn-9.12`） |

## 前向主结论（右半段）

| n | vec4_ms | cudnn_ms | vec4/cudnn | vec4 GB/s | cudnn GB/s |
|---:|---:|---:|---:|---:|---:|
| 2^26 | 0.3620 | 0.3569 | 1.015 | 1483 | 1504 |
| 2^27 | 0.7200 | 0.7117 | 1.012 | 1491 | 1509 |
| 2^28 | 1.4346 | 1.4192 | **1.011** | **1497** | **1513** |

反传定点 `n=2^26`：vec4 **0.521 ms** / cudnn **0.514 ms**，`vec4/cudnn`≈**1.014**（三流）。

## 怎么读

- 大数组手写 **达到** cuDNN（比值≈1.01）。  
- 前向两流 ~1.50 TB/s；仍贴带宽，未打成算力墙。  
- 左半段 L2 / 启动噪声不当结论。
