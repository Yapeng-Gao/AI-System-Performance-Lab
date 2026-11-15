# D-05 ReLU — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 示例：`examples/04_operators/05_relu.cu`  
> CSV：[D-05_sweep.csv](D-05_sweep.csv)  
> 官方对照：cuDNN `cudnnActivationForward` / `Backward`（`CUDNN_ACTIVATION_RELU`）

## 平台

| 项 | 值 |
|---|---|
| GPU | RTX 5090 / `sm_120` / 170 SM |
| 算子 | 前向 `max(0,x)`；反传 `dx=dy*(y>0)` |
| 有效字节 | 前向 `2*n*4`；反传 `3*n*4`（含 L2） |
| 编译 | `CMAKE_CUDA_ARCHITECTURES=120`；`CUDNN_ROOT`（本机 `~/cudnn-9.12`） |

## 前向主结论（右半段）

| n | vec4_ms | cudnn_ms | vec4/cudnn | vec4 GB/s | cudnn GB/s |
|---:|---:|---:|---:|---:|---:|
| 2^26 | 0.3594 | 0.3564 | 1.009 | 1494 | 1507 |
| 2^27 | 0.7156 | 0.7118 | 1.005 | 1501 | 1509 |
| 2^28 | 1.4326 | 1.4173 | **1.011** | **1499** | **1515** |

反传定点 `n=2^26`：vec4 **0.523 ms** / cudnn **0.515 ms**，`vec4/cudnn`≈**1.015**。

## 怎么读

- 大数组手写 **达到** cuDNN（`vec4/cudnn`≈1.01）。  
- 与 D-04 同堵带宽墙；去掉 `expf` 没有拉开。  
- 左半段 L2 峰值不当总线结论；主看右半段 + `vec4/cudnn`。
