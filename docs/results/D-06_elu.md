# D-06 ELU — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 示例：`examples/04_operators/06_elu.cu`  
> CSV：[D-06_sweep.csv](D-06_sweep.csv)  
> 官方对照：cuDNN `cudnnActivationForward` / `Backward`（`CUDNN_ACTIVATION_ELU`，α=1）

## 平台

| 项 | 值 |
|---|---|
| GPU | RTX 5090 / `sm_120` / 170 SM |
| 算子 | 前向 ELU(α=1)；反传 `dx=dy` 或 `dy(y+α)` |
| 有效字节 | 前向 `2*n*4`；反传 `3*n*4`（含 L2） |
| 编译 | `CMAKE_CUDA_ARCHITECTURES=120`；`CUDNN_ROOT`（本机 `~/cudnn-9.12`） |

## 前向主结论（右半段）

| n | vec4_ms | cudnn_ms | vec4/cudnn | vec4 GB/s | cudnn GB/s |
|---:|---:|---:|---:|---:|---:|
| 2^26 | 0.3600 | 0.3559 | 1.012 | 1491 | 1509 |
| 2^27 | 0.7180 | 0.7117 | 1.009 | 1496 | 1509 |
| 2^28 | 1.4358 | 1.4193 | **1.012** | **1496** | **1513** |

反传定点 `n=2^26`：vec4 **0.520 ms** / cudnn **0.514 ms**，`vec4/cudnn`≈**1.013**。

## 怎么读

- 大数组手写 **达到** cuDNN（`vec4/cudnn`≈1.01）。  
- ~**1.50 TB/s**；负半轴 `exp` **未掉出带宽**，与 D-04/D-05 同墙。  
- 左半段 L2 峰值不当总线结论；主看右半段 + `vec4/cudnn`。
