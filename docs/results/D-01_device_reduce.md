# D-01 Device Reduction — 实测摘要

> 状态：✅ float 阶梯已测（RTX 5090 / `sm_120`）  
> 正文：`article/04_operators/D-01*.md`  
> 示例：`examples/04_operators/01_device_reduce.cu`  
> CSV：[D-01_sweep.csv](D-01_sweep.csv)  
> 归档：[D-01_sweep_int.csv](D-01_sweep_int.csv)（int32 全 1，证明没丢元素；本章主结论用 float）

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| 架构 | `sm_120`，SMs=170 |
| 元素 | float，`x[i] = (float)(i & 255)`。host double 真值 `n * 255 / 2`。相对误差 ≤ 1e-3 |
| block | 256 |
| grid | `min(SMs*8, ceil(n/256))`。大 n 上是 1360 |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median**；event 与 kernel **同 stream** |
| 计时里有什么 | `scalar` / `vec4` 含一次 4 字节 `memset`。`twopass` / `cub` 由最后一次写覆盖输出。CUB 临时存储在计时外分配 |

## 复现命令

```bash
./bin/04_operators_01_device_reduce --mode sweep
```

单路径：`--mode vec4`（只跑这一条）。定点四条：`--mode modes`。

`CMAKE_CUDA_ARCHITECTURES=120`。增删 `.cu` 后先重跑 cmake。

## Sweep

加速比 = scalar 墙钟 / 该路径墙钟。`vec4/cub` 是手写最好档相对库。GB/s 用 `n * 4` 字节，**含 L2**。

| n | 字节 | scalar_ms | vec4_ms | twopass_ms | cub_ms | scalar/vec4 | vec4/cub | max_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2^16 | 256 KiB | 0.0170 | 0.0139 | 0.0180 | 0.0236 | 1.224 | 0.589 | 0 |
| 2^18 | 1 MiB | 0.0206 | 0.0136 | 0.0178 | 0.0254 | 1.513 | 0.535 | 0 |
| 2^20 | 4 MiB | 0.0147 | 0.0137 | 0.0177 | 0.0231 | 1.075 | 0.593 | 0 |
| 2^22 | 16 MiB | 0.0177 | 0.0157 | 0.0172 | 0.0238 | 1.122 | 0.660 | 0 |
| 2^24 | 64 MiB | 0.0327 | 0.0249 | 0.0239 | 0.0244 | 1.311 | 1.020 | 0 |
| 2^26 | 256 MiB | 0.1972 | 0.1748 | 0.1710 | 0.1747 | **1.128** | **1.001** | 1.0e-5 |
| 2^27 | 512 MiB | 0.3805 | 0.3321 | 0.3316 | 0.3321 | **1.146** | **1.000** | 1.1e-5 |
| 2^28 | 1 GiB | 0.7461 | 0.6463 | 0.6452 | 0.6446 | **1.154** | **1.003** | 1.1e-5 |

## 怎么读

- n≤2^24 时多条路径 median 仍低于 0.05 ms，比值噪声大。CUB 在左半段更慢，是额外 launch。**不当主结论。**
- 工作集掉出缓存以后（256 MiB～1 GiB），`scalar/vec4` 停在 **1.13×～1.15×**。`vec4/cub` 贴着 **1.00**。
- 有效带宽（`vec4`）：64 MiB **2.69 TB/s**（仍偏缓存），1 GiB **1.66 TB/s**。含 L2，不当总线利用率。
- float 相对误差最大约 **1e-5**。没丢元素看归档的 int 校验。

## 归档 int 档

`D-01_sweep_int.csv`：int32 全 1，和必须等于 `n`。256 MiB 以后 scalar/vec4 约 **1.15×**，形状与 float 一致。
