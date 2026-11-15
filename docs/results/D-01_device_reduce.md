# D-01 DeviceReduce — 实测摘要

> 状态：✅ float 阶梯已测（RTX 5090 / `sm_120`；新主机复测）  
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
| 编译 | `CMAKE_CUDA_ARCHITECTURES=120`（须原生 sm_120，勿落到默认 sm_75） |

## 复现命令

```bash
./bin/04_operators_01_device_reduce --mode sweep
```

单路径：`--mode vec4`。定点四条：`--mode modes`。

## Sweep

加速比 = scalar 墙钟 / 该路径墙钟。`vec4/cub` 是手写最好档相对库。GB/s 用 `n * 4` 字节，**含 L2**。

| n | 字节 | scalar_ms | vec4_ms | twopass_ms | cub_ms | scalar/vec4 | vec4/cub | max_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2^16 | 256 KiB | 0.0072 | 0.0061 | 0.0083 | 0.0094 | 1.172 | 0.655 | 0 |
| 2^18 | 1 MiB | 0.0071 | 0.0070 | 0.0073 | 0.0083 | 1.014 | 0.838 | 0 |
| 2^20 | 4 MiB | 0.0071 | 0.0071 | 0.0082 | 0.0081 | 1.009 | 0.870 | 0 |
| 2^22 | 16 MiB | 0.0113 | 0.0084 | 0.0083 | 0.0093 | 1.342 | 0.904 | 0 |
| 2^24 | 64 MiB | 0.0203 | 0.0175 | 0.0183 | 0.0174 | 1.159 | 1.004 | 0 |
| 2^26 | 256 MiB | 0.1889 | 0.1646 | 0.1656 | 0.1660 | **1.148** | **0.992** | 1.0e-5 |
| 2^27 | 512 MiB | 0.3707 | 0.3223 | 0.3247 | 0.3236 | **1.150** | **0.996** | 1.1e-5 |
| 2^28 | 1 GiB | 0.7333 | 0.6381 | 0.6390 | 0.6388 | **1.149** | **0.999** | 1.1e-5 |

## 怎么读

- n≤2^24 时多条路径 median 仍低于 0.05 ms，比值噪声大。CUB 在左半段更慢，是额外 launch。**不当主结论。**
- 工作集掉出缓存以后（256 MiB～1 GiB），`scalar/vec4` 停在 **1.15×** 附近。`vec4/cub` 贴着 **1.00**。
- 手写达到 CUB：`vec4/cub`≈1 = **达标**。本章大数组未超过；超车不在标准 `sum` 上硬抠。
