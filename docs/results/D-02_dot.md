# D-02 Dot — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`；新主机复测）  
> 正文：`article/04_operators/D-02*.md`  
> 示例：`examples/04_operators/02_dot.cu`  
> CSV：[D-02_sweep.csv](D-02_sweep.csv)

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| 架构 | `sm_120`，SMs=170 |
| 元素 | float 两向量。`x[i]=i&255`，`y[i]=(3i)&255`。host double = `(n/256)*period`，`period=4624000` |
| block / grid | 256 / `min(SMs*8, ceil(n/256))` |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median**；cuBLAS `POINTER_MODE_DEVICE`，handle/stream 在计时外 |
| 有效字节 | `2 * n * 4`（含 L2） |
| 编译 | `CMAKE_CUDA_ARCHITECTURES=120` |

## 复现命令

```bash
./bin/04_operators_02_dot --mode sweep
```

## Sweep（主结论看右半段）

| n | 字节×2 | scalar_ms | vec4_ms | cublas_ms | scalar/vec4 | vec4/cublas | max_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2^24 | 128 MiB | 0.0882 | 0.0857 | 0.0877 | 1.029 | 0.978 | ~0 |
| 2^26 | 512 MiB | 0.3284 | 0.3232 | 0.3250 | 1.016 | **0.994** | 1.3e-5 |
| 2^27 | 1 GiB | 0.6554 | 0.6392 | 0.6401 | 1.025 | **0.999** | 1.0e-5 |
| 2^28 | 2 GiB | 1.2942 | 1.2709 | 1.2748 | 1.018 | **0.997** | 1.3e-5 |

左半段 median 多低于 0.05 ms，CUBLAS 更慢是 launch/库开销，不当主结论。

## 怎么读

- 点积读两路，有效流量是 D-01 的约两倍。大数组上 `float4` 相对标量只有约 **1.02×**。
- `vec4/cublas` 在 256 MiB 以后贴着 **1.00** = 手写**达到**官方。本章大数组未超过。
