# D-02 写作大纲：Dot Product

> 状态：✅ 已收口（新主机 sm_120 复测）。5090：大数组 `vec4/cublas`≈**1.00**；`scalar/vec4`≈**1.02×**（双流收益小于 D-01）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-D.md`](Module-D.md) · 上章：[`D-01_device_reduce.md`](D-01_device_reduce.md)
>
> **路线**：手写算子。档：scalar → float4 → `cublasSdot`。达标=达到库；本章大数组未超过。
>
> **硬件门槛**：不限 sm_90+。需链 `CUDA::cublas`。
>
> **目录**：`article/04_operators/` · `examples/04_operators/02_dot.cu`
>
> **正文目录**：算子卡六节。配图：封面=Dot→cublasSdot；原理图=两路读；实测图=`vec4/cublas`。

**标题**：`D-02. Dot：手写点积达到 cublasSdot`

## 与前后章的边界

| 已有章节 | 已覆盖 | D-02 只做 |
|---|---|---|
| D-01 | 一元 sum、跨 block 收尾、贴 CUB | 两个向量；不重讲 device reduce 课 |
| D-19 SGEMV | 矩阵×向量 | 本章不是 GEMV。形状是 `dot(x,y)`，不是 `y = Ax` |
| C-01 | warp shfl | block 内归约积木沿用，不扫 nwarps |
| cuBLAS | 库 | `cublasSdot` 是天花板，不写库教程 |

## 数据与容差

- `x[i] = (float)(i & 255)`，`y[i] = (float)((i * 3) & 255)`。`n` 是 256 的正倍数。
- host 真值：对每个周期闭式累加进 double，或一次 host 双精度扫（n 到 2^28 时闭式更干净）。用周期和：对 k=0..255，贡献 `(k)*((3k)&255)` 再乘 `n/256`。实现里用 double 扫一遍 host 参考也可（分配在计时外）；为省事且可复现，用 device 结果对 **host double 逐元素累加** 的参考，容差相对误差 ≤ 1e-3。
- 手写路径：grid-stride 乘加 → block_sum（同 D-01）→ 每 block 一次 `atomicAdd`。
- `cublasSdot`：pointer mode device，结果在 device；workspace 无。handle 创建在计时外。

## TL;DR 目标结论（有数后改）

1. 点积读 2N 个数、写 1 个。有效字节 `2 * n * 4`。墙仍在加载。
2. 双流之后 `float4` 相对标量约 **1.02×**（小于 D-01 的 1.15×）。
3. 大数组上 `vec4/cublas` ≈ **1.00**；贴着就停。
4. 数值对 host double，rtol 1e-3。
5. 不做 GEMV、不做半精度、不扫 TF32。

## 建议正文结构

1. 问题：D-01 是一元和；点积是两个流上的乘加再归约。
2. 路径：scalar / vec4 / cublas。
3. 决策表：加宽加载；贴库就停；要矩阵×向量去 D-19。
4. 实验 + 实测。
5. 边界 / SOP / 钩子 → D-03 Elementwise。

## MVP

| 编号 | 产物 | 裁决 |
|---|---|---|
| A | `scalar` / `vec4` / `cublas` | **做** |
| B | sweep N = 2^16 … 2^28 | **做** |
| C | host double 参考，rtol 1e-3 | **做** |
| D | GEMV、半精度、twopass | **不做**（twopass 在 D-01 已证明大数组不赚） |
| E | NCU | **不做** |

网格 / block 同 D-01。有效带宽 = `2 * n * 4 / ms`（含 L2）。

主命令：

```bash
./bin/04_operators_02_dot --mode sweep
```

CSV：`n,scalar_ms,vec4_ms,cublas_ms,scalar_over_vec4,scalar_over_cublas,vec4_over_cublas,max_rel`

## 参考文献池

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [cuBLAS `cublasSdot`](https://docs.nvidia.com/cuda/cublas/#cublas-t-dot) | 单精度点积；可 device pointer 出结果 | 天花板调用 |
| A 官方 | [cuBLAS pointer mode](https://docs.nvidia.com/cuda/cublas/#cublassetpointermode) | `CUBLAS_POINTER_MODE_DEVICE` 时结果在 device | 计时口径 |
| B 工程 | D-01 本机结论 | 大数组向量化 ~1.15×，贴库可停 | 对照一元归约 |
| C 实证 | Wilt Handbook ch.12 带宽章（见 D-01 §9） | 向量化加载贴库是带宽题常见结局 | 形状预期，非 5090 数 |

明确不引入：把点积写成 SGEMV 教程；用 `cublasSgemv` 冒充 dot。
