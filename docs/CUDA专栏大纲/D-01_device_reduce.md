# D-01 写作大纲：Device Reduction

> 状态：✅ **float 阶梯已收口**（新主机 sm_120 复测）。5090：256 MiB～1 GiB，`scalar/vec4` ≈ **1.15×**；1 GiB 上 vec4 / twopass / CUB ≈ 0.638 / 0.639 / 0.639 ms，`vec4/cub`≈**1.00**。max_rel ≈ **1e-5**。int 归档见 [`../results/D-01_sweep_int.csv`](../results/D-01_sweep_int.csv)。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-D.md`](Module-D.md)
>
> **路线**：手写算子。档：scalar → vec4 → twopass → `cub::DeviceReduce::Sum`。达标=达到 CUB；本章大数组未超过。
>
> **硬件门槛**：不限 sm_90+。`cub::DeviceReduce` 随 CUDA toolkit 的 CCCL 头文件。float `atomicAdd` 全架构可用。
>
> **目录**：`article/04_operators/` · `examples/04_operators/01_device_reduce.cu`
>
> **正文目录**：TL;DR → 定义与形状 → 版本表 → 本机数字与判停 → 生产建议 → 下一算子 → 参考文献。  
> **配图**：封面=DeviceReduce→CUB；原理图=版本阶梯；实测图=`scalar/vec4` + `vec4/cub`。

**标题**：`D-01. DeviceReduce：手写求和达到 CUB`

## 与前后章的边界

| 已有章节 | 已覆盖 | D-01 只做 |
|---|---|---|
| C-01 | warp 内 shfl vs SMEM 树，最大约 **1.27×** | block 内 shfl+SMEM 只当积木。不扫 nwarps |
| C-03 | 同址 atomic 争用；SMEM staging **6.31×** | 不做「每元素一次 atomicAdd」 |
| C-04 | grid.sync **17.2×** | 跨 block 用每 block 一次 atomic，或再发一个 kernel |
| B-01 / A-10 | 合并、Roofline | 大数组求和是带宽问题。主看加速比；GB/s 含 L2，不当总线利用率 |
| D-02 | 两向量点积 | 本章只做一元 sum |
| CUB | 库 | `cub` 是天花板，不写库教程 |

## 数据与容差

- 元素是 float。`x[i] = (float)(i & 255)`，0…255 在 float 里是精确整数。
- `n` 是 256 的正倍数。host 真值用 double 闭式：`(n / 256) * (0+…+255) = n * 255 / 2`。
- 设备上是 float 树加每 block 一次 `atomicAdd`。大 n 上 float 的 ulp 盖过「少加一个元素」，所以 float **不能**再证明没丢元素。没丢元素看已归档的 int 校验（和必须等于 `n`）。
- 本章正确性：`abs(got - ref) / abs(ref) <= 1e-3`。四条路径都要过。容差写在章里，不改成「必须逐 bit 相等」。

## TL;DR 目标结论（float sweep 贴上之后改成实测）

1. **Device reduce 读 N 个 float、写出 1 个数。** 每元素 4 字节。墙在加载。
2. **block 内已经用 shfl。** 不重证 C-01。
3. **主对比是加载宽度，以及还要不要第二次 kernel。** `scalar`：标量加载 + 每 block 一次 `atomicAdd`。`vec4`：`float4` 加载 + 同样的 atomic。`twopass`：`float4` 加载，部分和写到 partials，第二个 kernel 收尾。`cub`：`cub::DeviceReduce::Sum`，临时存储在计时外。
4. **int 档已经看到的形状，float 用来核对是不是同一条：** N 够大时 vec4 快于 scalar，约 1.15×；第二次 kernel 贴着 vec4，不更快。Harris 2007 在 G80 上的十余倍不拿来当 5090 的及格线。CCCL 博客在 H200/B200 上、2^16 处约 20% 的确定性差距，也不当这台机器的达标线。
5. **判停**：向量化不再抬带宽就停。`twopass` 相对 `vec4` 贴着 1×，这一档就不当「还要再优化」的结论。校验失败先修归约。禁止 ncu 附着墙钟。

## 建议正文结构

1. 问题：C-01 停在一个 block 里面。整段数组谁来加完。
2. 四条路径。ASCII：标量加载 vs `float4`；atomic 收尾 vs 第二个 kernel。
3. 正确性：double 闭式 + 相对容差 1e-3。int 归档说明没丢元素。float 非结合性就在这一章用容差说完，不另开章。
4. 决策表：大数组先加宽加载；第二次 kernel 不是默认更快；要和库比，比 `DeviceReduce::Sum` 的时间。
5. 实验：`--mode sweep`。int 的 1.15× 作为已测的加载档引用，不把那张表当成 float 的结果。
6. 实测表（float，待贴）。
7. 边界：不重讲 C-01 / C-03。点积留给 D-02。
8. SOP / 误区 / 钩子 → D-02。

## MVP

| 编号 | 产物 | 裁决 |
|---|---|---|
| A | `scalar` / `vec4` / `twopass` / `cub`，元素改 float | **做** |
| B | sweep N = 2^16 … 2^28 | **做** |
| C | host double 闭式，相对误差 ≤ 1e-3 | **做** |
| D | fp8 归约 | **不做**（额外档。f32 阶梯先收口。fp8 累加仍进 float，带宽形状不同，不挡这一章） |
| E | 每元素 atomic、空 grid.sync、扫 CUB 三种 determinism | **不做** |
| F | 新 NCU/SASS | **不做**（先有裸跑 median） |

网格默认 `min(SMs*8, ceil(n/256))`，block 固定 256。`scalar` / `vec4` 的计时含一次 4 字节 `memset`（清累加器）。`twopass` 和 `cub` 由最后一次写覆盖输出。`cub` 的临时存储分配在计时外。

有效带宽 = `n * 4 / ms`，含 L2。

主命令：

```bash
./bin/04_operators_01_device_reduce --mode sweep
```

CSV：`n,scalar_ms,vec4_ms,twopass_ms,cub_ms,scalar_over_vec4,scalar_over_twopass,scalar_over_cub,max_rel`。

## 参考文献池

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUB `DeviceReduce`](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html) | 工作量线性；默认 `run_to_run` 是固定树，不用 atomic | `cub` 模式在对照什么 |
| A 官方 | [CCCL 3.1 determinism](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/) | `not_guaranteed` 单 kernel + atomic，小数组上更快；大数组差距变小。H200/B200 上 2^16 约快 20% | 解释「单次 atomic 不等于永远更快」，不当 5090 达标线 |
| B 工程 | [Harris, Optimizing Parallel Reduction in CUDA (2007, PDF)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) | 规约是带宽问题。G80 上 interleaved → 展开，带宽从约 2 GB/s 到约 30 GB/s | 只借「先合并加载、再减指令」的顺序。倍数不搬 |
| B 工程 | [Harris, Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) | shuffle 代替 SMEM warp 归约；block 部分和可以原子收尾 | shuffle 已经在 C-01。本章用它，不再当新结论 |
| C 实证 | Wilt, *The CUDA Handbook* ch.12（[在线节](https://cudahandbook.com/book/ch12/optimizing-for-bandwidth)） | RTX 3060、64M int：两遍 316 GB/s，int4 单次原子 339 GB/s，CUB 340 GB/s | 「向量化加载 + 单次收尾可以贴到库」。机器不是 5090 |

明确不引入：Harris G80 的 15× 累计加速、OpenMP offload 在 Grace-Hopper 上的倍数（那是另一条基线）。
