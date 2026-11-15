# Module D 大纲：算子题单（31–89）

> 状态：🟡 主流算子题单。**58 个算子**（D-01～D-58）加 Checklist **D-59**。D-01～D-04 ✅。其余未开。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §2 全景 · §4.4 · D-01：[`D-01_device_reduce.md`](D-01_device_reduce.md)
>
> dtype 和打包宽度写在同一章的版本表里。D-20 HGEMV 是例外，见下节。Triton / CuTe 对同一算子的另一种实现不单开章。

## 题单怎么算

58 个算子，每题一章。转置在 B-09，warp reduce 在 C-01，这里不重开。GEMM 的反传收进 D-21 / D-22 换布局的一档。卷积的反传是 D-40，单开。

f32 / f16 / bf16 / fp8 以及 x2、x4、x8 写进同一章的版本表。FP8 归约是 D-01 的一档。FP8 GEMM、FP4 GEMM、W4A16 数据通路不同，单独成章。D-20 单独成章，因为 K 方向打包之后，加载指令和 SGEMV 不同。

## 教法弧

```text
归约 D-01～D-02
  → 逐点与索引 D-03～D-14
  → 归一化 D-15～D-17
  → NMS D-18
  → 矩阵 D-19～D-22
  → 预填 Attention D-23～D-25
  → 选择与扫描 D-26～D-29
  → 门控与视觉归一化 D-30～D-32
  → 归一化反传 D-33
  → 分组 GEMM、低精度与卷积 D-34～D-41
  → 解码 Attention 与反传 D-42～D-45
  → MoE 路由与搬运 D-46～D-48
  → 采样与量化 D-49～D-50
  → 损失与强化学习 D-51～D-55
  → Dropout、优化器与网格采样 D-56～D-58
  → Checklist D-59
```

全景 §2 的五个词仍是这一层的入口标签。Conv、MoE、解码、训练和强化学习从本表进入。章号是目录顺序，不是学习顺序。

## D 怎么定位

**D = 手写算子。** 一章一个有名字的 op，版本阶梯自己写，用本机 median 对照官方实现（CUB / cuBLAS / cuDNN 等；无单算子库则对照有效带宽）。

| 水位 | 含义 | 算不算达标 |
|---|---|---|
| **达到官方** | 手写最好档与库同量级（比值≈1） | **达标**——手写成功 |
| **超过官方** | 手写稳定快于默认库档（有缝：融合、怪 shape、非默认确定性等） | **超车**——D 的进取目标 |
| 明显慢于官方 | 版本阶梯没走完，或计时/正确性有坑 | 未达标，先修手写 |

A/B/C 的 `.cu` 是机制教具。D 的 `.cu` 就是那个算子的实现。  
「生产可调库」= 标准形状上你已经写到与库同档，调用库是**同水位的封装**，不是「手写失败改去调库」。有缝、要融合、库覆盖不到时，继续用手写（或手写超车）。

一章一个算子。正文用**算子卡**。点算子通常两档就够（标量、向量化）。

### 前三章（D-01～D-03）：先练到「达到」

标准带宽/规约题上，5090 大数组常见结局是**贴库**（达到，未超过）。这三章把「达到官方」练成肌肉记忆：

| 章 | 手写对象 | 官方对照 | 本机大数组 |
|---|---|---|---|
| D-01 | `sum(x)` | CUB `DeviceReduce` | `vec4/cub` ≈ 1.00 → **达到** |
| D-02 | `dot(x,y)` | `cublasSdot` | `vec4/cublas` ≈ 1.00 → **达到** |
| D-03 | `a+b` | 有效带宽（无单算子） | ~1.55 TB/s 不再涨 → **达到带宽顶** |

「想赢默认 CUB」在这三章大数组上通常没有稳定缝——不是 D 不超车，是**这类形状库已经摸到带宽墙**。超车放到后面有缝的章。

## 算子卡与配图

正文目录固定：

```text
TL;DR → 定义与形状 → 版本表 → 本机数字与判停 → 生产建议 → 下一算子 → 参考文献
```

| 配图 | 画什么 | 不画什么 |
|---|---|---|
| 封面 | 手写算子 → 官方对照（parity / ceiling） | 「调库就好、手写多余」海报 |
| 原理图 | 版本阶梯，或本章流量形状（几读几写） | 与结论无关的微架构解剖 |
| 实测图 | `scalar/vec4`、`vec4/库`、或有效带宽随 n | 把噪声左半段当主结论 |

短章（D-04～D-10）更薄：公式、相对 D-03、是否算力墙。

## 章内版本表

D-22、D-23、D-45 的档都写在同一章里。正文可以分成上、下两篇，章号不变，不插入新号，后面的总序不动。上一档已经贴着库，后面的档不测。

| 章 | 节 |
|---|---|
| D-22 | naive → smem tile → WMMA → `mma` 加 warp tile → 多级流水 → smem swizzle → 写回（含 bias） |
| D-23 | 物化分数矩阵 → split-KV → split-Q → K/V 共用 smem → 沿序列维切块 |
| D-45 | 保存每行 max 与 sum → 重算概率 → dV、dK、dQ |

## 短章

D-04～D-10 共用 D-03 的网格、block、字节数和 median，只换函数。正文写公式、相对 D-03 的带宽，以及特殊函数有没有把算子打成算力瓶颈。向量化加载不在这七章重讲。这七章和 D-30 各加一档反传：输入是前向激活和上游梯度，输出是对输入的梯度。网格仍共用 D-03。不因为反传再开章。

D-53、D-55 也是短章。它们是逐元素再加一次归约，只写公式和 mask。归约回 D-01，组内均值方差不在这两章重算。D-52、D-54 是完整章。

## 证据

主证据是裸跑 CUDA event **median**。有效 GB/s 含 L2，正文分开写缓存内和已经掉进显存。GEMM / Attention 报 TFLOPS。及格线是本机测到的比例。随机算子的正确性另见后文，速度仍用 median。

## 官方对照

没有一个库覆盖这 58 个算子。官方入口是三家。章节表的「天花板」和本节是同一套，不另写一套。

| 库 | 对哪些章 | 调用 |
|---|---|---|
| CUB（CCCL） | D-01 Reduce、D-11 Histogram、D-26 ArgMax、D-28 Scan、D-29 Radix Sort | `DeviceReduce` / `DeviceHistogram` / `DeviceScan` / `DeviceRadixSort`。ArgMax 用 `DeviceReduce` 的 ArgMax。D-27 TopK 的库对照是先排序再取前 k，不假设存在单独的 TopK API |
| cuBLAS / cuBLASLt | D-02 Dot、D-19 SGEMV、D-20 HGEMV、D-21 SGEMM、D-22 HGEMM、D-34 Grouped GEMM、D-35 FP8 GEMM、D-36 FP4 GEMM | `cublasSdot` / `cublasSgemv` / `cublasLtMatmul` / grouped。GEMM 章以 `cublasLtMatmul` 为天花板。章里写明 compute type：FP32 严格算，还是允许 TF32 |
| cuDNN | D-15 Softmax、D-16 LayerNorm、D-31 BatchNorm、D-41 Pool、D-23 的稠密预填、D-56 Dropout 的时间列 | 前向用对应描述符。算法写明是 heuristic 还是 benchmark。D-23 只在 QKV 是一块稠密块时和 SDPA 比。D-38 的必做对照是 im2col 加 D-21，cuDNN 是加列，没有不挡成章。D-17 RMSNorm 有对应描述符就加一列 cuDNN。D-56 的正确性对 CPU 同一种子，不对 cuDNN 的随机数 |

这三家之外的题，正确性对 host（double 或逐元素参考），速度只和自己的上一档比。容易被当成有官方单算子的，格子里写「无官方单算子」：D-12 Embedding、D-13 Embedding 反传、D-14 RoPE、D-30 SwiGLU、D-37 W4A16、D-42 分页解码、D-43 MLA、D-44 写页、D-46～D-48 的 router / dispatch / combine、D-49 Top-p、D-52～D-55 的 logprob / KL / GAE / 策略损失。W4A16 的库对照是先把 int4 展开成 f16，再调一次 HGEMM，不把这个展开后的 GEMM 当成同一种核。逐点激活、卷积反传、AdamW、Grid sample 同样没有官方单算子，天花板写 host 或带宽。

计时和手写核用同一套：

- 同一块设备内存、同一形状、同一 dtype。
- 库的 workspace 查询和分配放在计时外面。
- warmup 之后取 CUDA event 的 median。库的毫秒和手写核的毫秒各一列，再给比值。
- 正确性先过：和库的输出或 host 参考比，容差写在该章。过了再读快慢。
- **达到官方（比值≈1）= 达标**；稳定超过默认库档 = 超车。未达到则先修手写。
- **官方只对照，不讲课**：点名 API、计时/确定性边界各一两句，链到官方文档；不写 CUB/cuBLAS 教程。

仓库现在链的是 cudart 和 CUB 头文件。开到 cuBLAS、cuDNN 那一章时再链对应库。机器上没有 cuDNN 时，D-38 仍用 im2col 加 D-21 的 GEMM 做对照，章节照写。

## 训练、推理、模态

同一个算子在训练和推理里不是同一次调用。框架做的是挑选核、把相邻算子焊在一起。D 写单算子，不写框架的调度。调度、CUDA Graph、多卡归到 E。

训练侧，PyTorch 把卷积、BatchNorm、Pool 交给 cuDNN，把大 GEMM 交给 cuBLASLt。注意力在 Megatron 里由 Transformer Engine 选：FlashAttention，或 cuDNN 的 fused attention，并带上反传。低精度训练走 Transformer Engine 的 FP8 / FP4。LayerNorm 与 Linear、bias-dropout-add 是焊在一起的核。优化器步进是另一支核。这些反传和融合，题单里已有对应章（D-33、D-40、D-45、D-51、D-56、D-57）。激活和 RoPE 的反传是前向章里的一档，见「短章」和 D-14。

强化学习的 rollout 复用解码路径，变长打包用 D-23 的 `cu_seqlens` 档。单独会发出去的是四支核：按 token 取 logprob，含被抽中 token 的梯度（D-52）、policy 与 reference 的 KL（D-53）、GAE 回报（D-54）、策略损失（D-55，含 PPO clip、GRPO 的加权、DPO）。组内均值方差用 D-01，不在 D-55 重算。

推理侧，vLLM 按形状挑后端，不固定一家。注意力走 FlashInfer，后端可以是 FlashAttention、cuDNN、CUTLASS 或 TensorRT-LLM 的核，KV 在页里。MoE 的 `moe_backend` 在 Triton、CUTLASS、FlashInfer、Marlin、DeepGEMM 之间切：大 batch 用 grouped GEMM，Blackwell 上很小的 batch 会退回单 kernel，因为 grouped 的启动划不来。W4A16 走 Marlin 或 FlashInfer 的量化 MoE。`silu_and_mul`、RoPE 是单独焊上的小核。解码用 CUDA Graph 把启动吃掉。SGLang、TensorRT-LLM 也调用 FlashInfer，核的来源同一类。

模态改的是形状和哪一段在关键路径上，不改算子名单。

| 阶段 | 关键路径 | 对照时用的形状 |
|---|---|---|
| 训练一步 | GEMM 与注意力的前向加反传，再加 AdamW | 大 batch、带反传的那一列 |
| 强化学习 | rollout 用解码；更新用 logprob、KL、GAE、策略损失 | D-52～D-55。logprob 不对着整张词表物化。组内优势用 D-01 |
| 预填 | 方阵 GEMM + 长序列注意力 | cuBLAS / cuDNN SDPA 的大矩阵 |
| 解码 | 瘦 GEMM 或 GEMV + 分页注意力 | 每条请求一个新 token。量化路径另加 W4A16 那一列，不拿方阵 HGEMM 当这一列的天花板 |
| MoE | grouped GEMM + dispatch + combine | 大 batch 与小 batch 各一列。小 batch 上 grouped 变慢要写出来 |
| 卷积网 | cuDNN 的卷积、归一化、池化 | 前向一列，训练再加反传 |
| 视觉编码器（ViT、VLM 的图像塔） | 与语言模型相同的 GEMM 和注意力，一次算完，没有 KV 页 | 序列长度按图像 token 计。启动占比高时，单算子时间之外再记一句，Graph 本身不在 D 展开 |
| 扩散 | 每步都是全序列注意力，或 UNet 的卷积 + GroupNorm | 复用 D-23 与 D-38 / D-32，不单开模态章 |

一篇里若两个阶段都重要，天花板写两列，各用各的形状。不因为模态再增加题目。

目录：`article/04_operators/` · `examples/04_operators/`。开写前先有该章 `D-0N_*.md`。

## 章节总表

| 总序 | 章 | 算子 | 章内至少这些档 | 天花板 | 状态 |
|---|---|---|---|---|---|
| 31 | **D-01** | Reduce | block all-reduce → device 多 block → 向量化加载。f32 为主，fp8 只作为额外一档 | CUB `DeviceReduce` | ✅ 5090 大数组 **1.15×**，贴 CUB |
| 32 | **D-02** | Dot | 两向量点积。标量 → 向量化。不写成 GEMV | `cublasSdot`。数值另对 host double | ✅ 5090 大数组 `vec4/cublas`≈1.00；scalar/vec4≈1.02× |
| 33 | **D-03** | Elementwise | 逐元素加（只测 add）。标量 → `float4`（half2 不做）。vec4 与 scalar 同 CTA 数 | 有效 GB/s | ✅ 5090 大数组 ~1.55 TB/s，`scalar/vec4`≈1.03× |
| 34 | **D-04** | Sigmoid | 标量 → 向量化。特殊函数的算力是否盖过带宽 | 有效 GB/s 或与 elementwise 的差距 | ✅ 5090 大数组 ~1.50 TB/s，仍贴带宽；`scalar/vec4`≈1.09× |
| 35 | **D-05** | ReLU | 标量 → 向量化 | 有效 GB/s | ⏳ |
| 36 | **D-06** | ELU | 标量 → 向量化 | 有效 GB/s | ⏳ |
| 37 | **D-07** | GELU | 标量 → 向量化。tanh 近似与 erf 的数值差写明 | host 参考 + 带宽 | ⏳ |
| 38 | **D-08** | Swish | 标量 → 向量化 | host 参考 + 带宽 | ⏳ |
| 39 | **D-09** | HardSwish | 标量 → 向量化 | 有效 GB/s | ⏳ |
| 40 | **D-10** | HardShrink | 标量 → 向量化 | 有效 GB/s | ⏳ |
| 41 | **D-11** | Histogram | 全局原子 → smem 私有化再合并。原子课回链 C-03，不重讲 | CUB `DeviceHistogram` | ⏳ |
| 42 | **D-12** | Embedding | 按行 gather。标量 → 向量化搬运 | 无官方单算子。有效 GB/s | ⏳ |
| 43 | **D-13** | Embedding 反传 | 按 index 把梯度原子加回权重行。前向在 D-12。同一行有冲突时结果仍要对上 | 无官方单算子 | ⏳ |
| 44 | **D-14** | RoPE | 标量旋转 → 向量化。反传是角度取反的一档。和逐点激活不是同一题 | 无官方单算子 | ⏳ |
| 45 | **D-15** | Softmax | naive → 减 max → online `(m, ℓ)`。注意力里的反传在 D-45，交叉熵的 dlogits 在 D-51 | cuDNN。naive 会出 Inf，safe 不出 | ⏳ |
| 46 | **D-16** | LayerNorm | 两遍（均值、方差）→ 向量化统计 | cuDNN | ⏳ |
| 47 | **D-17** | RMSNorm | 不减均值的那一版。各写各的统计量。QK-norm 是注意力前的一次调用，不单开 | host。有 cuDNN 描述符时加一列 | ⏳ |
| 48 | **D-18** | NMS | 框上的抑制。正确性先于带宽 | host 参考 | ⏳ |
| 49 | **D-19** | SGEMV | K 方向标量 → 向量化 → 仍带宽不够才 smem | `cublasSgemv`。报带宽，不报 TFLOPS | ⏳ |
| 50 | **D-20** | HGEMV | f16 矩阵向量乘。K 方向按 8 或 16 个数打包加载，加载指令和 SGEMV 不同 | cuBLAS。报带宽，并相对 D-19 | ⏳ |
| 51 | **D-21** | SGEMM | naive → smem tile → 向量化或 padding，二选一留下还在涨的那一档。反传是换布局的一档 | `cublasLtMatmul`。写明 compute type：FP32 严格算，还是允许 TF32 | ⏳ |
| 52 | **D-22** | HGEMM | 节次见「章内版本表」。反传是换布局的一档。全部在本章 | `cublasLtMatmul`。写明 compute type | ⏳ |
| 53 | **D-23** | FlashAttention | 节次见「章内版本表」。预填，query 与 KV 都长。变长打包（`cu_seqlens`）是一档，推理 batch 和 RL rollout 用它。全部在本章 | 稠密块对 cuDNN SDPA。变长对 host，或同等 token 数的稠密核。后几节报 TFLOPS | ⏳ |
| 54 | **D-24** | Split-D Attention | 沿 head 维切开，使 SRAM 不随 headdim 线性增长。持久化 kernel 是其中一档 | 相对 D-23 在大 headdim 上的时间 | ⏳ |
| 55 | **D-25** | Merge attention states | 把分段 attention 的输出按 log-sum-exp 拼起来 | host 参考 | ⏳ |

激活（D-04～D-10）的写法见「短章」。

## 主流补充（D-26～D-58）

这 33 题是推理、训练步进、强化学习、卷积网里会单独发出去的 kernel。形状参数不另开章，见「不做什么」。

| 总序 | 章 | 算子 | 这一题和已有章的差别 | 天花板 | 状态 |
|---|---|---|---|---|---|
| 56 | **D-26** | ArgMax | 要下标，不只是和。贪心解码用它 | CUB `DeviceReduce` ArgMax | ⏳ |
| 57 | **D-27** | TopK | 选择。MoE router 也会用到 | 先 D-29 排序再取前 k。不假设存在单独的 TopK API | ⏳ |
| 58 | **D-28** | Scan | 前缀和。top-p 的 CDF、stream compaction 都靠它 | `cub::DeviceScan` | ⏳ |
| 59 | **D-29** | Radix Sort | 按位分桶。TopK 可以不依赖完整排序，这一题单独把排序写完 | `cub::DeviceRadixSort` | ⏳ |
| 60 | **D-30** | SwiGLU | `silu(x) * y`，两段张量。Swish 只有一段。反传是一档，见「短章」 | 无官方单算子。带宽 | ⏳ |
| 61 | **D-31** | BatchNorm | 统计量在 batch 上，推理时用冻结的均值方差 | cuDNN | ⏳ |
| 62 | **D-32** | GroupNorm | 统计量在 group 上。InstanceNorm 是 group 数等于通道数的一档 | host 参考 | ⏳ |
| 63 | **D-33** | Norm 反传 | LayerNorm、RMSNorm、BatchNorm、GroupNorm 的输入梯度。BatchNorm 与 GroupNorm 另算 dγ、dβ。前向在 D-16、D-17、D-31、D-32 | host 参考 | ⏳ |
| 64 | **D-34** | Grouped GEMM | 一组大小不同的 GEMM，MoE 专家。调度和单次 HGEMM 不同 | cuBLAS grouped。章内另报一次分组调用相对逐专家启动。小 batch 上分组变慢要写出来 | ⏳ |
| 65 | **D-35** | FP8 GEMM | 缩放因子进累加。数据通路和 HGEMM 不同 | cuBLAS FP8 的 TFLOPS 比例 | ⏳ |
| 66 | **D-36** | FP4 GEMM | Blackwell 上的 4-bit Tensor Core。Tensor Memory 只作为这一题的机制，不单开章 | cuBLAS FP4 的一列比例。编译不了就停并写明 | ⏳ |
| 67 | **D-37** | W4A16 GEMM | 权重以 int4 存放，寄存器里反量化再乘 f16 激活 | 无官方单算子。相对先反量化再 HGEMM 的时间。展开后的 GEMM 不是同一种核 | ⏳ |
| 68 | **D-38** | Conv | implicit GEMM。空间维进 K。对照计时用物化 im2col 加上 D-21 的 GEMM。机器上有 cuDNN 时加一列，没有不挡成章 | 相对 im2col + D-21 的时间 | ⏳ |
| 69 | **D-39** | Depthwise Conv | 每个通道自己的核。和 D-38 的访存不一样 | host 参考 + 带宽或时间 | ⏳ |
| 70 | **D-40** | Conv 反传 | 输入梯度用 col2im 或 implicit GEMM；权重梯度是 im2col 之后的 GEMM。前向在 D-38 / D-39 | host 参考 | ⏳ |
| 71 | **D-41** | Pool | Max 与 Avg 两档。窗口归约 | cuDNN | ⏳ |
| 72 | **D-42** | Paged decode attention | 一条 query，KV 在页表里。预填在 D-23；多页的部分和用 D-25 拼 | 无官方单算子。长 KV 报时间 | ⏳ |
| 73 | **D-43** | MLA | 压缩后的 KV 上做注意力。分数的形状和 D-23 不同 | 无官方单算子 | ⏳ |
| 74 | **D-44** | Reshape-and-cache | 把新的 K/V 写进页。索引 + 搬运 | 无官方单算子。有效 GB/s，写完位置正确 | ⏳ |
| 75 | **D-45** | Attention 反传 | 节次见「章内版本表」。前向在 D-23 | host 参考 | ⏳ |
| 76 | **D-46** | MoE router | 专家维 softmax 加 top-k。TopK 本体在 D-27 | 无官方单算子 | ⏳ |
| 77 | **D-47** | MoE dispatch | 按专家号把 token 搬进各组。gather / scatter | 无官方单算子。正确的分组 + 带宽 | ⏳ |
| 78 | **D-48** | MoE combine | 按路由权重把专家输出加回 token 原序。去程在 D-47 | 无官方单算子。带宽 | ⏳ |
| 79 | **D-49** | Top-p | 对 CDF 截断再抽样。Scan 在 D-28。投机解码的接受或拒绝是一档：逐位置比较 draft 与 target 的概率。正确性见「随机算子」 | 无官方单算子。同一种子下与 CPU 参考逐元素一致；另报保留概率质量 | ⏳ |
| 80 | **D-50** | FP8 quant | 按行或按张量算 scale，量化再反量化。GEMM 在 D-35 | 数值范围 + 带宽 | ⏳ |
| 81 | **D-51** | Cross entropy | log-softmax 与 NLL 一次做完，并给出 dlogits = softmax(logits) − label。标签平滑是一档。只取一个 token 的 logprob 在 D-52 | 无官方单算子。host 参考，含梯度 | ⏳ |
| 82 | **D-52** | Token logprob | 只取给定 token 的 log p，不物化整张词表。反传只在被抽中的 token 上散射梯度。整词表交叉熵在 D-51 | 无官方单算子。时间要短于先物化整张词表再 gather | ⏳ |
| 83 | **D-53** | KL | 短章。policy 与 reference 两条 logprob。k1 与 k3 各一档，再做 mask 后的均值。归约回 D-01 | 无官方单算子 | ⏳ |
| 84 | **D-54** | GAE | 沿时间反向的折扣扫描。前向扫描在 D-28。停的时候要看到相对逐步 kernel 的时间 | 无官方单算子 | ⏳ |
| 85 | **D-55** | 策略损失 | 短章。PPO 的 clip、GRPO 的加权 logprob、DPO 的 log-sigmoid 各一档。组内均值方差用 D-01，不在本章重算 | 无官方单算子 | ⏳ |
| 86 | **D-56** | Dropout | Philox 随机数 + 掩码。正确性见「随机算子」 | 同一种子下掩码与 CPU 参考逐元素一致。cuDNN 只作时间加列 | ⏳ |
| 87 | **D-57** | AdamW | 动量、二阶矩、解耦权重衰减。逐元素，状态在显存里 | host 参考 | ⏳ |
| 88 | **D-58** | Grid sample | 双线性采样。规则网格上采样是其中一档 | host 参考 | ⏳ |
| 89 | **D-59** | Checklist | 症状 → 打开哪一个算子 → 库那一列 | 无新 `.cu` | ⏳ |

## D-01 收口

float 阶梯已收口。`x[i] = (float)(i & 255)`，host double 真值 `n * 255 / 2`，相对误差 ≤ 1e-3（本机约 1e-5）。256 MiB～1 GiB：`scalar/vec4` = **1.128× / 1.146× / 1.154×**；1 GiB 上 vec4 / twopass / CUB 贴在 0.645 ms。int 归档在 `docs/results/D-01_sweep_int.csv`。fp8 这一档不做。

现有文件仍叫 `01_device_reduce`。章号保持 D-01，不因为题单变长就改名。

## 及格线

| 章 | 停法 |
|---|---|
| D-01、D-19、D-20 | 向量化不再抬带宽就停。D-19、D-20 同时报相对 cuBLAS 的时间 |
| D-02 | 向量化不再贴近 `cublasSdot` 就停。数值对 host double |
| D-03、D-12 | 向量化相对标量的带宽 |
| D-04～D-10、D-30 | 相对 D-03 的带宽。特殊函数把算子打成算力瓶颈时写明。反传档和 host 梯度对上 |
| D-11、D-26、D-28、D-29 | 和对应的 CUB 对上，并看到相对上一档还动不动 |
| D-14、D-18、D-25、D-32、D-43、D-58 | 无官方单算子。和 host 参考对上。快慢是第二列 |
| D-15 | naive 会出 Inf；safe 不出。稠密形状报相对 cuDNN 的时间 |
| D-16、D-17 | 统计量不同，各写各的。D-16 报相对 cuDNN 的时间。D-17 有描述符才加这一列 |
| D-21、D-22、D-34～D-36 | 报本机相对 cuBLAS 的 TFLOPS 比例，并写明 compute type。D-34 另报相对逐专家启动 |
| D-23、D-24 | 分数矩阵是否还在显存里。稠密预填报相对 cuDNN SDPA 的时间。变长对 host 或同等 token 数的稠密核。大 headdim 上 D-24 相对 D-23 的时间 |
| D-27 | 和「先排序再取前 k」对上 |
| D-31、D-41 | 报相对 cuDNN 的时间 |
| D-37～D-39、D-42、D-44、D-46～D-48、D-50 | 无官方单算子，或库对照不是同一种核。和 host 对上，并写明带宽或时间相对上一档还动不动 |
| D-13、D-33、D-40、D-45、D-51、D-57 | 梯度或状态更新和 host 参考对上 |
| D-52 | 前向和被抽中 token 的梯度都对上 host。时间短于先物化整张词表再 gather |
| D-53、D-55 | 短章。公式对上 host。归约回 D-01 |
| D-54 | 反向扫描对上 host，并报相对逐步 kernel 的时间 |
| D-49、D-56 | 见「随机算子」。速度仍报 median。D-56 的 cuDNN 列只比时间 |

## 随机算子

D-49 Top-p、D-56 Dropout。

- 种子固定。Dropout 用 Philox，计数器按元素编号。
- 同一种子下，掩码或样本与 CPU 参考逐元素一致。Top-p 另报保留下来的概率质量与参考的差。
- 速度仍是裸跑 median。

## 不做什么

- 不重写 B-09 的转置、C-01 的 warp reduce。
- 不写 Triton、CUTLASS、CuTe 教程。对照库时只调用一次。
- 不把算子接进 PyTorch。那是 Module E。
- GQA / MQA、因果掩码写在 D-23 / D-42 的参数里。batched GEMM、split-K 写在 D-21 / D-22 的一档。GEGLU 写在 D-30，换门控函数。InstanceNorm 写在 D-32，group 数等于通道数。
- GEMM 反传不单开。激活反传、RoPE 反传不单开。cat / slice 回 B。AllReduce 在 E-07。SpMM、LSTM 不在题单里。
- 温度是 softmax 前的缩放。重复惩罚是按 token 下标的乘。reward 头和 value 头是 D-21 / D-22 的 GEMM。投机解码的草稿模型复用解码，接受或拒绝在 D-49。这些都不单开章。
- 版本表不新开章号。

## 落地顺序

1. D-01 已收口。int CSV 保留作归档。
2. 一次只开一章的大纲，确认后再写 `.cu`。依赖先于章号：D-03 先于 D-04～D-10；前向先于该章的反传档；D-12 先于 D-13；D-15 先于 D-23；D-21 先于 D-22 和 D-38；D-23、D-25 先于 D-42 和 D-45；D-27、D-28 先于 D-46 和 D-49；D-28 先于 D-54；D-47 先于 D-48；D-51 先于 D-52；D-01 先于 D-53 和 D-55。
3. D-22、D-23、D-45 的节都留在本章。上一档已经贴着库，后面的节不测。
4. 低精度三章（D-35～D-37）以 5090 能编译的指令为准，编译不了就停在该章并写明。
5. D-59 等前面有数再索引。
