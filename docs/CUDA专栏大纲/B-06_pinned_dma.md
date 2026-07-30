# B-06 写作大纲：Pinned Memory 与 DMA

> 状态：✅ 已落地。对照审稿用；以正文为准。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

> **已交付**：
> - 正文：`article/02_memory_optim/B-06*.md`（ASCII 原理 + RTX 5090 实测表/图、NSYS CLI 旁证）
> - 封面：`article/02_memory_optim/assets/B-06-pinned-dma-cover.png`
> - 实测图：`B-06-mode-gbs-bars.png` / `B-06-overlap-median-bars.png`（`scripts/plot_b06_pinned_dma.py`）
> - 示例：`examples/02_memory_optim/06_pinned_dma.cu` + `06_profile_pinned_dma.sh`
> - 结果：`docs/results/B-06_pinned_dma_rtx5090.md` + CSV
>
> 下方保留大纲便于对照审稿；以正文为准。

**标题**：`B-06. Pinned Memory 与 DMA：H2D/D2H 吞吐上限与 Overlap 条件`

**与前后章的边界**

| 已有章节 | 已覆盖 | B-06 应深化 / 避免重复 |
|---|---|---|
| A-07 | UVA、Zero-Copy 概念警示 | 用 micro-bench 量化 mapped vs memcpy；给出“只读一次 / 不缓存”判定 |
| A-08 | Stream 流水线、Pinned 是硬前置 | 不重讲三级流水线教程；改讲 **为何伪异步、CE 数量、双向饱和、chunk 粒度** |
| B-05 | UM fault/prefetch/advise → 显式管理 | 承接“显式路径怎么做到可控”；对照表可引用 B-05 §6 |
| B-07（规划） | 设备侧 async / pipeline | 本章只到 Host↔Device；GMEM→SMEM 留给下一章 |

**TL;DR 目标结论（写作时先写死 5 条）**

1. Pageable 上的 `cudaMemcpyAsync` **不是真异步**：驱动先 stage 到临时 pinned，再 DMA；吞吐低、且易与其他流串行化。
2. Pinned（`cudaMallocHost` / `cudaHostAlloc`）是 **DMA 直达 + 真 overlap** 的物理前提；`cudaHostRegister` 可用但通常更慢、更易踩 NUMA/对齐坑。
3. Overlap **三条件同时成立**：pinned + 非默认 stream + `asyncEngineCount≥1`；H2D∥D2H 还要求足够 CE（通常看 `asyncEngineCount≥2`）且主机内存带宽跟得上。
4. 吞吐上限常不是“理论 PCIe”，而是 **min(PCIe有效带宽, DRAM/NUMA带宽, 驱动开销/小包启动)**；小传输 latency-bound，大传输才逼近链路墙。
5. Zero-Copy（mapped pinned）省的是 memcpy launch，**不省 PCIe**；离散卡上仅适合“触达少、几乎不复用”的路径，否则直接打穿 PCIe。

**建议正文结构**

1. **问题定义**：B-05 之后，显式路径仍可能“看起来 Async 但不加速”——伪异步、伪 overlap、双向打不满。
2. **物理模型**：Pageable staging → Pinned DMA；Copy Engine 与 Compute Engine 分家；`asyncEngineCount` 含义。
3. **分配与 flags**：`cudaMallocHost` vs `cudaHostAlloc`（Default / Portable / Mapped / WriteCombined）vs `cudaHostRegister`；Pinned 过量会挤占 OS 可分页内存。
4. **Overlap 决策表**：单条件失败时的 NSYS 症状（串行 Copy、Host sync、同流依赖）。
5. **吞吐实验矩阵（MVP）**：扫 size；对照 pageable / pinned /（可选）WriteCombined；单向 H2D、单向 D2H、双向并行；可选 NUMA local vs remote。
6. **Zero-Copy 分支**：mapped kernel 直读 vs 显式 memcpy；与 A-07 警示对齐，用数据判停。
7. **工程边界（2024–2026）**：小包合并；`cudaMemcpyBatchAsync`（CUDA 12.8+）摊销 launch 开销（扩展阅读）；Grace NVLink-C2C / HMM 与“传统 PCIe+pinned”对照（注明硬件门槛，不作本章必跑）。
8. **误区清单 + SOP + 下一章钩子**（→ B-07 设备内 async）。

**最小可复现实验（`06_pinned_dma.cu`）**

| 编号 | 配置 | 要回答的问题 |
|---|---|---|
| A | pageable + `cudaMemcpyAsync` | 是否退化为 sync / staging？吞吐多少？ |
| B | pinned + Async H2D | 单向 DMA 吞吐是否明显上升？ |
| C | serial（1 stream 切块 H2D→Kernel） | overlap 的公平串行基线 |
| D | overlap（多 stream 切块） | 相对 serial 端到端是否下降？NSYS 是否跨 chunk 重叠？ |
| E | pinned + 双向 H2D∥D2H | 合计是否接近 2× 单向，还是被主机内存/CE 卡住？ |
| F | mapped zero-copy kernel | 有效 host-read 带宽（勿直接对比 memcpy GB/s） |

**证据最低要求**：CUDA event / 墙钟得到 GB/s（first/median）；优先用 `serial` vs `overlap` vs `pinned` 对照判定（copy-bound 时 overlap≈pinned 即成功）；NSYS CLI/`stats` 可作旁证，有 GUI 再看时间线。可选：记录 `asyncEngineCount`、PCIe 代数、NUMA 绑定。

**参考文献池（与正文 §9 对齐）**

- 官方：CUDA Best Practices（Pinned / Async Overlap）、Programming Guide（Async Execution）、Runtime API（[API sync behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)、`cudaHostAlloc` flags）、Nsight Systems User Guide
- 经典博客：[How to Optimize Data Transfers](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)、[How to Overlap Data Transfers](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- 新 API：CUDA 12.8+ [`cudaMemcpyBatchAsync`](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)
- 近年研究/工程：Grace Hopper system memory（[ICPP’24 / arXiv:2407.07850](https://arxiv.org/abs/2407.07850)）、MultiPath H2D（[arXiv:2512.16056](https://arxiv.org/abs/2512.16056)）、PCIe Gen5/NUMA 实测（nvbandwidth 类工具链）
