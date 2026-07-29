# B-07 写作大纲：Async Copy / Pipeline

> 状态：✅ 已落地。对照审稿用；以正文为准。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

> **已交付**：
> - 正文：`article/02_memory_optim/B-07*.md`（ASCII 原理 + RTX 5090 完整 intensity sweep + NCU WarpStateStats + SASS 旁证）
> - 示例：`examples/02_memory_optim/07_cp_async_pipeline.cu` + `07_profile_cp_async_pipeline.sh` + `07_dump_sass.sh`
> - 结果：`docs/results/B-07_cp_async_pipeline.md` + `B-07_sweep.csv` / `B-07_modes.csv`
> - 实测图：`B-07-speedup-vs-fma.png` / `B-07-mode-speedup-bars.png`（`scripts/plot_b07_cp_async.py`）
>
> 路线：**Ampere-first 多级流水线（优先 thread-local / unified）+ arithmetic intensity 扫描**；TMA 整章交给 B-08；warp specialization / CUTLASS Pipeline 仅扩展阅读。

**标题**：`B-07. Async Copy / Pipeline：GMEM→SMEM 何时能藏延迟，何时反而变慢`

**与前后章的边界**

| 已有章节 | 已覆盖 | B-07 应深化 / 避免重复 |
|---|---|---|
| A-08 | Host 侧 Stream / H2D→Compute→D2H 三级流水线 | **不重讲** CE/Stream；一句话对照「Host CE overlap ≠ SM 内 async copy」 |
| B-01 | `cp.async`→TMA 概念演进、合并访问 | 把 Ampere 路径做成 **可复现 micro-bench + 决策表**；TMA 细节不展开 |
| B-02 | SMEM bank / padding / swizzle | 强调数据 **落地之后** bank/swizzle 仍适用；本章不重做 bank conflict 教程 |
| B-06 | Host↔Device pinned / DMA / overlap | 本章只谈 **GMEM→SMEM**；不重复 pageable/pinned |
| B-08（规划） | Hopper TMA / `cp.async.bulk` | 本章只给钩子：大批量多维搬运、指令带宽墙 → 下一章 |

**TL;DR 目标结论（写作时先写死 5 条）**

1. `cp.async` / `cuda::memcpy_async` 是 **SM 内** DMA：GMEM→SMEM，旁路寄存器；与 B-06 的 Host Copy Engine **不是一层**。
2. 收益来自 **outstanding stages × 足够 compute overlap**，不是「async 指令本身比 sync load 更快」。
3. **低 arithmetic intensity / latency-bound** 才值得上；已 compute-bound 或 occupancy 已能藏 LDG 时，pipeline 同步与多 stage SMEM 常净亏损（见 Svedin 等实证）。
4. Stage 加深换延迟，但挤占 SMEM → 掉 occupancy；shared/partitioned `cuda::pipeline` 有 per-stage barrier 开销——能 **thread-local** 就不要 block shared。
5. 对齐/尺寸不满足时可能回退或走非预期路径；Hopper+ 大批量多维搬运交给 **B-08 TMA**，本章只给钩子。

**建议正文结构**

1. **问题定义**：B-06 后数据已在 HBM，kernel 仍「等 LDG」——用一行对照表区分 Host CE overlap vs 设备内 async copy。
2. **物理模型**：`LDG → RF → STS` vs `LDGSTS` / `cp.async`；MIO / async copy 路径；为何不占长 scoreboard、可旁路 L1。
3. **API 分层与同步**：sync load → 低层 `__pipeline_memcpy_async` / PTX → `cuda::memcpy_async` + `cuda::barrier` / `cuda::pipeline`；unified vs partitioned；commit 需 warp 收敛（官方 Warp Entanglement 警示）。
4. **决策表**：何时上 pipeline、几 stage、何时回退 sync（对照文献 + 本机 intensity 曲线）。
5. **MVP 实验矩阵**：见下表；主证据用 CUDA event；NCU 作旁证。
6. **工程边界**：SMEM 预算 vs occupancy；`mio_throttle`；与 B-02「落地后仍要管 bank/swizzle」。
7. **扩展阅读（2021–2026）**：CudaDMA 专用 copy warp → Ampere 硬件 async；CUTLASS multistage vs warp-specialized（不写生产级 GEMM）；Blackwell 仍保留 cp.async 路径 → 说明本章在消费级新卡仍有价值；钩子 → B-08 TMA。
8. **误区清单 + SOP + 下一章钩子**（→ B-08 Hopper TMA）。

**最小可复现实验（`07_cp_async_pipeline.cu`）**

| 编号 | 配置 | 要回答的问题 |
|---|---|---|
| A | sync：`gmem→reg→smem`（或 sync load 后直接消费） | 公平基线时延/吞吐？ |
| B | `memcpy_async` + 单缓冲 wait（无 overlap） | 仅换指令、不做流水线时有无收益/开销？ |
| C | 2-stage `cuda::pipeline` | 相对 A 是否加速？ |
| D | 4-stage `cuda::pipeline` | 更深 stage 是否继续赚，还是被 SMEM/occupancy 反噬？ |
| E | 扫 compute intensity（FMA 次数或等价 AI） | 画出「加速比 vs AI」：低 AI 受益、高 AI 持平/变慢？ |
| F（可选） | thread-local vs block shared pipeline | shared pipeline 的 barrier 开销是否可测？ |

**证据最低要求**：CUDA event 得到 median 时延或有效带宽；**intensity 扫表**写入 `docs/results/`（主结论载体）。旁证：NCU 至少一组 A vs C/D（关注 `long_scoreboard` 下降、`mio_throttle`、或 sm vs dram 吞吐）。可选：SASS 确认出现 `LDGSTS` / `CP.ASYNC`。完整对照见已落地的 `examples/02_memory_optim/07_cp_async_pipeline.cu`。

**参考文献池（与正文参考文献节对齐）**

- 官方：CUDA Programming Guide — [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)、[Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)（thread-local 优先；Warp Entanglement）；[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)（GMEM→SMEM 硬件加速）
- 工程博客：[Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)（对照 CudaDMA）
- CCCL / libcu++：[`cuda::memcpy_async`](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html)（对齐门槛与架构回退；Hopper TMA 细节不在本章展开）
- 高质量实证：Svedin et al., *Benchmarking the Nvidia GPU Lineage… with Asynchronous Memory Transfers*（PMBS@SC’21 / [arXiv:2106.04979](https://arxiv.org/abs/2106.04979)）——低 AI 约 1.07–1.35×，高 AI 可至 ~0.95×；Li et al., *Performance Implications of Async Memcpy and UVM*（IISWC’23，[PDF](https://lca.ece.utexas.edu/pubs/Li_IISWC_2023.pdf)）——GMEM→SMEM 非瓶颈时 async 无收益
- 扩展阅读：Colfax / SIGARCH [Efficient GEMM Kernel Designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)；[CUTLASS Pipeline](https://docs.nvidia.com/cutlass/media/docs/cpp/pipeline.html)；MLC.ai [Pipelining GEMM with TMA](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_async/index.html)（为 B-08 铺垫）
