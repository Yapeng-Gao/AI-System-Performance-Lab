# B-08 写作大纲：Hopper TMA

> 状态：✅ 已落地。对照审稿用；以正文为准。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

> **已交付**：
> - 正文：`article/02_memory_optim/B-08*.md` + 封面/实测图
> - 示例：`examples/02_memory_optim/08_tma_intro.cu`（主命令 `--mode sweep`）
> - 结果：`docs/results/B-08_tma.md` + `B-08_sweep.csv` / `B-08_modes.csv`
>
> **路线**：Microbench-first；不写生产级 GEMM / WGMMA / warp-spec。
>
> **硬件门槛**：`sm_90+`（实测 RTX 5090 / sm_120）。

**标题**：`B-08. Hopper TMA：从单线程 Bulk Copy 到吞吐墙（何时该上、何时别上）`

**与前后章的边界**

| 已有章节 | 已覆盖 | B-08 应深化 / 避免重复 |
|---|---|---|
| B-01 | TMA 叙事、multicast 概念钩子 | 做成 **可复现 micro-bench + 决策表**；multicast/DSM 一句话挂钩，不展开 |
| B-02 | bank / swizzle | TMA swizzle mode 只强调「落地后仍要对齐 B-02」；不重做 bank 教程 |
| B-07 | Ampere `cp.async` / pipeline / intensity 曲线 | 回答 **指令带宽墙 / 多维 tile / 单线程 issue**；不重跑 2/4-stage Ampere 全家桶（只留最小对照） |
| Module D / FA3 | 生产 GEMM / Attention | warp-spec + WGMMA **扩展阅读 only** |

**TL;DR 目标结论（写作时先写死 5 条）**

1. TMA 是 **专用异步拷贝引擎**（GMEM↔SMEM；亦可 cluster DSM）；与 B-07 的 SM 内 `cp.async`、B-06 的 Host CE **不是一层**。
2. 收益核心是 **单线程 issue + descriptor 卸地址/边界/predication** → 省寄存器与指令带宽；不是「单次延迟一定更短」（Luo et al. 完整路径可 **+~170 cycle**，需大 tile / compute overlap 摊销）。
3. **该上**：大 tile、多维/跨步、地址计算或 copy 循环本身吃指令带宽、需要与 compute 深度重叠。
4. **别上 / 先别上**：小块、对齐/尺寸不满足、descriptor 重建过频（PyTorch Triton TMA 反例）、已 compute-bound 且 B-07 pipeline 已够用。
5. 同步模型：G2S 用 `mbarrier` + `expect_tx`；S2G 常走 **bulk async-group**——混用是高频坑。

**建议正文结构**

1. **问题定义**：B-07 之后仍可能卡在「每线程算地址 + 发一堆 copy」——对照表：`cp.async` vs TMA。
2. **物理模型**：tensor map / 坐标 vs 指针；单线程 elect；旁路 L1、受 L2 影响的实证含义（引 arXiv:2501.12084）。
3. **API 分层**：① 1D `cp.async.bulk` ② `cuTensorMapEncode*` + `cp.async.bulk.tensor` ③ CCCL / `cuda::memcpy_async`（注明 Hopper+ 可能已走 TMA 回退路径，对照实验要用显式 bulk）④ CuTe/CUTLASS（扩展）。
4. **同步与完成**：`mbarrier` tx 字节记账 vs `bulk_group`；init → expect_tx → issue → arrive/wait 顺序。
5. **决策表**：何时 TMA、何时回退 B-07、何时只换引擎不做 overlap。
6. **MVP 实验矩阵**：见下；主证据 CUDA event；SASS/NCU 有工具再做。
7. **工程边界**：对齐与粒度；swizzle↔WGMMA 布局钩子；cluster multicast 一节挂钩；Blackwell 仍保留 cp.async。
8. **扩展阅读**：Colfax TMA、FA3、ACTA、Cypress。
9. **误区 + SOP + 钩子 → B-09 布局**。

**MVP 可行性评估（路线 A；能做就做）**

| 编号 | 配置 | 可行性（相对 RTX 5090 / CUDA 12+） | 本章裁决 |
|---|---|---|---|
| A | sync 或 B-07 风格 `cp.async` 最小基线（公平对照，避免笼统 `memcpy_async` 在 sm_90+ 上静默走 TMA） | ✅ 低风险，模式已有 | **必做** |
| B | 1D `cp.async.bulk` + `mbarrier`（无/弱 compute overlap） | ✅ 官方路径清晰，无需 tensor map | **必做（主路径）** |
| C | 2D `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor` G2S | ✅ 样板成熟，样板代码量中等 | **必做**（对齐 B-07「多维」钩子） |
| D | B/C + FMA intensity 扫（同 B-07 叙事） | ✅ 依赖 B/C | **必做**（主结论载体） |
| E | tile 尺寸扫（小→大，对齐 ACTA「配置敏感」叙事） | ✅ 参数化即可 | **可选**：D 稳定后再加 |
| F | SASS / NCU 旁证（TMA issue、对比指令数） | ⚪ 依赖本机 `cuobjdump`/`ncu`（B-07 已有脚本范式） | **可选**：有工具就做 |
| — | cluster multicast / DSM / WGMMA 耦合 / 完整 warp-spec | ❌ 复杂度与 Module D 重叠 | **本章不做**（扩展阅读一句带过） |

**最小可复现实验（`08_tma_intro.cu`）**

| 编号 | 配置 | 要回答的问题 |
|---|---|---|
| A | sync / 显式 Ampere 风格 copy 基线 | 公平对照时延/吞吐？ |
| B | 1D bulk + mbarrier，立刻 wait | 换引擎本身开销/吞吐？ |
| C | 2D tensor-map tile G2S + mbarrier | 多维搬运能否稳住有效带宽？ |
| D | B 或 C + intensity 扫 | 加速比 vs AI：能否摊销 TMA+barrier 固定开销？ |
| E（可选） | 扫 tile 边长 / bytes | 过小被开销吃掉、过大撞 SMEM？ |
| F（可选） | SASS/NCU：A vs B/C | 是否出现 TMA 类指令、issue 是否下降？ |

**证据最低要求**：CUDA event median 时延或有效带宽；**D 的 intensity 扫表**写入 `docs/results/B-08_tma_*.md`（主结论）。旁证：有 NCU/SASS 则至少一组 A vs B/C。启动打印 `sm_XX`；非 sm_90+ 直接失败并提示。

**参考文献池（与正文参考文献节对齐）**

- 官方：CUDA Programming Guide — [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)（1D bulk vs tensor map、`mbarrier`/`expect_tx`、store/`bulk_group`）；[Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)；Hopper Tuning Guide；[CCCL TMA](https://nvidia.github.io/cccl/unstable/cccl/tma.html)
- 工程教程：Colfax [Mastering Hopper TMA](https://research.colfax-intl.com/tutorial-hopper-tma/)；MLC.ai [Pipelining GEMM with TMA](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_async/index.html)；PyTorch [Hopper TMA for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/)（descriptor 开销反例）
- 高质量实证：Luo et al., IPDPS’24 / [arXiv:2402.13499](https://arxiv.org/abs/2402.13499)；Luo et al., [arXiv:2501.12084](https://arxiv.org/abs/2501.12084)（TMA 延迟/吞吐专章，~+170 cycle）
- 扩展阅读：Shah et al., FlashAttention-3（NeurIPS’24 / [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)）；ACTA（GPGPU’25 / [DOI:10.1145/3725798.3725802](https://doi.org/10.1145/3725798.3725802)）；Yadav et al., Cypress（PLDI’25 / [arXiv:2504.07004](https://arxiv.org/abs/2504.07004)）；Colfax GEMM pipelining / CUTLASS Pipeline
