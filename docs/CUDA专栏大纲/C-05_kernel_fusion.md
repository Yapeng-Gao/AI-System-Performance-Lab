# C-05 写作大纲：Kernel fusion 代价边界（寄存器 / occupancy / 可维护性）

> 状态：✅ 文章+示例+RTX 5090 实测已落地。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-C.md`](Module-C.md) · 上章：[`C-04_sync_layers.md`](C-04_sync_layers.md)
>
> **已交付**：`05_kernel_fusion.cu`；正文；`docs/results/C-05_*`；plot；封面/原理图。  
> **本机要点**：`fused/naive` @k=2…8 → **3.87×→9.81×**；fused 墙钟平坦；`fat` occ **6→1**，fat/fused **8.01×**（更慢）。  
> **裁决**：`fat` = `modes` 定点必做；主曲线只扫 `k` 上 naive vs fused。
>
> **路线**：Microbench-first——主对照 **多 kernel 逐级 elementwise（中间结果走 global）** vs **单核垂直融合（中间结果留寄存器）**；**`fat` 定点必做（不进主 sweep）** 造 occupancy 悬崖。
>
> **硬件门槛**：**不限 sm_90+**。elementwise + occupancy API 全架构常用。
>
> **目录**：已有 `article/03_*` / `examples/03_*`；本章只增 `05_kernel_fusion.cu` 等，**勿再建空壳**。

**标题（拟定）**：`C-05. Kernel Fusion 代价边界：少写回 vs 寄存器压力与 occupancy`

**与前后章的边界**

| 已有 / 规划章节 | 已覆盖 | C-05 应深化 / 避免重复 |
|---|---|---|
| **C-04** | 同步分层；「少 sync ≠ 一定 fuse」钩子 | **不重测** warp/block/grid 空同步；承接句：少 launch/同步点只是 fusion 的副作用之一 |
| A-08 / **C-06**（下） | Stream overlap；Graph / launch 墙 | **不深挖** host launch 分解与 Graph capture；fusion 省 launch 只作次要收益一句 → C-06 |
| Module B / B-10 | 带宽 / 布局 | 融合收益用 **少 global 往返** 挂钩；不重开 coalescing/TMA |
| Module D | Softmax / LN / GEMM epilogue | **不做** 生产算子融合 / FlashAttention 复现；elementwise 链即可 |
| Module E | torch.compile / Inductor | **不做** 框架自动融合正文；§7 钩子「先验证编译器是否已 fuse」 |
| C-07 | Persistent / warp specialization | 不做 persistent mega-kernel |

**TL;DR 目标结论（写作时先写死；有实测后改成带本机数字）**

1. **该 fuse 的典型信号**：短 elementwise / pointwise 链、中间结果本可不必落 global、带宽墙明显——垂直融合把中间量留在寄存器，少一次（或多次）GMEM 往返（NVIDIA fusion blog / Filippone–Filipovič 形状）。
2. **fuse 也可能更慢**：融合抬高每线程寄存器（或 SMEM）→ occupancy 掉 → 藏不住延迟；文献与工程共识是 **occupancy/spill 悬崖存在**——本机用「胖融合」对照校准。
3. **先量两条轴再决策**：墙钟（median）+ 资源（occupancy / 可选 regs·spill）；只看「少了几个 kernel」不够。
4. **可维护性也是代价**：手写长融合难测、难复用；短链优先；长链/框架侧留给编译器（→ E），本章只给手工边界感。
5. **判停**：`fused/naive` 随链长或压力参数的形状；胖融合相对瘦融合是否掉速；**禁止**把 ncu 附着墙钟当结论（NCU 可选看 DRAM traffic / reg spill）。

**建议正文结构（8～9 节）**

1. **问题**：C-04 说少同步点不等于该 fuse——本章回答 fusion 的收益与翻车边界。
2. **物理模型（短）**：垂直融合 vs 多 kernel；ASCII：中间量 GMEM 往返 vs 寄存器直通；一句对照水平融合（独立核拼一块，本章不做主线）。
3. **代价账**：省流量 / 省 launch（次要） vs 寄存器压力 / occupancy / spill / 可维护性。
4. **决策表**：短 elementwise 链 / 已带宽打满 / 寄存器已紧 / 该退回多核或拆 fuse。
5. **MVP 实验** + 主命令 `--mode sweep`。
6. **实测**：表 + 曲线；口径 median；附 occupancy。
7. **旁证（可选）**：NCU DRAM sectors / `ptxas -v` regs；SASS 非必须。
8. **扩展阅读**：torch.compile / Inductor；水平融合 HFUSE；GEMM epilogue → D；钩子 → C-06。
9. **误区 + SOP + 下一章钩子**。

**写作路线**

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Microbench-first**：同一 elementwise 链，naive 多核 vs fused；`sweep` 扫链长或压力旋钮 | 与 C-01～C-04 同构；直接回答边界 |
| 2 | 复述 NVIDIA fusion blog + 框架 API | 易与官方重复；**不推荐**作主线 |
| 3 | 复现 Transformer / FlashAttention 融合 | 重、抢 Module D；**仅扩展阅读** |

**MVP 可行性评估**

| 编号 | 配置 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | Naive：`k` 个独立 elementwise kernel，中间结果写 global（如 `y=a*x+b` → `relu` → `y=c*y+d` …） | ✅ | **必做（基线）** |
| B | Fused：同语义单核，中间量留寄存器 | ✅ | **必做（主对照）** |
| C | Fat-fused：同融合骨架但人为抬高 live 寄存器（数组累加 / 多临时 float），逼 occupancy 下降 | ✅ | **必做（翻车对照）** |
| D | `sweep`：扫链长 `k∈{2,3,4,6,8}`（或等价压力档）上 A/B（+C 定点）时延与加速比 | ✅ | **必做（主曲线）** |
| E | 每配置打印 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`（及推导的 theoretical occupancy 直觉） | ✅ | **必做（旁证轴，进表）** |
| F | 定点 `modes`：固定大 `n`，A/B/C 全表 + 正确性（host 参考） | ✅ | **必做** |
| G | NCU：A vs B 的 DRAM / L2；C 的 spill（若有） | ⚪ | **可选旁证** |
| — | Graph / launch 微秒拆解；torch.compile 主线；GEMM+epilogue 生产核；水平融合全家桶 | ❌ | **不做**（→ C-06 / E / D） |

**最小可复现实验（`05_kernel_fusion.cu`）**

| mode | 要回答的问题 | 进主结论？ |
|---|---|---|
| `naive` | 多核链时延？ | 基线 |
| `fused` | 瘦融合时延？ | 对照 |
| `fat` | 胖融合（高寄存压力）时延？ | 翻车对照 |
| `sweep` | 扫链长 `k`：`fused/naive`、（定点）`fat/fused`？ | **主曲线** |
| `modes` | 定点全表 + occupancy 列 | 写结果用 |

实现约定：

- 工作负载：**点式链**（float），索引一一对应，便于正确性与「垂直融合」叙事；默认大 `n`（如 `1<<24`）偏带宽。
- 主证据：CUDA event **median**（整条链墙钟：naive = 各核之和包在同一对 event 内，或逐核累加后报告——**大纲写死一种并在正文标明**；推荐 **一次 event 包住整条 naive 序列**，与 fused 单核公平比总墙钟）。
- 防 DCE：最终写回 `out[]`；verify 与 host 逐步参考比对。
- 启动打印 GPU / `sm_XX`；每个 kernel 打印 occupancy 提示。
- 默认 **不加** profile shell；用户明确要再补。
- Fat 路径：用编译器不易 DCE 的 live 值集合（如依赖 `threadIdx`/`i` 的多临时累加进最终结果），避免「看起来胖、实际被优化掉」。

**证据最低要求**

- 主证据：median 时延与 `fused/naive`；CSV → `docs/results/C-05_*.csv` + 摘要 md。
- 资源列：occupancy（blocks/SM 或 %）随模式变化可见。
- 正文必须写清：本机加速比是 **elementwise 微基准形状**，不直接对齐 blog 上 abs+reduce 的 3×，也不替代框架自动融合数字。

**本机要验证的「文献形状」假设（1～3）**

1. **短链 + 大 n**：`fused` 相对 `naive` **明显更快**（少 GMEM 往返；预期至少稳定拉开）。
2. **链变长或 fat**：加速比收窄或反转——occupancy 下降应与墙钟恶化同向（或 spill 可见）。
3. **occupancy  alone 不裁决**：若仍带宽墙且无 spill，中等 occupancy 下降可能仍可接受——以墙钟为准，occupancy 作解释轴。

**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA Best Practices — Memory / Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | occupancy 藏延迟；寄存器限制活跃 warp | §2～3；决策表 |
| A 官方 | Occupancy API（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`） | 运行时估 blocks/SM | MVP E |
| B 工程 | NVIDIA, [Kernel Fusion in NVIDIA CUDA](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)（2026） | 中间量留寄存器少 GMEM；与 Graph「launch 融合」分层；4090 上 abs+reduce 示例约 **3×** | TL;DR①；边界→C-06 |
| B 工程 | NVIDIA, [Shared Memory Register Spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)（CUDA 13） | spill 可改道 SMEM；launch bounds 纪律 | §7 / 误区；不做主线 |
| B 工程 | 寄存器阶跃与 spill 工程笔记（如 Elashri 文） | regs/thread 跨阈 → occupancy 阶跃下降 | §2；假设② |
| C 实证 | Filipovič et al., [arXiv:1305.1183](https://arxiv.org/abs/1305.1183)（Kernel fusion on BLAS） | 融合省流量也可能因 on-chip 压力降低 occupancy 而变慢 | TL;DR②；形状假设 |
| C 实证 | Wahib & Maruyama 等水平融合 / HFUSE（CGO’22 等） | 水平融合另叙事；issue slot；regs bound 有时赢有时输 | §7；本章不做 HF 主线 |
| C 实证 | C-04 本机：grid vs 多 kernel ≈1 | 「少同步/少 launch」不自动更快——与本章「少核 ≠ 必赢」同族 | 承接 |
| D 前沿 | Fused Kernel Library [arXiv:2508.07071](https://arxiv.org/abs/2508.07071)；MCFuser / FlashFuser / Blockbuster 等 | 自动 VF/HF、编译器侧搜索 | §7；不进 MVP |
| D 前沿 | torch.compile / Inductor 融合 | 框架已常自动 fuse elementwise | §7 → Module E |

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 短 elementwise 链 → 垂直融合少 GMEM | TL;DR / MVP |
| 寄存器/occupancy 悬崖 → 可能反融合 | TL;DR② / 决策表 / fat |
| 可维护性 / 先查编译器是否已 fuse | TL;DR④ / 误区 |
| Graph / launch 拆解 | **仅** → C-06 |
| GEMM epilogue / FlashAttn / 自动融合框架 | **仅** §扩展 → D / E |

**交付进度**

- [x] 用户确认本大纲（含 `fat` 定点必做、不进主 sweep）
- [x] `examples/03_compute_primitives/05_kernel_fusion.cu`
- [x] 本机实测 + `docs/results/C-05_*` + plot
- [x] 正文 + 封面/原理图（TL;DR 以 fused/naive 与 fat 悬崖）
- [x] 回填规划总表 ✅
