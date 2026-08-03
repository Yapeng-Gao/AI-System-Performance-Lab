# C-01 写作大纲：Warp Primitives（ballot / shfl / warp reduce·scan）

> 状态：🟡 大纲已确认；示例 + 正文骨架已落；**实测数字待本机 sweep**。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-C.md`](Module-C.md)
>
> **已交付**：`examples/03_compute_primitives/01_warp_primitives.cu`；正文 `article/03_compute_primitives/C-01*.md` + 封面/原理图；结果占位 `docs/results/C-01_warp_primitives.md`。

> **路线**：Microbench-first——**主 sweep 只扫 `nwarps`（block 内 warp 数）**，主曲线 = `shfl/smem` 加速比；`redux` / `ballot` 各做定点表，**不进主 sweep**；scan **首版砍掉**（正文一小节口述即可）。
>
> **硬件门槛**：**不限 sm_90+**。`__shfl_*_sync` / `__ballot_sync` 全架构（系列常用 RTX 5090）；`__reduce_*_sync` 需 **sm_80+**，作可选对照路径（不够则清晰跳过并打印原因）。
>
> **目录**：确认后建 `article/03_compute_primitives/` + `examples/03_compute_primitives/`——**禁止先建空 README**。

**标题（拟定）**：`C-01. Warp Primitives：寄存器级通信、mask 正确性与规约加速比`

**与前后章的边界**

| 已有 / 规划章节 | 已覆盖 | C-01 应深化 / 避免重复 |
|---|---|---|
| A-03 / A-04 | Warp 映射、SIMT、Divergence、Replay | **不重讲**「什么是 warp / 为何 divergence 贵」；只固化 **参与 mask + `*_sync`** |
| B-02 | SMEM bank / padding | naive SMEM reduce 作基线时 **一句**挂钩 bank；不做 bank 全家桶 |
| B-10 | 访存 Checklist | 承接「访存收束 → 并发原语」；不重开 coalescing/TMA |
| **C-02**（规划） | Cooperative Groups | 本章 **不用** `cg::reduce` 作主线；一句钩子「安全分组 API → 下一章」 |
| **C-03**（规划） | Atomics / contention | ballot+elect 只演示 **聚合原子的入口形态**；contention 曲线留给 C-03 |
| **C-04**（规划） | 同步分层 | `__syncwarp` 只在 mask/可见性误区出现；不写 block/grid sync 教程 |
| Module D | DeviceReduce / Softmax / LN | **不做** 多 block DeviceReduce、数值稳定 Softmax；只交付 **单 block 内** warp→SMEM→warp 模式原料 |

**TL;DR 目标结论（写作时先写死；有实测后改成带本机数字）**

1. **Warp 原语 = 寄存器总线**：`__shfl_*_sync` 在同 warp 内直接换寄存器，省掉「STS → bar → LDS」；收益来自 **更少指令 + 更少 SMEM 压力**，不是「魔法比寄存器还快」。
2. **必须用 `*_sync` + 显式 mask**（Volta+ Independent Thread Scheduling）：legacy `__shfl` / 隐式 warp-synchronous **不安全**；`mask` 应由程序逻辑决定，**不要**把 `__activemask()` 当「我想要的参与集合」直接塞进集体操作（NVIDIA warp-level primitives 博客）。
3. **该上 shuffle reduce**：热路径是 **warp 内** 规约/广播/扫描，且 SMEM 已成为 occupancy 或 bank 压力来源；文献与工程常见 **数倍级** 相对 naive SMEM 的加速（文献形状，非本机绝对数）。
4. **Ampere+ 整数硬件 reduce**：`__reduce_add/min/max_sync`（及 bit 逻辑）对 **32-bit 整型** 是单指令集体；**不支持 float**——浮点仍走 shfl 树或库（Ampere Tuning Guide / 论坛口径）。
5. **判停**：先看 CUDA event **median 时延 / 加速比**（shfl vs SMEM；有则 vs `__reduce_*_sync`）；正确性用 host 参考和；有 NCU 再看 `inst_executed` / 是否仍大量 SMEM 事务——**禁止**把 ncu 附着墙钟当结论。

**建议正文结构（8～9 节）**

1. **问题定义**：B-10 之后进入并发原语——对照表：访存已会，本章回答 **同 warp 如何正确、便宜地交换数据**。
2. **物理模型（短）**：SHFL / VOTE /（sm_80+）REDUCE 在 SIMT 中的位置；ASCII：SMEM 往返 vs 寄存器 shuffle；一句对照 A-04「锁步假设已死」。
3. **API 分层**：Vote（`ballot/any/all`）→ Shuffle（`shfl/up/down/xor`）→ Reduce（硬件）→ Match（扩展）；**mask 决策表**。
4. **规约处方**：naive SMEM tree → warp shfl tree →（可选）`__reduce_add_sync`；block 级「warp reduce → SMEM 放 warp 结果 → 首 warp 再 reduce」只作 **模式**，不扩成 DeviceReduce。
5. **MVP 实验矩阵** + 主命令 `--mode sweep`。
6. **实测**：表 + 曲线；口径 median；正确性检查必过。
7. **旁证（可选）**：NCU `inst_executed` / SMEM 吞吐；SASS 出现 `SHFL` / `REDUX`（命名以本机 dump 为准）。
8. **扩展阅读**：CUB `WarpReduce`、CG tile reduce、PTXASW shuffle 综合、ARC（→ C-03）；钩子 → C-02。
9. **误区 + SOP + 下一章钩子**。

**写作路线（2～3 条；默认推荐 #1）**

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Microbench-first**：同一 block reduce，对照 SMEM / shfl /（sm_80+）hardware reduce；`sweep` 扫 `N` 或 `nwarps` | 与 B-07～B-09 同构；结论可复现 |
| 2 | 教程重写「Kepler Shuffle 史」+ 大量 API 罗列 | 易与 NVIDIA 博客重复；**不推荐**作主线 |
| 3 | NCU-first：先盯指令数再讲故事 | 旁证强；写作机依赖大；作补强勿作唯一门禁 |

**MVP 可行性评估**

| 编号 | 配置 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | Block reduce：naive SMEM（可含 `__syncthreads`） | ✅ | **必做（基线）** |
| B | Warp `__shfl_down_sync` 树 + SMEM 仅存每 warp 部分和 | ✅ | **必做（主对照）** |
| C | `float` 主路径走 shfl（对照；硬件 reduce 不适用 float） | ✅ | **必做**（含在 B） |
| D | `int` + `__reduce_add_sync` 定点对照（`__CUDA_ARCH__ >= 800`） | ✅ sm_80+ | **定点必做 / CC\<80 跳过**；**不进 sweep** |
| E | `__ballot_sync` + elect leader 定点（聚合计数） | ✅ | **定点必做**；**不进 sweep** |
| F | Warp inclusive scan | — | **首版不做**（正文口述 + → Module D） |
| G | NCU / SASS：A vs B（vs D） | ⚪ 依赖工具 | **可选旁证** |
| — | `cub::DeviceReduce` / Softmax / CG `cg::reduce` 主线 / 多 GPU | ❌ | **不做** |

**最小可复现实验（`01_warp_primitives.cu`）**

| mode | 要回答的问题 | 进主结论？ |
|---|---|---|
| `smem` | naive SMEM block reduce 时延？ | 定点 |
| `shfl` | warp shfl + 少量 SMEM 时延？ | 定点 |
| `redux` | `__reduce_add_sync`（int，sm_80+）相对手写 shfl-int？ | 定点表 |
| `ballot` | ballot+elect 聚合计数正确且可测？ | 定点表 |
| `sweep` | 扫 `nwarps∈{1,2,4,8,16,32}`：`shfl/smem` 加速比形状？ | **主曲线** |
| `modes` | 定点全表（smem/shfl/redux/ballot） | 写结果用 |

**证据最低要求**

- 主证据：CUDA event **median** 时延与加速比；`sweep` CSV → `docs/results/C-01_*.csv` + 摘要 md。
- 正确性：与 host 参考和（或已知公式）比对；失败则非零退出。
- 启动打印 GPU 名 / `sm_XX`；`redux` 在 CC\<80 时跳过并打印。
- 默认 **不加** profile shell；用户明确要批量 NCU 再补。
- 防 DCE：规约结果写回 device 可见存储。

**本机要验证的「文献形状」假设（1～3）**

1. **通信密集的 warp 内交换**：shfl 路径相对 naive SMEM **显著更快**（文献应用侧常见约 **1.2×～2×+**；本机以 reduce microbench 校准，不追求对齐基因组论文绝对数）。
2. **Shuffle 延迟落在「寄存器」与「SMEM+同步」之间**（Wang/Xie/Cong IPDPS’17 microbench 形状）——正文用相对加速比讲述，不编造 cycle 表冒充本机。
3. **sm_80+ 上 int `__reduce_add_sync` ≤ 手写 shfl 树时延**（指令更少）；float 无此 intrinsic，shfl 仍是默认。

**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA C++ PG — Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions) | 同 warp 寄存器交换；官方 broadcast / scan / reduce 示例 | §2～3 API；MVP B/F |
| A 官方 | [CUDA C++ PG — Warp Vote Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions) | `ballot/any/all` 谓词集体 → 位掩码 | §3；MVP E |
| A 官方 | [CUDA C++ PG — Warp Reduce Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-reduce-functions) | sm_80+：`__reduce_*_sync`；32-bit 整型算术/逻辑 | TL;DR④；MVP D |
| A 官方 | [Volta Tuning Guide — Independent Thread Scheduling](https://docs.nvidia.com/cuda/volta-tuning-guide/) | 必须迁移到 `*_sync` + 显式 mask；勿假设锁步 | TL;DR②；误区 |
| A 官方 | [Ampere Tuning Guide — Warp level Reduction](https://docs.nvidia.com/cuda/ampere-tuning-guide/) | 硬件 warp reduce：add/min/max 与 bit and/or/xor | TL;DR④；§3 |
| B 工程 | NVIDIA, [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) | mask 五条纪律（勿滥用 `activemask`；逻辑算 mask；`syncwarp` 分离依赖） | TL;DR②；决策表 |
| B 工程 | Harris et al., [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) | shfl 树 + 每 warp 写 SMEM；CUB 可对齐或超过手写 | §4 处方；§7→CUB |
| B 工程 | CCCL/CUB [Warp-wide primitives](https://nvidia.github.io/cccl/cub/) | 生产优先 `cub::WarpReduce`；架构自适应 | §7 扩展；不做 Device 层 |
| C 实证 | Wang, Xie, Cong, **IPDPS’17**（[DOI:10.1109/IPDPS.2017.79](https://doi.org/10.1109/IPDPS.2017.79)；[PDF](https://vast.cs.ucla.edu/sites/default/files/publications/ipdps-submission.pdf)） | shuffle 延迟介于寄存器与 SMEM 之间；相对 SMEM 通信路径 **1.2× / 2.1×**（SW / PairHMM） | TL;DR①③；形状假设①② |
| C 实证 | Zhang et al., [arXiv:2004.05371](https://arxiv.org/abs/2004.05371) | warp shuffle / tile vs coalesced group 等同步原语的延迟–吞吐微观对比 | §2 旁证；同步成本直觉 |
| C 实证 | NVIDIA cuda-samples `reduction_kernel.cu`（公开） | sm_80+ 对 `int` 特化 `__reduce_add_sync`，否则 shfl 树 | MVP D 实现对齐 |
| D 前沿 | Takeshima & Honda, **CGO’23** PTXASW（[arXiv:2301.11389](https://arxiv.org/abs/2301.11389)） | 在 PTX 层自动综合 shuffle 替代 load；处理非完整 warp 边角 | §7 扩展；不进 MVP |
| D 前沿 | Durvasula et al., **ARC**, ASPLOS’25（[DOI:10.1145/3669940.3707238](https://doi.org/10.1145/3669940.3707238)） | warp 级自适应原子规约；软件路径仍建在 shfl/reduce 上 | §7 钩子 → **C-03** |
| D 前沿 | Luo et al., Hopper microbench（[arXiv:2501.12084](https://arxiv.org/abs/2501.12084)）；Blackwell microbench（[arXiv:2507.10789](https://arxiv.org/abs/2507.10789)） | 新架构指令/调度微观数据；**不**替代本章 reduce 主结论 | §7 可选；避免抢 B-08 叙事 |

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| `*_sync` + 逻辑 mask；勿滥用 `activemask` | TL;DR / 决策表 |
| shfl reduce vs SMEM：少指令、少 SMEM 压力 | TL;DR / MVP |
| `__reduce_*_sync` 仅 32-bit 整型；float 走 shfl | TL;DR④ / 决策表 |
| CUB WarpReduce / CG reduce | **仅** §扩展阅读（→ C-02 / 库） |
| ARC / 自动 shuffle 综合 | **仅** §扩展阅读（→ C-03 / 编译器） |
| 基因组论文 1.2×/2.1× | 只作「形状」；正文写「本机 sweep」 |

**交付进度**

- [x] `examples/03_compute_primitives/01_warp_primitives.cu`
- [x] `article/03_compute_primitives/C-01*.md` + 封面/原理图
- [ ] `docs/results/C-01_*` 填数 + 可选 `scripts/plot_c01_*.py`（等本机 sweep）
- [ ] 回填规划总表 ✅（有数后）
