# B-10 写作大纲：Module B Checklist

> 状态：✅ 文章+索引已落地（无新 `.cu`；确认路线：纯正文+图+索引）。
>
> **已交付**：正文 `article/02_memory_optim/B-10*.md`（含 §2 认出症状）；封面/分层图；`docs/results/B-10_checklist.md`。
> **不做**：新 micro-bench、入口打印脚本 G、新 NCU/SASS。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

> **路线**：Checklist-first——主产物是「症状 → 证据入口 → 处方 → 回链」总表；复用 B-01～B-09 已落地结论与 `docs/results/`，**不重开 micro-bench 全家桶**。
>
> **硬件门槛**：**不限**（汇总导航；各行处方继承对应章门槛，如 B-08 需 sm_90+）。
>
> **相对骨架的收紧**：
> 1. 主表按 **症状层**（Host↔Device / UM / GMEM / SMEM / Reg / L2 / 设备内 async / 布局）分行，禁止按「章号流水账」堆砌。
> 2. 每行必须挂 **证据入口**（binary/`--mode` 或 `docs/results/B-0N_*`）；无入口的条目降级为「仅回链正文」。
> 3. Host CE、SM `cp.async`、TMA bulk **三层分清**——单独成「分层速查」块，禁止混成「async 就快」。
> 4. 布局优先于引擎：sectors/request 或 useful GB/s 形状异常时，先 B-09/B-01，再 B-07/B-08。

**标题**：`B-10. Module B Checklist：从「症状」到「证据」到「处方」`

## 与前后章的边界

| 已有章节 | 已覆盖 | B-10 应深化 / 避免重复 |
|---|---|---|
| B-01～B-09 | 各章 TL;DR、决策表、误区、SOP、实测 | **汇总成可检索 triage 表**；不重讲机制长文；数字用「本机形状 + 链 results」 |
| B-06 / B-07 / B-08 | Host CE / SM async / TMA | 本章只做 **分层对照一行表**；不扫 AI、不写 descriptor/mbarrier 教程 |
| B-09 | AoS/SoA/transpose 决策与本机数 | 只收「布局症状→处方」条目；不重跑 `touch_fields` sweep |
| Module C | Warp / atomics / Graph / fusion | 本章只收束 **访存 / 内存体系**；钩子一句交给 C-01，不抢并发原语 |

## TL;DR 目标结论（写作时先写死；落地时挂本机索引）

1. **先定症状层，再选证据，勿一上来换 API**：带宽墙 / latency / Host↔Device / UM / 设备内 copy / 布局——层错了，处方必废。
2. **每条处方挂证据入口**：对应 `examples/02_memory_optim/0N_*.cu` 的 `--mode` +（若有）`docs/results/B-0N_*`；主证据永远是裸跑 CUDA event **median**，禁止 ncu 附着墙钟。
3. **三层 Async 分清**：B-06 Host Copy Engine ≠ B-07 SM 内 `cp.async`/`pipeline` ≠ B-08 TMA bulk；「写了 Async」不等于已 overlap。
4. **布局先于引擎**：sectors/request 偏高或 useful/总线形状异常 → 先 B-09/B-01；TMA/`cp.async` 救不了第一天选错的 stride（B-09 本机 touch=1：SoA/AoS ≈ **13.6×**）。
5. **Checklist 是导航不是新理论**：细节回链各章；判停看「证据是否复现该章形状」，不是再发明一套指标。

## 建议正文结构

1. **问题定义**：Module B 读完仍不知从哪开刀 → 统一 triage（插一张「分层地图」封面/原理图即可）。
2. **Triage 总流程（短）**：NCU Speed-of-Light / 症状分层 → 选证据 → 改一处 → 回归；对齐 Compute Triage 的 compute vs memory vs latency 分流。
3. **症状 → 证据 → 处方总表**（**主产物**；按症状层分行，见下节草案）。
4. **三层 Async 速查** + **布局 vs 引擎** 两张小表（防混层）。
5. **证据落点索引**：binary / 主命令 / CSV / plot / 可选 profile 脚本一览。
6. **误区合并 Top N**（从 B-05～B-09 去重；早期章用一句话）。
7. **SOP**：5～7 步可执行；判停条件。
8. **钩子 → Module C**（访存收束；下一站并发原语）。

## 写作路线

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Checklist-first**：主表 + 证据索引 + 回链；封面/分层图；无新 `.cu` | 与规划一致；成本可控；不与 B-09 抢实测 |
| 2 | 补「一键打印入口表」极短脚本 / target | 可选；仅当用户要 CLI 速查时再做 |
| 3 | 重跑 B-01～B-09 全家桶合成总榜 | **不推荐**；重复劳动且易与各章口径打架 |

## MVP 可行性评估

| 编号 | 产物 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | 正文症状→证据→处方总表（覆盖 B-01～B-09） | ✅ | **必做** |
| B | 三层 Async 对照表 + 布局优先于引擎表 | ✅ | **必做** |
| C | `docs/results/` 索引段（链 B-05～B-09；B-01～B-04 链示例） | ✅ | **必做** |
| D | 封面 + 1 张「分层 triage」原理图 | ✅ | **必做**（风格对齐 B-06～B-09） |
| E | 误区 Top N + SOP + Module C 钩子 | ✅ | **必做** |
| F | 新 `.cu` micro-bench / 新 CSV 主曲线 | ❌ | **默认不做** |
| G | `10_print_checklist` 一类入口打印脚本 | ❌ | **不做**（用户确认：纯正文+图+索引） |
| H | 新 NCU/SASS 旁证 | ❌ | **不做**（沿用各章已有旁证链接） |

## 症状 → 证据 → 处方总表（正文主表草案）

> 写正文时压成一张大表；下列为条目原料（数字 = 本机 RTX 5090 形状，完整表见 results）。

| 症状层 | 典型信号 | 证据入口 | 处方（一句话） | 回链 |
|---|---|---|---|---|
| Host↔Device 伪异步 / 伪 overlap | pageable + `MemcpyAsync`；单流；端到端远低于 pinned 上限 | `06_pinned_dma`：`pageable`/`pinned`/`serial`/`overlap`；[`B-06_*`](../results/B-06_pinned_dma_rtx5090.md) | pinned + 多非默认 stream；成功≈贴单向 pinned（本机 overlap 贴 ~52 GB/s） | B-06 |
| UM 冷启动 / 迁移风暴 | first≪steady 差、fault 时间线、多 GPU ping-pong | `05_unified_memory_pf`：`fault`/`prefetch`/`advise`；[`B-05_*`](../results/B-05_unified_memory.md) | 先 prefetch+同步；advise 当 hint；多 GPU 可写默认弃 UM | B-05 |
| GMEM 合并差 / 有效带宽低 | 跨步、sectors/request 偏高、向量化未用 | `01_global_mem_bandwidth`；NCU Memory / sectors | 合并 + 向量化；大 tile 再谈 TMA（钩子 B-08） | B-01 |
| SMEM bank 冲突 | 同 bank 多路；transpose/tile 掉带宽 | `02_shared_mem_bank_conflict`；pad/swizzle 对照 | padding / swizzle；transpose 用 `TILE+1`（B-09 挂钩） | B-02 |
| 寄存器 spill / Occupancy 误判 | ptxas spill↑；盲目追 Occupancy | `03_register_spill`；`-Xptxas=-v` | 先看 spill 是否在 hot path；Occupancy 非 KPI | B-03 |
| L2 热点可复用但未驻留 | streaming 却开 persisting；忘记 reset | `04_l2_residency`；time+DRAM+L2 三件套 | 有热点才 residency；扫 set-aside×hitRatio；必 reset | B-04 |
| 设备内 copy 延迟藏不住 | 低 AI 仍 sync；写了 async 立刻 wait | `07_cp_async_pipeline`：`sync`/`async1`/`pipe2`/`sweep`；[`B-07_*`](../results/B-07_cp_async_pipeline.md) | 低 AI 用 thread-local `pipe2`（本机极低 AI ~1.15–1.31×）；高 AI 停 | B-07 |
| 大 tile / 多维搬运指令税 | B-07 已够或不够；descriptor 开销 | `08_tma_intro`（**sm_90+**）：`sweep`/`pipe2`；[`B-08_*`](../results/B-08_tma.md) | 先证明 pipeline 段>1 再上 TMA；立刻 wait 常不赚（本机 ~0.86–1.05×） | B-08 |
| 布局 / 跨步写自杀 | 少字段扫 AoS；naive transpose | `09_layout_transform`：`sweep`/`transpose_*`；[`B-09_*`](../results/B-09_layout.md) | 少字段→SoA；transpose→SMEM tile（本机 tiled≈copy 91%） | B-09 |

## 证据最低要求

- **不新增**主结论曲线；以各章已落地 CSV/摘要为旁证链接。
- 正文凡写本机倍数 / GB/s：标明来源章 + `docs/results/`，并提醒带宽章 **L2 可能抬高绝对 GB/s，主看加速比形状**。
- 主证据口径统一提醒：CUDA event median；**禁止** ncu 附着程序自打印 ms。
- 默认 **不加** 新 profile shell。

## 本机要挂进 TL;DR / 总表的「形状」锚点（索引，非新测）

1. **B-06**：pinned 单向 ~52 GB/s；overlap 贴 pinned，不是「必须远快于 serial」。
2. **B-07**：低 AI `pipe2` 可 >1；`async1` 立刻 wait 可更慢；高 AI → ≤1。
3. **B-08**：TMA 收益来自引擎∥compute；立刻 wait 常 ≤1；与 Host CE / `cp.async` 不同层。
4. **B-09**：touch=1 SoA/AoS ≈ 13.6×；touch→8 收窄至 ~1.8×；tiled transpose ≈ copy 91%，naive ≈ 39%；NCU AoS 32 vs SoA 4 sec/req。

## 参考文献池（与正文参考文献节对齐）

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)（Pinned / Coalescing / Shared / Local） | 优化应按瓶颈分层；合并按 sector 事务；pinned 是 Host↔Device 高带宽前提 | §2 triage；总表 Host/GMEM 行 |
| A 官方 | [Nsight Compute — Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)（若镜像 404 则用同套件 [Profiling Guide — Speed of Light / Memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)） | Compute vs Memory SOL 分流；双低 → latency；Memory 重 → sectors/bank/DRAM；勿未分流就换 API | §2 总流程；TL;DR① |
| A 官方 | [CUDA Programming Guide — Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) + [Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) | 设备内 async 语义；与 Host `cudaMemcpyAsync` 分离 | 三层 Async 表；防混层 |
| A 工具 | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | Host/Device 时间线、UM migration、overlap 是否成立看时间线 | B-05/B-06 证据入口措辞 |
| B 工程 | GPU MODE Lecture 8 笔记：[CUDA Performance Checklist](https://christianjmills.com/posts/cuda-mode-notes/lecture-008/)（Christian Mills） | 合并 / bound 类型 / tiling 等 checklist 叙事；**作结构参考，数字以本仓库为准** | §1 问题；§8 SOP 条目对照 |
| B 工程 | 各章已引 NVIDIA blog（Harris transpose、Ampere data movement 等） | 不在此重复展开；正文写「见对应章 §10」 | §7 去重出口 |
| C 实证 | Svedin et al., *Benchmarking the Nvidia GPU Lineage…*（[arXiv:2106.04979](https://arxiv.org/abs/2106.04979) / PMBS’21） | 低 AI 时 async copy ~1.1–1.4×，高 AI 可 ≤1——**强度扫描判停**的文献形状 | 总表 B-07 行；与本机 B-07 互证 |
| C 实证 | Li et al., IISWC’23 / 后续 TACO：Async Memcpy vs UVM 分层 | Host 侧 UVM 与设备内 Async Memcpy 是不同 stage；可组合但勿混诊断 | 三层 Async + UM 行 |
| D 前沿 | Module C/D 大纲指向的 warp / Graph / CUTLASS | **不写进必做**；仅 §7「下一站」 | §7 / §9 钩子 |

**进 TL;DR / 总表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| SOL/症状分层再开刀；三层 Async；布局先于引擎 | TL;DR / 主表 |
| Svedin 低 AI 受益、高 AI 掉速 | 总表 B-07 行（文献形状）+ 链本机 B-07 |
| GPU MODE checklist 条目结构 | SOP 对照；不抄其绝对数 |
| Warp / Graph / Tensor Core 布局代数 | **仅** §7→Module C/D |

## 进正文时的硬约束（写稿检查）

- §7 ≤4 条出口；完整文献在 §10；已在 §10 的写「见 §10-x」。
- 不重开 B-01～B-09 机制章；表格单元格保持「一句话 + 链接」。
- 不擅自 commit/push。
