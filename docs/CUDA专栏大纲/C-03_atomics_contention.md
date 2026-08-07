# C-03 写作大纲：Atomics 与 contention（global / shared / warp-aggregated）

> 状态：⏳ 大纲待确认（确认后再写 `.cu` / 正文）。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-C.md`](Module-C.md) · 上章：[`C-02_cooperative_groups.md`](C-02_cooperative_groups.md)

> **路线**：Microbench-first——主曲线扫 **争用强度**（同地址命中比例 / bin 数）；对照 **naive global atomic** vs **shared staging** vs **warp-aggregated**（ballot/coalesced 入口承接 C-01/C-02）。
>
> **硬件门槛**：**不限 sm_90+**。`atomicAdd`（int/ull）全架构常用路径；float atomic 若测则标清架构支持，首版主路径用 **32-bit int 计数**。
>
> **目录**：已有 `article/03_*` / `examples/03_*`；本章只增 `03_atomics_contention.cu` 等，**勿再建空壳**。

**标题（拟定）**：`C-03. Atomics 与 Contention：争用曲线、分层 staging 与 warp 聚合`

**与前后章的边界**

| 已有 / 规划章节 | 已覆盖 | C-03 应深化 / 避免重复 |
|---|---|---|
| **C-01** | ballot + elect 聚合计数入口 | **不重讲** mask/`*_sync`；把 ballot/elect 当聚合原料，主线是 **争用曲线** |
| **C-02** | `coalesced_threads` 聚合形态 + 正确性 | **不重测** CG 抽象税/tile 悬崖；可用 coalesced 作聚合实现之一 |
| A-04 | Divergence | 不重讲机制课；聚合路径注意活跃集 |
| B-02 / B-10 | SMEM bank / 访存 Checklist | staging 用 SMEM 时一句挂钩 bank；不做 bank 全家桶 |
| **C-04**（下） | 同步分层 | 不做 `__syncthreads` 教程；staging 后 barrier 只当处方一步 |
| Module D | DeviceReduce / Softmax / histogram 算子 | **不做** 生产直方图库 / 多 block 全局规约正文 |
| 渲染 / GS 训练 | ARC 应用侧 | 只作形状/动机引用；**不复现** 可微渲染 workload |

**TL;DR 目标结论（写作时先写死；有实测后改成带本机数字）**

1. **争用决定原子墙钟**：同地址（或极少 bin）上堆满 `atomicAdd` 时，墙钟被原子单元串行化吞掉；处方优先级是 **少原子次数**，不是先换「更快的 atomic 指令」。
2. **该上 warp 聚合**：warp 内多线程更新同一计数器/同一 bin——先 ballot/coalesced（或 shfl reduce）聚合成一份，**每 warp 一次** `atomicAdd`（NVIDIA Pro Tip；本机以 sweep 校准加速比）。
3. **Shared staging 有条件**：block 内先撞 SMEM atomic、再写回 global，在部分架构/形态上有用；现代 GPU 上 **未必** 赢过「warp 聚合 + global」（论坛与 Kepler 博客已提示）——以本机对照为准。
4. **编译器可能已聚合**：同 warp 同地址、相同增量的简单 `atomicAdd` 有时会被 NVCC 优化；microbench 要防「手写聚合 ≈ 朴素」的假阴性——正文写清对照条件与 verify。
5. **判停**：裸跑 median 看加速比随争用强度的形状；高争用下聚合应明显拉开；NCU 可选看 atomic throughput / L2 atomic stall——**禁止**把 ncu 附着墙钟当结论。

**建议正文结构（8～9 节）**

1. **问题**：C-02 给了聚合入口——本章回答「撞同一地址有多贵、何时聚合/何时 staging」。
2. **物理模型（短）**：原子在哪执行（L2 atomic unit 直觉）；争用 = 多请求串行；ASCII：每线程一 atomic vs 每 warp 一 atomic。
3. **API / 模式分层**：global atomic → shared staging → warp-aggregated（ballot / coalesced / shfl+atomic）。
4. **决策表**：高争用计数 / 过滤写回 / 多 bin 直方图浅谈 / 何时别聚合。
5. **MVP 实验** + 主命令 `--mode sweep`。
6. **实测**：表 + 曲线；口径 median。
7. **旁证（可选）**：NCU atomic / L2 相关 metric。
8. **扩展阅读**：ARC（自适应聚合）→ 应用侧；CUB/device histogram → D；钩子 → C-04。
9. **误区 + SOP + 下一章钩子**。

**写作路线**

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Microbench-first**：固定 filter/计数核，扫争用强度；三路径对照 | 与 C-01/C-02 同构；直接回答「何时聚合」 |
| 2 | 复述 NVIDIA filtering blog + API 罗列 | 易与官方重复；**不推荐**作主线 |
| 3 | ARC/渲染 workload 复现 | 重、抢 Module 外叙事；**仅扩展阅读** |

**MVP 可行性评估**

| 编号 | 配置 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | Naive：每活跃线程 `atomicAdd` 到 **global** 计数器（或极少 bin） | ✅ | **必做（基线）** |
| B | Shared staging：block 内 SMEM atomic → 每 block 一次（或少量）global | ✅ | **必做（对照）** |
| C | Warp-aggregated：`ballot`/`coalesced` + leader `atomicAdd`（global） | ✅ | **必做（主处方）** |
| D | `sweep`：扫争用强度（建议 `hit_rate∈{0.05,0.125,0.25,0.5,1.0}` 或等价 `nbins`） | ✅ | **必做（主曲线）** |
| E | 定点：同配置下 A/B/C 全表 + 正确性（期望计数） | ✅ | **必做**（`modes`） |
| F | 可选：warp 聚合 + SMEM atomic（与 C 对照） | ✅ | **可选定点**；不进主 sweep 也可 |
| G | NCU：高争用点 A vs C | ⚪ | **可选旁证** |
| — | 多 GPU atomics / 可微渲染全链路 / DeviceHistogram 生产 API | ❌ | **不做** |

**最小可复现实验（`03_atomics_contention.cu`）**

| mode | 要回答的问题 | 进主结论？ |
|---|---|---|
| `naive` | 每线程 global atomic 时延？ | 基线 |
| `smem` | block staging 时延？ | 对照 |
| `agg` | warp 聚合后 global atomic 时延？ | 主处方 |
| `sweep` | 扫 `hit_rate`（或 nbins）：`agg/naive`、`smem/naive` 形状？ | **主曲线** |
| `modes` | 定点全表（固定高争用 + 中等争用各一档亦可） | 写结果用 |

实现约定（写入大纲供实现时遵守）：

- 工作负载形态对齐 NVIDIA filtering Pro Tip 的简化版：**谓词过滤计数**（或「命中则 +1」），便于改 `hit_rate` 而不改访存 footprint 故事。
- 主证据：CUDA event **median**；结果写回防 DCE；verify 与 host 期望计数比对。
- 启动打印 GPU / `sm_XX`。
- 默认 **不加** profile shell；用户明确要 NCU 再补。
- 聚合实现二选一写清：优先 **`coalesced_threads`（承接 C-02）** 或 **`__ballot_sync`+elect（承接 C-01）**；正文可提两者等价意图，示例只维护一条主路径以免分叉。

**证据最低要求**

- 主证据：median 时延与加速比；`sweep` CSV → `docs/results/C-03_*.csv` + 摘要 md。
- 正确性：各 mode 计数一致；失败非零退出。
- 正文必须写清：本机加速比是 **争用微基准形状**，不直接对齐 Kepler blog 的 20× 绝对数；也不替代 ARC 应用侧数字。

**本机要验证的「文献形状」假设（1～3）**

1. **高争用**：warp 聚合相对 naive global **显著更快**（文献可达数量级；本机以 5090 曲线校准，预期至少稳定拉开）。
2. **低争用**：聚合收益收窄甚至被额外指令抹平——sweep 应显示加速比随 hit_rate（或争用）变化。
3. **SMEM staging vs 聚合 global**：不预设谁永远赢；本机对照后写进决策表（可能「聚合 global 足够、staging 反贵」）。

**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA Programming Guide — Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)（§ Atomic Functions / `atomicAdd`） | `atomicAdd` 等语义；整数/浮点支持范围随架构；global/shared | §3 API；MVP 类型选择 |
| A 官方 | [Advanced Kernel Programming — scoped atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html) | `cuda::atomic` / scope；SMEM vs global 选型提示 | §3 旁注；不做 libcu++ 主线 |
| B 工程 | NVIDIA, [CUDA Pro Tip: Warp-Aggregated Atomics](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/) | 过滤场景聚合；Kepler 上相对 naive 可大幅提升；SMEM 未必更优；编译器可能自动聚合 | TL;DR②③④；MVP；形状假设①② |
| B 工程 | NVIDIA, [Cooperative Groups blog](https://developer.nvidia.com/blog/cooperative-groups/)（aggregated atomic 段） | `coalesced_group` 写聚合更安全 | 承接 C-02；MVP C |
| B 工程 | NVIDIA 论坛讨论（shared vs global atomic，Turing+） | 现代 GPU 上 SMEM atomic 不总是更快 | TL;DR③；假设③ |
| C 实证 | C-01/C-02 本机结果（ballot/coalesced 入口） | 聚合形态已 verify；本章补争用轴 | 基线钩子 |
| C 实证 |（可选）社区/内部 microbench 笔记：争用度 vs 加速比形状 | 佐证「低争用收益收窄」 | §5 预期曲线 |
| D 前沿 | Durvasula et al., **ARC**, ASPLOS’25（[DOI:10.1145/3669940.3707238](https://doi.org/10.1145/3669940.3707238)；[PDF](https://www.embarclab.com/static/media/arc.3f76b7ecf6e4d4fcda8f.pdf)） | 同 warp 同地址原子可寄存器规约；自适应分流；渲染梯度原子墙；SW 路径仍建在 warp reduce 上；应用侧平均约 **2.6×**（至 5.7×） | §7；动机；**不**替代本章 microbench |
| D 前沿 | Garcia de Gonzalo et al., CGO’19（自动生成 warp 原语/原子规约） | 编译器侧聚合思路 | §7 可选 |

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 高争用 → warp 聚合少原子次数 | TL;DR / 决策表 / MVP |
| SMEM staging 有条件，以本机为准 | TL;DR③ / 决策表 |
| 编译器可能已聚合 → 对照要诚实 | TL;DR④ / 误区 |
| ARC / 渲染梯度原子墙 | **仅** §扩展阅读 |
| DeviceHistogram / 多 block 规约 | **仅** → Module D |

**交付进度**

- [ ] 用户确认本大纲
- [ ] `examples/03_compute_primitives/03_atomics_contention.cu`
- [ ] 正文 + 封面/原理图 + plot
- [ ] `docs/results/C-03_*`（本机实测后）
- [ ] 回填规划总表
