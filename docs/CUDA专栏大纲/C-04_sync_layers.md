# C-04 写作大纲：同步分层（warp / block / grid）

> 状态：✅ 文章+示例+RTX 5090 实测已落地。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-C.md`](Module-C.md) · 上章：[`C-03_atomics_contention.md`](C-03_atomics_contention.md)
>
> **已交付**：`04_sync_layers.cu`；正文；`docs/results/C-04_*`；plot；封面/原理图。  
> **本机要点**：modes **grid/block ≈ 17.2×**；`block/warp` 仅 **~1.1×～1.23×**（次级）；`sweep_grid` @coop_max **0.63 ms（~4× vs 1 block）**；phases ratio **≈1.0**。
>
> **路线**：Microbench-first——主曲线对照 **三层同步相对代价**（`__syncwarp` / `__syncthreads` / `grid.sync`）；副线 **cooperative launch 门槛 + 多阶段 vs 多 kernel**；正确性定点：SMEM 交接缺 barrier 必挂。
>
> **硬件门槛**：主路径 **不限 sm_90+**。`__syncwarp` / `__syncthreads` 全架构常用；`this_grid().sync()` 需 **cooperative launch**（查 `cudaDevAttrCooperativeLaunch`；grid 规模受 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 约束）。不支持则清晰跳过 grid 模式并打印原因。
>
> **目录**：已有 `article/03_*` / `examples/03_*`；本章只增 `04_sync_layers.cu` 等，**勿再建空壳**。

**标题（拟定）**：`C-04. 同步分层：warp / block / grid 的代价、可见性与 cooperative launch`

**与前后章的边界**

| 已有 / 规划章节 | 已覆盖 | C-04 应深化 / 避免重复 |
|---|---|---|
| **C-01** | `*_sync` mask；`__syncwarp` 仅误区一句 | **不重测** shfl vs SMEM reduce；把 syncwarp 当 **可见性/会合** 原语测代价 |
| **C-02** | CG 分组 / tile / coalesced；`block.sync` 作 API | **不重测** tile 悬崖 / 抽象税；`this_grid` 全家桶留给本章；`block.sync` ≈ `__syncthreads` 一句对齐即可 |
| **C-03** | 争用曲线 / staging / 聚合原子 | **不重测** atomic 加速比；staging 后 barrier 只当「处方一步」承接 |
| A-04 | Divergence / ITS | 不重讲机制课；发散路径上 sync 参与集纪律一句挂钩 |
| B-07 / B-08 | `cp.async` / mbarrier / TMA | **不做** async arrive-wait / named barrier 主线；§7 钩子 |
| **C-05**（下） | Kernel fusion 代价 | 不做 fusion vs 多 kernel 寄存器/occupancy 全家桶；一句「同步点变少 ≠ 一定该 fuse」 |
| **C-06**（下） | Graph / launch overhead | **不深挖** host launch 墙钟与 Graph capture；多 kernel 只作 grid sync 对照基线 |
| **C-07 / C-09** | Persistent / named barrier 候选 | persistent 与 `bar.sync` 子集同步 **不做**；钩子即可 |
| Module D | 多 block 算子 | 不做生产 DeviceReduce；grid sync 只交付原语成本与正确用法 |

**TL;DR 目标结论（写作时先写死；有实测后改成带本机数字）**

1. **同步有层，且层间差数量级**：warp 会合 ≪ block barrier ≪ grid sync（文献 SyncMicrobenchmark / IISWC’24 形状）；处方是 **选对层**，不是「能 sync 就 sync」。
2. **`__syncwarp` ≠ 免费、≠ `__syncthreads` 子集替代**：同 warp 经 **内存** 交接要 syncwarp（或等价 fence+会合）；collective `*_sync` 已含参与集纪律——**不要**用 syncwarp 冒充 block 可见性。
3. **Block 默认 `__syncthreads` / `block.sync()`**：SMEM 生产者–消费者、整 block 阶段分界走硬件 block barrier；C-02 已证大 tile CG sync 勿当 block 默认。
4. **Grid sync 有门槛**：必须 `cudaLaunchCooperativeKernel` + 设备支持 + **可常驻** 的 grid 规模；收益场景是 **跨 block 多阶段且要保留 SMEM/寄存器状态**；否则多 kernel（隐式 barrier）往往更简单，绝对时延差未必大（文献：小规模下与 launch 差可达微秒级）。
5. **判停**：裸跑 median 看三层相对时延形状 +（有则）`nblocks` 对 grid sync 的斜率；正确性定点必过；**禁止**把 ncu 附着墙钟当结论。

**建议正文结构（8～9 节）**

1. **问题**：C-03 处方里的 barrier——本章回答「哪一层、多贵、何时上 grid」。
2. **物理模型（短）**：intra-warp → intra-SM(block) → inter-SM(grid)；ASCII：可见性范围；一句「隐式 barrier = kernel 边界」。
3. **API 分层**：`__syncwarp(mask)` → `__syncthreads` / `cg::sync(block)` → `this_grid().sync()` + cooperative launch 清单（属性 / occupancy / 失败码）。
4. **决策表**：warp 内存会合 / block 交接 / 跨 block 多阶段 / 何时退回多 kernel / 何时别上 grid。
5. **MVP 实验** + 主命令 `--mode sweep`。
6. **实测**：表 + 曲线；口径 median。
7. **旁证（可选）**：NCU stall 相关；或 `cuda-memcheck`/`compute-sanitizer --tool synccheck` 演示缺 sync（可选，不进主证据）。
8. **扩展阅读**：async barrier / named barrier → B-08/C-09；Graph/launch → C-06；钩子 → C-05。
9. **误区 + SOP + 下一章钩子**。

**写作路线**

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Microbench-first**：固定迭代空同步；对照 warp/block/grid 相对时延；`sweep` 扫 block 内 warp 数与（grid）block 数 | 与 C-01～C-03 同构；直接回答「层间差多少」 |
| 2 | 教程重写 CG / barrier API 罗列 | 易与官方重复；**不推荐**作主线 |
| 3 | 多阶段算子 + sanitizer 正确性-first | 教学强、曲线弱；作定点补强，勿作唯一门禁 |

**MVP 可行性评估**

| 编号 | 配置 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | `__syncwarp()` 空循环 × `iters`（全 warp 参与） | ✅ | **必做（基线层）** |
| B | `__syncthreads()` 空循环 × 同 `iters` | ✅ | **必做（对照层）** |
| C | `cg::this_grid().sync()` 空循环 × 同 `iters`（cooperative launch） | ✅ 需 coop | **必做（主对照）**；不支持则跳过并打印 |
| D | `sweep`：扫 `nwarps∈{1,2,4,8,16,32}`（固定 blockDim=nwarps×32）上 A/B 相对时延 | ✅ | **次级曲线**（本机幅度有限；防误读） |
| E | `sweep_grid` 或并入 sweep：扫 `nblocks`（受 coop occupancy 上限夹紧）上 C 的时延形状 | ✅ 需 coop | **必做（主曲线）**；文献：更相关 blocks/SM |
| F | 定点正确性：SMEM 生产者写 →（有/无）`__syncthreads` → 消费者读；缺 sync 期望失败或 sanitizer 可抓 | ✅ | **定点必做**（`correctness`） |
| G | 定点：两阶段「写 global → sync → 再读」：**单核 grid.sync** vs **两 kernel**（隐式 barrier）墙钟 | ✅ | **定点必做**；**不**深挖 launch 分解（→ C-06） |
| H | NCU / synccheck | ⚪ | **可选旁证** |
| — | multi-grid 多卡 / named `bar.sync` / `cuda::barrier` arrive-wait / persistent kernel / Graph | ❌ | **不做**（→ C-06/C-07/C-09/B-08） |

**最小可复现实验（`04_sync_layers.cu`）**

| mode | 要回答的问题 | 进主结论？ |
|---|---|---|
| `warp` | `__syncwarp` 空同步 median？ | 基线 |
| `block` | `__syncthreads` 空同步 median？ | 对照 |
| `grid` | `this_grid().sync` 空同步 median？（coop） | 对照 |
| `sweep` | 扫 `nwarps`：`block/warp` 相对比形状？ | **次级** |
| `sweep_grid` | 扫 `nblocks`：grid sync 时延形状？ | **主曲线** |
| `correctness` | SMEM 交接缺/有 barrier？ | 定点 |
| `phases` | grid.sync 单核 vs 两 kernel？ | 定点表 |
| `modes` | 定点全表一次跑齐 | 写结果用 |

实现约定（写入大纲供实现时遵守）：

- 主证据：CUDA event **median**；空同步核要防 DCE（累加 `clock64`/`threadIdx` 写回或 `volatile` 副作用——以本机不被优化掉为准）。
- 启动打印 GPU / `sm_XX` / `CooperativeLaunch` / coop 最大 grid 提示。
- `grid` / `sweep_grid` / `phases`：失败时打印 `cudaGetErrorString`，非零退出或跳过（与 C-02 cluster 同风格）。
- 默认 **不加** profile shell；用户明确要再补。
- `block.sync()` 可作为附录一行对齐，**不**另开主 mode（避免与 C-02 重复）。

**证据最低要求**

- 主证据：median 时延与 **层间加速比/相对比**；CSV → `docs/results/C-04_*.csv` + 摘要 md。
- 正确性：`correctness` 有 barrier 路径 verify OK；缺 barrier 路径不得冒充成功（或明确「UB/未定义，仅演示工具可抓」——正文写清，勿把 UB 当稳定失败断言）。
- 正文必须写清：本机数字是 **空同步微基准形状**，不直接对齐文献 V100/P100 绝对 ns；grid 收益叙事以「可复用状态」为主，不以「永远快过双 kernel」为卖点。

**本机要验证的「文献形状」假设（1～3）**

1. **层间差**：同 `iters` 下 `block` 明显贵于 `warp`；`grid` 再贵一截（数量级或稳定倍数——以 5090 曲线为准）。
2. **grid 规模**：grid sync 时延随 **block 数（尤其每 SM blocks）** 抬升，对 `blockDim` 不敏感（Zhang et al. 形状）。
3. **phases**：小规模下 grid.sync 单核相对两 kernel **未必更快**；若接近，正文写「选 grid 是为状态复用/算法表达，不是自动加速」。

**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA PG — Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/)（`__syncthreads` / memory fence 语义） | block barrier：会合 + 组内可见性 | §2～3 |
| A 官方 | [CUDA PG — Warp shuffle / syncwarp 语境](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions) + [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) | `__syncwarp`：warp 内会合 + memory fence；与 `*_sync` 集体分工 | TL;DR②；误区 |
| A 官方 | [CUDA PG — Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) | `this_grid` / `sync`；**必须** `cudaLaunchCooperativeKernel` | TL;DR④；MVP C |
| A 官方 | [CUDA Runtime — `cudaLaunchCooperativeKernel`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html) | 原子式 launch；失败则不可假定 grid sync 合法 | MVP；SOP |
| A 官方 | [Async Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html) | 全 block/warp 同步仍建议 `__syncthreads`/`__syncwarp` 更优；arrive-wait 另叙事 | §7 边界 |
| B 工程 | NVIDIA 论坛：`this_grid` + `<<<>>>` → `is_valid()==0` / launch failure | 常见坑：未走 cooperative launch | 误区；SOP |
| B 工程 | C-02 本机：大 tile CG sync 慢于硬件 block barrier | 强化「block 默认 syncthreads」 | 决策表；承接 |
| C 实证 | Zhang et al., **IPDPS’20** / [arXiv:2004.05371](https://arxiv.org/abs/2004.05371)（SyncMicrobenchmark） | warp/block/grid 微基准；grid 延迟更相关 **blocks/SM**；小规模相对 launch 差可达 **~2.5µs**（2 blocks/SM）；grid 价值含 SMEM/寄存器复用 | TL;DR①④；假设②③；§2 |
| C 实证 | IISWC’24，Characterizing CUDA and OpenMP syncs（[PDF](https://userweb.cs.txstate.edu/~burtscher/papers/iiswc24b.pdf)） | `__syncthreads` 吞吐至 warp 大小后随活跃 warp 变差；`__syncwarp` 跨机行为差异 | §2 旁证；假设① |
| C 实证 | C-01/C-02/C-03 本机结果 | 原语与聚合已落地；本章补同步轴 | 承接 |
| D 前沿 | Programmatic Dependent Launch（sm_90+ PG） | 跨 kernel 依赖的另一条路；**不**进 MVP | §7 → C-06/E |
| D 前沿 | Named barrier / warp specialization（CUTLASS/课程笔记） | 子集 warp 同步；候选 C-07/C-09 | §7 |

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 三层代价差；选对层 | TL;DR / 决策表 / MVP |
| syncwarp 可见性纪律；勿冒充 block | TL;DR② / 误区 |
| grid 需 coop launch + occupancy | TL;DR④ / SOP |
| grid 未必快过双 kernel | TL;DR④ / 假设③ / 定点 G |
| mbarrier / named barrier / Graph / PDL | **仅** §扩展（→ B-08 / C-06 / C-09） |

**交付进度**

- [x] 用户确认本大纲（含 `phases` 定点必做）
- [x] `examples/03_compute_primitives/04_sync_layers.cu`
- [x] 本机实测 + `docs/results/C-04_*` + plot
- [x] 正文 + 封面/原理图（TL;DR 以 grid/block 为主）
- [x] 回填规划总表 ✅
