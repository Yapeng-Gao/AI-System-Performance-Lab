# C-02 写作大纲：Cooperative Groups（tile / coalesced；Cluster 可选）

> 状态：✅ 文章+示例+RTX 5090 实测已落地。对照审稿用；以正文与 `docs/results/C-02_*` 为准。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-C.md`](Module-C.md) · 上章：[`C-01_warp_primitives.md`](C-01_warp_primitives.md)
>
> **已交付**：`02_cooperative_groups.cu`；正文；`docs/results/C-02_*`；`scripts/plot_c02_cooperative_groups.py`；封面/原理图。  
> **本机要点**：抽象税 tile32/cg_reduce ≈ **1.00× / 0.99×**；tile 悬崖 64/128 → **1.60× / 1.66×**；coalesced+cluster verify OK。

> **路线**：Microbench-first——主对照 **C-01 手写 `*_sync` vs `thread_block_tile`**；副线 **`coalesced_group` 聚合形态**；**tile size 悬崖**（≤32 vs >32）；Cluster/`this_cluster` **可选支线**（sm_90+，5090=`sm_120` 可跑但不进主结论）。
>
> **硬件门槛**：主路径 **不限 sm_90+**（CG tile/coalesced 自 CUDA 9 起常用）。Cluster 模式 CC\<90 清晰跳过。
>
> **目录**：已有 `article/03_*` / `examples/03_*`；本章只增 `02_cooperative_groups.cu` 等，**勿再建空壳**。

**标题（拟定）**：`C-02. Cooperative Groups：安全分组、tile 集体与 coalesced 聚合`

**与前后章的边界**

| 已有 / 规划章节 | 已覆盖 | C-02 应深化 / 避免重复 |
|---|---|---|
| **C-01** | `__shfl_*_sync` / ballot / mask 纪律；smem vs shfl 加速比 | **不重测** smem 树主曲线；只把「手写 shfl」当 **基线标签**，证明 CG tile≤32 ≈同速 |
| A-04 | Divergence / Independent Thread Scheduling | **不重讲**机制课；`coalesced_threads` 只作安全分组入口 |
| B-10 | 访存 Checklist | 不重开 coalescing/TMA |
| **C-03**（下） | Atomics / contention | coalesced 聚合原子只给 **入口形态 + 正确性**；争用曲线留给 C-03 |
| **C-04**（下） | 同步分层 / grid sync | **不做** `this_grid` + `cudaLaunchCooperativeKernel` 全家桶；一句钩子 |
| Module D | DeviceReduce / Softmax | **不做** 多 block 全局规约；`cg::reduce` 只作 tile/block 子集对照 |
| Hopper/Blackwell Cluster | DSMEM / cluster barrier | **可选支线**：`this_cluster` + 小 DSMEM 读写正确性；不抢 B-08 TMA 叙事 |

**TL;DR 目标结论（写作时先写死；有实测后改成带本机数字）**

1. **CG = 分组句柄，不是更快的 shuffle**：`thread_block_tile<N>`（N≤32）把 mask/rank 收进类型；热路径性能应对齐手写 `__shfl_*_sync`（本机以加速比≈1 为预期形状）。
2. **该上 tile**：固定子 warp / 半 warp 集体、要少手写 mask、要把 group 当参数传到 `__device__` 函数时——用静态 `tiled_partition<N>`。
3. **tile 大小悬崖**：N≤32 走 warp 原语；**N>32** 的 CG tile sync/reduce 是更通用的软件路径，常 **显著慢于** CUB/`__syncthreads` 硬件 block barrier（NVIDIA 论坛口径：勿拿 CG 大 tile 当 block-wide 默认）。
4. **`coalesced_group`**：发散后「此刻活跃线程」的安全集合；适合聚合原子入口；**不是**「我想要的任意逻辑 mask」的替代品（与 C-01 `activemask` 纪律同族）。
5. **判停**：裸跑 median 看 `cg_tile32 / intrinsic` 形状 + `tile>32` 是否崩；Cluster 有则 verify OK；**禁止**把 ncu 附着墙钟当结论。

**建议正文结构（8～9 节）**

1. **问题**：C-01 钉死 intrinsic 后——如何少写 mask、把分组变成可组合 API。
2. **物理 / 模型（短）**：implicit group（block）→ partition → tile / coalesced；ASCII：固定 tile vs 动态活跃集。
3. **API 分层**：`this_thread_block` → `tiled_partition` / `thread_block_tile` → `coalesced_threads` →（可选）`this_cluster`；集体：`sync` / `shfl_*` / `reduce`。
4. **决策表**：何时 tile、何时 coalesced、何时退回手写、何时上 CUB、何时别用大 tile CG reduce。
5. **MVP 实验** + 主命令 `--mode sweep`。
6. **实测**：表 + 曲线；口径 median。
7. **旁证（可选）**：NCU `inst_executed`（tile32 vs intrinsic 应对齐；tile>32 指令/stall 恶化）。
8. **扩展阅读**：CUB BlockReduce；grid cooperative launch → C-04；ARC → C-03；Cluster DSMEM 深挖（扩展）。
9. **误区 + SOP + 下一章钩子（C-03）**。

**写作路线**

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Microbench-first**：同一 block reduce，对照 intrinsic / CG tile32 / CG reduce(tile) / coalesced 聚合；`sweep` 扫 tile 大小 | 与 C-01 同构；直接回答「抽象税」与「>32 悬崖」 |
| 2 | 教程重写 NVIDIA CG blog API 罗列 | 易与官方重复；**不推荐**作主线 |
| 3 | Cluster-first（DSMEM） | 5090 能跑，但易抢 C-04 / 访存叙事；**作可选支线** |

**MVP 可行性评估**

| 编号 | 配置 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | 手写 `__shfl_down_sync` warp→SMEM→warp（复用 C-01 模式，作基线） | ✅ | **必做（基线）** |
| B | `thread_block_tile<32>` + `tile.shfl_down` 同构规约 | ✅ | **必做（主对照）** |
| C | `cg::reduce(tile, val, plus)`（tile=32） | ✅ | **定点必做**；进 sweep 的 tile=32 点 |
| D | `tiled_partition<N>`，`N∈{8,16,32,64,128}` 的 reduce 时延 | ✅ | **必做（主曲线 = 相对 tile32 或相对 intrinsic）**；预期 N>32 变差 |
| E | `coalesced_threads` + 聚合计数/atomicAggInc 形态（正确性） | ✅ | **定点必做**；不进主加速比横比 |
| F | Cluster：`this_cluster` + `map_shared_rank` 小读写 + `cluster.sync`（sm_90+） | ✅ sm_90+ | **可选支线**；CC\<90 跳过；**不进主 sweep** |
| G | NCU：A vs B；D 在 N=32 vs N=128 | ⚪ | **可选旁证** |
| — | `this_grid` / multi-grid / 生产 DeviceReduce / 深挖 DSMEM+TMA | ❌ | **不做**（→ C-04 / D / B-08） |

**最小可复现实验（`02_cooperative_groups.cu`）**

| mode | 要回答的问题 | 进主结论？ |
|---|---|---|
| `intrinsic` | C-01 风格手写 shfl block reduce 时延？ | 基线 |
| `tile32` | `thread_block_tile<32>.shfl_down` 同构？ | 对照 |
| `cg_reduce` | `cg::reduce` @ tile32？ | 定点 |
| `coalesced` | coalesced 聚合计数正确？ | 定点表 |
| `sweep` | 扫 `tile∈{8,16,32,64,128}`：CG reduce（或 tile shfl 树）时延形状？ | **主曲线** |
| `cluster` | sm_90+ DSMEM 小读写 verify？ | 可选 |
| `modes` | 定点全表一次跑齐 | 写结果用 |

**证据最低要求**

- 主证据：CUDA event **median**；`sweep` CSV → `docs/results/C-02_*.csv` + 摘要 md。
- 正确性：float/int 规约与 host 参考比对；coalesced/cluster 失败非零退出。
- 启动打印 GPU / `sm_XX`；`cluster` 在 CC\<90 跳过并打印。
- 默认 **不加** profile shell；用户明确要批量 NCU 再补。
- 防 DCE：结果写回 device 可见存储；大 tile 若需 `block_tile_memory`（旧 CC）按文档预留，5090/sm_80+ 按官方说明处理。

**本机要验证的「文献形状」假设（1～3）**

1. **tile≤32**：CG tile / `cg::reduce` 相对手写 intrinsic **≈持平**（抽象税可忽略；形状假设，非保证 bit 级相同指令数）。
2. **tile>32**：CG 路径相对 tile32（或 CUB/硬件 block sync 叙事）**明显变慢**——论坛结论「大 tile 勿当 block-wide 默认」应在本机曲线上可见。
3. **coalesced**：正确聚合；时延含原子，不与 reduce 加速比硬比。Cluster（若跑）：功能正确即可，不作主加速比故事。

**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA Programming Guide — Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) | implicit group → partition；`tiled_partition` / `coalesced_threads` / `this_cluster`；集体 sync/reduce | §2～3 API；MVP |
| A 官方 | [Device-Callable APIs — Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html) | `thread_block_tile`、`cluster_group`、`block_tile_memory`（老 CC 大 tile）、DSMEM `map_shared_rank` | §3；MVP D/F |
| A 官方 | [CUDA C++ PG — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（Clusters 节） | sm_90+ 共调度；portable cluster size≤8；`cluster.sync` | 可选支线；边界 |
| B 工程 | NVIDIA, [Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/) | 静态 tile 优化；`coalesced_threads` 发散分支；warp-aggregated atomic 示例 | TL;DR②④；MVP E |
| B 工程 | NVIDIA 论坛，[CG vs CUB reduce](https://forums.developer.nvidia.com/t/cooperative-groups-are-much-slower-than-cub/312479) | tile≤32 走 warp sync 快；**>32 软件通用 sync，不宜当 block-wide 默认**；建议子集 tile | TL;DR③；决策表；假设② |
| B 工程 | CCCL/CUB — [Block/Warp collectives](https://nvidia.github.io/cccl/cub/) | 生产 block reduce 优先 CUB；CG 负责分组表达 | §7；误区 |
| C 实证 | Zhang et al., [arXiv:2004.05371](https://arxiv.org/abs/2004.05371)（SyncMicrobenchmark） | CG 引入后 warp/block/grid sync 微基准；grid sync 成本远高于 intra-SM | §2 旁证；钩子 → C-04 |
| C 实证 | C-01 本机结果 [`C-01_warp_primitives.md`](../results/C-01_warp_primitives.md) | 手写 shfl 基线形状；本章对比用 | 基线标签 |
| D 前沿 | Collange, [Warp-synchronous programming with CG](https://www.irisa.fr/alf/downloads/collange/talks/collange_warp_synchronous_19.pdf)（讲义） | coalesced 可组合性 vs tile 性能；不规则 partition 路线图 | §7；误区 |
| D 前沿 | Durvasula et al., ARC, ASPLOS’25（[DOI:10.1145/3669940.3707238](https://doi.org/10.1145/3669940.3707238)） | 自适应原子仍建在 warp/CG 聚合上 | 钩子 → **C-03** |
| D 前沿 | Cluster / DSMEM 工程笔记（如 cuda-oxide cluster 章） | DSMEM 延迟量级、occupancy 约束；**不**替代本章主结论 | 可选支线扩展 |

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| tile≤32 ≈ intrinsic；少写 mask | TL;DR / MVP |
| tile>32 勿默认当 block reduce | TL;DR③ / 决策表 |
| coalesced ≠ 任意逻辑 mask | TL;DR④ / 误区 |
| CUB BlockReduce / grid cooperative launch | **仅** §扩展（→ 库 / C-04） |
| Cluster DSMEM 深挖 / ARC | **仅** 可选支线或 → C-03 |

**交付进度**

- [x] 用户确认本大纲
- [x] `examples/03_compute_primitives/02_cooperative_groups.cu`
- [x] `scripts/plot_c02_cooperative_groups.py` + `docs/results/C-02_*` + plot
- [x] 正文 + 封面/原理图
- [x] 回填规划总表 ✅
