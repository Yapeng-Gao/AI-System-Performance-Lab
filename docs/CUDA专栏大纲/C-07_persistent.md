# C-07 写作大纲：Persistent Kernel（常驻网格 / 拉活）

> 状态：✅ **5090 已测**（sweep 32×→3740×；work=4096→314×；oversub 更慢）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §4.3 · [Module-C.md](Module-C.md) · 上章：[`C-06_cuda_graph.md`](C-06_cuda_graph.md)

**路线**：**Microbench-first**——同一批短任务：逐次 `<<<>>>` vs occupancy 尺寸的常驻网格、block leader 拉活。  
**硬件门槛**：**不限 sm_90+**。主路径普通 launch + occupancy 定网格。cooperative / `grid.sync` **不做主测**（已在 C-04）。  
**证据口径**：裸跑 CUDA event **median**。NCU/NSYS 可选，不当墙钟结论。  
**读者用语**：正文 § 称「实验怎么设计」，勿写「MVP」。

**标题**：`C-07. Persistent Kernel：常驻网格、拉活与何时别再 launch`

**选题裁决（相对候选槽「Persistent 或 Warp specialization」）**

| 选项 | 裁决 | 原因 |
|---|---|---|
| **Persistent / 常驻网格** | **本章主线** | C-06 钩子就是「另一种摊销 launch」；命题独立可测 |
| Warp specialization | **不做主测** | 产消 warp / TMA 路径是 B-07 / D-05；本章只 §7 钩子 |
| Megakernel / 编译器融合多算子 | **不做** | 抢 D / 前沿编译器；§7 一句 |

**边界**

| 已有章节 | 已覆盖 | 本章深化 / 避免 |
|---|---|---|
| C-06 | 短核链 Graph vs 逐次 launch | **不重测** Graph replay 曲线；一句：Graph 摊的是提交，Persistent 摊的是「不退出」 |
| C-04 | `grid.sync` / cooperative 门槛 / 多阶段 vs 多 kernel | **不重测** 三层 sync；Persistent **不依赖** `this_grid().sync()` |
| C-05 | body fusion | 不重测 elementwise 融合 |
| C-03 | 每线程 `atomicAdd` 争用 | 队列只用 **block leader 一次 atomic**；不重开聚合课 |
| A-08 | Host Stream / CE | 不讲 Host 重叠 |
| B-03 | occupancy 台阶 | 只调用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 定网格，不重开 spill 课 |
| B-07 / B-08 | `cp.async` / TMA | **不做** 异步引擎 / 产消 warp |
| Module D | GEMM tile scheduler / CUTLASS persistent | 不做算子正文；CUTLASS persistent GEMM 只钩子 |
| C-08（候选） | 设备侧多 kernel 重叠 | 不做多 stream 全家桶 |
| Module E | 框架 Graph / 推理 serving | 不做 vLLM / megakernel 产品栈 |

边界一句话：**同一批短任务，比「一任务一发 launch」和「occupancy 网格常驻后 block leader 拉活」；Graph 已在 C-06，不再当第三主对照。**

**已锁分叉（开写前必须钉死）**

| 分叉 | 锁定 | 禁止 |
|---|---|---|
| `launch` 粒度 | **1 task = 1 `<<<1, block>>>`** | 把多任务 batch 进同一个核再 launch（会把加速比做成 ~1×） |
| 拉活 | **`threadIdx.x==0` `atomicAdd` 取 1 个 task，`__syncthreads` 广播** | 每线程抢活（滑回 C-03）；纯 grid-stride 扫完全部工作（假常驻，更像 C-05） |
| 常驻网格 | `blocks_per_sm = OccupancyAPI(persistent_kernel)` × `multiProcessorCount` | 用 `maxBlocksPerSM` 硬猜；主路径不用 coop |
| 读数 | 1×、队列尾波不均都是合法结论 | 为了让 persist 赢而改成 batch / 加长单任务 / 换夹具 |

**默认夹具（禁止为赢而改）**

| 项 | 默认 | 扫描 |
|---|---|---|
| `n_tasks` | 4096 | `sweep`：64, 256, 1024, 4096, 16384 |
| `work` | 1（FMA iters，与 C-06 同族） | `sweep_work`：0, 1, 8, 64, 512, 4096（固定 `n_tasks=4096`） |
| `block` | 256 | 定点可改；须为 32 的倍数 |
| `launch` grid | **1**（每个任务一个 block） | `--grid` 不用于 launch 主路径 |
| `persistent` grid | occupancy × SM | `--grid` 可覆盖；`oversub` 用 `occ_grid × factor` |
| `oversub` factor | 8 | `modes` 定点一行，不进主 sweep |
| 总工作量 | 两边都是 `n_tasks × work` 次 FMA + 各写 `out[task]` | 必须相同 |
| runs / warmup | 7 / 2 | — |

**TL;DR 目标**

1. **Persistent = 网格按 occupancy 裁到能常驻，block leader 循环拉任务。** 不是「一个 block 对应一个 tile 全发射出去」。网格过大只是 oversubscribe，不是常驻。
2. **和 Graph 摊的不是一层。** Graph：拓扑稳定时一次提交多次 replay。Persistent：少退出 / 少再 launch，适合任务数多、单任务极短、或拓扑不好 capture。
3. **短任务 + 一任务一发时，常驻会赢得很夸张。** 单任务变重，加速比会收，但夹具是 1-block 串行 vs 整卡，**收不到 C-06 那种 ~1×**（本机 1500×→314×）。
4. **定网格必须走 Occupancy API**，不要硬猜。regs / SMEM 一变，`numBlocksPerSM` 会掉。常驻是「按 occupancy 构造」，不是 coop 契约。
5. **判停**：主看 `persistent` 相对 `launch` 的 median。队列争用、尾波不均、加速比贴 1× 都是有效章。禁止把 ncu 附着 ms 当结论。禁止为赢改夹具。

**建议正文结构**

1. 问题：C-06 用 Graph 摊提交；还有一条路是核不退出。  
2. 物理模型：oversubscribe 调度 vs occupancy 常驻；一张图 **或** ASCII（二选一，不叠）。  
3. API：Occupancy 定网格 + block leader `atomicAdd`；与 coop / Graph 一句对照。  
4. 决策表：短任务多发 / 长任务 / 静态 DAG→Graph / 要跨 block 阶段→C-04 / 产消 warp→D-05。  
5. 实验怎么设计 + `--mode sweep`。  
6. 实测表 + plot（无 5090 数则标待测）。  
7. §7：warp spec / CUTLASS persistent GEMM / megakernel → 见 §10，不抢 D。钩子不写成必须 C-08。  
8. SOP / 误区 / 钩子（C-10 或候选 C-08，可选）。

**可行性（实验矩阵）**

| 配置 | 裁决 |
|---|---|
| `launch`：1 task = 1 `<<<1,256>>>`，总工作量固定 | **必做（基线）** |
| `persistent`：occupancy 定网格，block leader `atomicAdd` 拉 1 task 直到做完 | **必做** |
| `sweep`：扫 `n_tasks`（`work=1`）→ `launch` vs `persistent` | **必做（主曲线）** |
| `sweep_work`：固定 `n_tasks=4096`，扫 `work` | **必做（收窄轴）** |
| `modes`：定点全表 + 打印 `blocks_per_sm` / `persist_grid` + **oversub 一行** | **必做** |
| `oversub`：同一 persistent 核，网格 = occ×SM×8 | **定点必做**（兑现「过大不是常驻」） |
| 正确性：`out[task]` 与 host 参考一致 | **必做** |
| cooperative + `grid.sync` 常驻 | **不做**（C-04） |
| Graph 第三档主对照 | **不做**（C-06；正文一句） |
| 多任务 batch 进一个 launch | **不做**（会抹掉命题） |
| 每线程 `atomicAdd` / 纯 grid-stride 主 mode | **不做** |
| 工作窃取 / 多优先级队列 | **不做** |
| Warp specialization / TMA producer | **不做** |
| NCU / NSYS | 可选；默认不加 profile 壳 |

**主命令**：`./bin/03_compute_primitives_07_persistent --mode sweep`

| mode | 要回答的问题 |
|---|---|
| `launch` | 一任务一发有多贵？ |
| `persistent` | 常驻拉活相对 launch 快吗？ |
| `sweep` | 扫任务数：加速比抬升还是贴 1？ |
| `sweep_work` | 任务变重是否收向 1×？ |
| `oversub` | 网格打到 occupancy 数倍是否仍叫常驻？ |
| `modes` | 定点 + occupancy 网格 + oversub 一行 |

实现约定：

- 启动打印 GPU / `sm_XX` / `SMs` / `blocks_per_sm` / `persist_grid`。  
- 计时：warmup + 多次 run → **median**；量整段做完。  
- **event 与 kernel 同 stream**（`cudaEventRecord(start, stream)`）。禁止跨 stream 记空 gap（B-04 作废跑）。热路径内 **不要** `StreamSynchronize` 再记另一条流上的 event。  
- `next` 计数器在每次 persist 热路径前 `MemsetAsync` 到 0（可在 event 前，与 C-06 instantiate 一样不摊进「机制税」之外的公平性：memset 一次 vs n 次 launch，相对仍成立；默认 **memset 放 event 前**）。  
- 防 DCE：结果写回 `out[task]`。  
- 默认 **不加** profile shell。

**本机要验证的文献形状**

1. 极短任务 + 足够 `n_tasks`：`persistent` 相对 `launch` 应明显更快（摊 launch 税）。  
2. 增大 `work`：加速比应下降；**不要求**收到 ~1×（1-block vs 整卡）。  
3. `persist_grid` 应等于 `SMs × blocks_per_sm`（或按任务数截断），并打印出来。  
4. `oversub` 相对 occupancy 网格不应系统性大赢；过大只是普通 oversubscribe。

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 短任务 + 一任务一发 → 常驻可能赢 | TL;DR / 主曲线 |
| 长任务 → 加速比收、夹具不到 1× | TL;DR④ / `sweep_work` |
| 网格过大 ≠ 常驻 | TL;DR① / `oversub` |
| 静态短核链 → Graph（C-06） | 决策表 |
| ExecUpdate / torch Graph / megakernel / CUTLASS | **仅** §7 → E / D |

**参考文献池**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A | [CUDA Runtime — Occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html) | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 按 kernel 资源估每 SM block 数 | 定网格 |
| A | [CUDA PG — Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 常驻/coop 网格不能超过 occupancy×SM；超了 coop launch 失败 | §2；本章主路径**不用** coop |
| B | [CUDA Pro Tip: Occupancy API](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/) | 用 API 定 launch，不要硬猜 | SOP |
| B | [NVIDIA 论坛：persistent kernel](https://forums.developer.nvidia.com/t/question-about-persistent-kernel-concept/320600) | 常驻去掉 block 反复调度；无反复 launch 税；数据可留在片上 | TL;DR① |
| B | Gupta et al., [InPar 2012 Persistent Threads](https://doi.org/10.1109/inpar.2012.6339596) | PT 用队列绕过硬件 wave 调度；单队列 FIFO 是基线 | §2 模型 |
| C | C-06 本机：短核 `graph/stream` 3.7～4.1×，长核 →1.01× | 同族「税 vs body」；Graph 是另一杠杆 | §1 |
| C | [PyGraph, arXiv:2503.19779](https://arxiv.org/abs/2503.19779) | 文献：单次 launch 常 ~5–10 µs；短核会被吃掉 | 形状预期，不当本机绝对数 |
| D | CUTLASS persistent GEMM tile scheduler | 生产 GEMM 常驻是算子调度 | §7 → D-05 |
| D | Megakernel / Event Tensor 等 2025 编译器融合 | 多算子进一个常驻核 | §7；不进必做 |

**交付 checklist**

- [x] 用户确认本大纲（Persistent 主线；1 task=1 launch；block leader 拉活）
- [x] `examples/03_compute_primitives/07_persistent.cu`
- [x] 5090 `--mode sweep` → `docs/results/C-07_*` + plot
- [x] 正文 + 封面/原理图（图与 ASCII 不叠）
- [x] 回填规划 / Module-C / README
