# C-08 写作大纲：PDL（同流依赖核提前启动）

> 状态：✅ **已收口**（5090：`serial/pdl` 0.95×～1.14×；定点 1.13×；`pdl_full/pdl`=0.999×）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §4.3 · [Module-C.md](Module-C.md) · 上章：[`C-07_persistent.md`](C-07_persistent.md)

**路线**：**Microbench-first**——同 stream 上两个**有数据依赖**的短核：先发完再发 vs PDL 允许次核提前启动。  
**硬件门槛**：**sm_90+**（Hopper / Blackwell）。5090 = `sm_120` 可测。CC 不够则打印属性后清晰退出。  
**证据口径**：裸跑 CUDA event **median**。NCU/NSYS 可选，不当墙钟结论。  
**读者用语**：正文 § 称「实验怎么设计」，勿写「MVP」。

**标题**：`C-08. PDL：同流依赖核的提前启动与何时叠不上`

**选题裁决（相对候选槽「计算侧多 kernel 重叠」）**

| 选项 | 裁决 | 原因 |
|---|---|---|
| **PDL / Programmatic Dependent Launch** | **本章主线** | 规划 §2.2 写明 C-07～C-09 可收 PDL；C-06 已把 PDL 深挖指到本槽；命题是**依赖核**提前启动，不是 Host 再讲一遍 stream |
| Host 双 stream 两个独立核 | **不做主测** | 独立核重叠是 A-08 的 issue order；本章只一句对照 |
| CDP / 设备 launch 子网格 | **不做** | launch 税从 device 再付一遍；不是「依赖核重叠」 |
| Graph 并发 node | **不做主测** | 图内依赖 ≠ 自动重叠已在 C-06 一句；不重开 capture |
| Named barrier / `bar.sync` | **不做** | C-09 候选；PDL 用 `cudaGridDependencySynchronize` |

**边界**

| 已有章节 | 已覆盖 | 本章深化 / 避免 |
|---|---|---|
| A-08 | Host Stream / CE：拷贝∥计算、depth-first | **不重测** H2D 流水线；一句：独立核走异流，依赖核同流才谈 PDL |
| C-06 | Graph replay 摊提交 | **不重测** `stream/graph` 曲线；一句：Graph 摊提交，PDL 摊的是「次核不必等主核整网格退休」 |
| C-07 | occupancy 常驻、1-block 串行 vs 整卡 | **不重测** 一任务一发；一句：常驻占满 SM 时次核会饿，PDL 也要留 occupancy |
| C-04 | `grid.sync` / coop | 不把 PDL wait 写成第三层 barrier 课 |
| C-05 | body fusion | 不重测 elementwise 融合；该 fuse 就不该拆成两核再 PDL |
| B-07 / B-08 | 设备内 async / TMA | 不做 pipeline / mbarrier 主线 |
| Module D | CUTLASS PDL GEMM | 只钩子 |
| C-09（候选） | named barrier | 不做 |
| Module E | 框架 Graph | 不做 |

边界一句话：**同 stream、有依赖的两个计算核，比「等主核退完再发」和「PDL 允许次核提前 boot」；独立核去 A-08，Graph 已在 C-06。**

**已锁分叉（开写前必须钉死）**

| 分叉 | 锁定 | 禁止 |
|---|---|---|
| 依赖关系 | K2 **必须**读 K1 写过的数据；K2 入口 `cudaGridDependencySynchronize` | 两个独立核假装 PDL（那是 A-08） |
| 重叠从哪来 | K1：写完可见数据 → `cudaTriggerProgrammaticLaunchCompletion` → 再跑 **tail**（K2 不需要） | 只测「空核提前 launch」当主曲线（会变成数 µs 噪声） |
| 网格 | 默认 **半 occupancy**（两边都能住得下）。`modes` 加一行 **满 occupancy** 反例 | 一上来就占满 SM，再抱怨 PDL ~1× |
| 读数 | 1×、满 occupancy 叠不上都是合法结论 | 为赢把两核改成无依赖，或把 tail 做成和 K2 无关的假重叠却不写清 |

**默认夹具（禁止为赢而改）**

| 项 | 默认 | 扫描 |
|---|---|---|
| `n` | 1<<20 floats | 定点可改；两边相同 |
| `work` | 512（K2 在 wait **之前**的独立 FMA） | `sweep`：扫 `work` |
| `tail` | 512（K1 在 trigger **之后**的 FMA） | `sweep_tail`：扫 `tail` |
| `block` | 256 | 32 的倍数 |
| `serial` grid | 半 occupancy（`occ×SM/2`，至少 1） | — |
| `pdl` grid | 与 serial **相同** | — |
| `pdl_full` grid | occupancy × SM | `modes` 定点一行 |
| runs / warmup | 7 / 2 | — |

**TL;DR 目标**

1. **PDL = 同流上下一个核可以在主核退休前启动。** Host 仍按顺序 `<<<K1>>>` 再 `cudaLaunchKernelEx(K2, ProgrammaticStreamSerialization)`。不是异流、不是 CDP。
2. **次核提前 boot ≠ 数据已经可见。** K2 必须 `cudaGridDependencySynchronize`（或等价 fence）。只开 attribute、不 wait，是数据竞态，不是加速。
3. **能叠的是主核 tail 与次核 wait 之前的独立 work。** trigger 之后主核还要干活、次核在 fence 前也要有活，墙钟才可能短于串行。两边都极短，只剩 launch 隐藏，加速比贴 1×。
4. **占满 occupancy 就叠不上。** 两个满网格会抢 SM；`pdl_full` 不应系统性大赢。C-07 常驻占满时次核会饿，同一条物理约束。
5. **判停**：主看 `serial/pdl` median。1× 有效。禁止 ncu 附着墙钟。禁止为赢拆掉依赖或改成双 stream。

**建议正文结构**

1. 问题：A-08 叠的是独立活 / 拷贝∥计算；依赖核通常同流串行。还能不能提前启动次核？  
2. 物理模型：同流退休 vs PDL 提前 boot；ASCII 或一张图（二选一）。满 occupancy vs 半 occupancy。  
3. API：`cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization`；K1 `cudaTriggerProgrammaticLaunchCompletion`；K2 `cudaGridDependencySynchronize`。  
4. 决策表：独立核→A-08；静态短链→C-06；该 fuse→C-05；依赖且有 tail/prologue→试 PDL；满 occupancy→别指望叠计算。  
5. 实验怎么设计 + `--mode sweep`。  
6. 实测表 + plot（无 5090 则待测）。  
7. §7：CUTLASS / Graph programmatic edge → D / C-06 已述；不写成必须 C-09。  
8. SOP / 误区。

**可行性（实验矩阵）**

| 配置 | 裁决 |
|---|---|
| `serial`：同 stream `K1` 然后 `K2`（无 PDL attribute） | **必做（基线）** |
| `pdl`：K1 trigger + tail；K2 `LaunchKernelEx` + 独立 work → wait；半 occupancy | **必做** |
| `sweep`：固定 `tail`，扫 `work` | **必做（主曲线）** |
| `sweep_tail`：固定 `work`，扫 `tail` | **必做（重叠轴）** |
| `modes`：定点 + 打印 CC / occ / grid + **`pdl_full` 一行** | **必做** |
| 正确性：K2 读到的值与 host 参考一致 | **必做** |
| CC < 9.0 清晰退出 | **必做** |
| Host 双 stream 独立核 | **不做主测**（A-08；正文一句） |
| Graph programmatic edge 主对照 | **不做**（C-06；§7 一句） |
| CDP | **不做** |
| NCU / NSYS | 可选；默认不加 profile 壳 |

**主命令**：`./bin/03_compute_primitives_08_pdl --mode sweep`

| mode | 要回答的问题 |
|---|---|
| `serial` | 依赖核同流串行有多长？ |
| `pdl` | 半 occupancy 下提前启动能不能短？ |
| `sweep` | 扫 K2 `work`：加速比抬还是贴 1？ |
| `sweep_tail` | 扫 K1 `tail`：没有尾巴还能赢吗？ |
| `modes` | 定点 + `pdl_full` 满 occupancy 反例 |

实现约定：

- 启动打印 GPU / `sm_XX` / `SMs` / `blocks_per_sm` / `grid` / `grid_full`。  
- CC < 90：退出码非 0，写明门槛。  
- 计时：event 与 kernel **同 stream**（B-04 作废跑）。  
- 防 DCE：K1 写 `out[]`，K2 读后写 `sink`。  
- 默认不加 profile shell。

**本机要验证的文献形状**

1. 半 occupancy + `tail` 与 `work` 都够：`pdl` 相对 `serial` 应更快（叠 tail 与次核）。  
2. `tail→0`：加速比应收向 ~1×（只剩提前 boot / launch 隐藏）。  
3. `pdl_full` 不应系统性大赢。  
4. 正确性：去掉 K2 的 wait 若能稳定复现错数，只写进误区，**不进主 sweep**。

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 依赖核 + 有 tail → 可试 PDL | TL;DR / 主曲线 |
| 无 tail / 满 occupancy → 别指望 | TL;DR③④ / `sweep_tail` / `pdl_full` |
| 独立核 → A-08 | 决策表 |
| 该 fuse → C-05 | 决策表 |
| Graph programmatic edge / CUTLASS PDL GEMM | **仅** §7 |

**参考文献池**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A | [CUDA PG — PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) | 同流主次核；次核 `ProgrammaticStreamSerialization`；主核 `cudaTriggerProgrammaticLaunchCompletion`；次核必须 `cudaGridDependencySynchronize` | §2～3 |
| A | Runtime：`cudaLaunchKernelEx` / `cudaLaunchAttributeProgrammaticStreamSerialization` | 可扩展 launch 才带 attribute | API |
| B | [CUTLASS Dependent kernel launches](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html) | Hopper/Blackwell：有 GMEM 冲突的两核也可安全重叠；次核等主核 flush | SOP；§7→D |
| B | [H100 course: Kernel Launch Control](https://cudacourseh100.github.io/pages/lesson-8.2.html) | `griddepcontrol.launch_dependents` / `wait`；attribute 只是 Host 许可位 | §3 |
| C | C-06 本机：短核 Graph ~4×，长核 →1.01× | 另一层杠杆（提交） | §1 |
| C | C-07 本机：满 occupancy 常驻会占住 SM | 满网格 PDL 叠不上的同族约束 | TL;DR④ |
| D | CUTLASS / CuTe Blackwell PDL GEMM 例 | 生产 GEMM 用 PDL 叠 epilogue/prologue | §7 → D-05 |
| D | Graph `cudaGraphDependencyTypeProgrammatic` | 图上的 PDL 边 | §7；不进必做 |

**交付 checklist**

- [x] 用户确认本大纲（按序写 C-08；PDL 主线）
- [x] `examples/03_compute_primitives/08_pdl.cu`
- [x] 5090 `--mode sweep` → `docs/results/C-08_*` + plot
- [x] 正文 + 封面/原理图（图与 ASCII 不叠）
- [x] 回填规划 / Module-C / README
