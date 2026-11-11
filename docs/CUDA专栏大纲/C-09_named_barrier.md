# C-09 写作大纲：Named Barrier（CTA 内子集会合）

> 状态：✅ **已收口**（5090：两边相等 **1.97×**；work=0 / tail=0 为 **1.05× / 1.04×**；`named_pre` **1.007×**；`indep` **0.97×**）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §4.3 · [Module-C.md](Module-C.md) · 上章：[`C-08_pdl.md`](C-08_pdl.md)

**路线**：**Microbench-first**。一个 CTA 里两个 warp 做生产者 / 消费者：`__syncthreads` 把两边的独立活串成关键路径，对比 `barrier.arrive` + `barrier.sync` 只叠 **arrive 之后** 的那段。  
**硬件门槛**：**sm_70+**。子集参与、且不是每个未退出线程都执行同一条 aligned barrier 时，PTX 写明 sm_6x 及以下的 `barrier`（无 `.aligned`）等价于 aligned，行为按 aligned 的限制走。5090 = `sm_120` 可测。CC < 7.0 打印属性后非零退出。  
**证据口径**：裸跑 CUDA event **median**。NCU/SASS 可选，不当墙钟结论。  
**读者用语**：正文 § 称「实验怎么设计」，勿写「MVP」。

**标题**：`C-09. Named Barrier：子集会合、单边 arrive 与何时叠不上`

**选题裁决（C-09 必写；主题从候选里收这一条）**

规划 §2.2 允许 C-07～C-09 收 named barrier。C-07 已收 persistent，C-08 已收 PDL。本章收 **硬件 named barrier**（每 CTA 16 个，编号 0..15）。

| 选项 | 裁决 | 原因 |
|---|---|---|
| **Named barrier**（`barrier.sync` / `barrier.arrive`，带参与线程数） | **本章主线** | C-04 明确不做子集 `bar.sync`；PTX 把「不同 warp 对同一 barrier 一个 sync、一个 arrive」写成生产者/消费者模型 |
| Divergence 工程处方 | **不做主章** | 展开权在 A-04。本章只写一条纪律：按 **warp** 分角色，禁止 warp 内一部分线程进 barrier |
| `cuda::barrier` / mbarrier / `expect_tx` | **不做主测** | 那是 SMEM 里的异步 barrier，完成计数和 TMA 字节在 B-07 / B-08。官方 Async Barriers 章的 warp specialization 例走的是这条，不在本章复写 |
| Cluster / DSMEM barrier | **不做** | C-02 可选支线已碰到 `cluster.sync`；§2.2 仍只留钩子 |
| 空 `__syncthreads` 层间代价 | **不做** | C-04 已测 warp / block / grid |
| CUTLASS warp-specialized GEMM | **不做** | 只钩子 → D-05 |

**边界**

| 已有章节 | 已覆盖 | 本章深化 / 避免 |
|---|---|---|
| C-04 | `__syncwarp` / `__syncthreads` / `grid.sync` 的层间代价 | **不重测**空同步曲线。`__syncthreads` 只当「全体未退出线程」基线 |
| C-02 | `thread_block_tile` / coalesced；cluster 可选 | tile≤32 的 sync 不是这 16 个 CTA barrier 槽。不重测 tile 悬崖 |
| C-01 | `*_sync` mask | 不重讲 shfl |
| A-04 | SIMT / divergence / replay | 不重开机制课 |
| B-07 / B-08 | `cp.async` / TMA / mbarrier | 不 init mbarrier，不 `expect_tx` |
| C-05 | fusion 代价 | 该合成一个角色就不要拆成两个 warp 再 barrier |
| C-06 | Graph 摊提交 | 不重测 launch 墙 |
| C-07 | 常驻、占满 SM | 不重测一任务一发 |
| C-08 | 跨 kernel 的 PDL tail ∥ 次核 | **不重测** PDL。本章是 **同一个 CTA 里** 两个 warp 的 tail |
| Module D | CUTLASS mainloop 同时用 NamedBarrier 和 mbarrier | 只钩子 |

边界一句话：**CTA 内按 warp 分成两个角色，用编号 1..15 的 named barrier 做单边 arrive；全体 `__syncthreads` 串行的是同一段活。拷贝完成计数回 B-07/B-08，跨 kernel 重叠回 C-08。**

**已锁分叉**

| 分叉 | 锁定 | 禁止 |
|---|---|---|
| 指令 | 子集或分角色用 **非 aligned** `barrier.sync` / `barrier.arrive`。`bar.sync` 是 `barrier.sync.aligned` 的别名 | 只有一部分未退出线程执行 `bar.sync` 还当正确实现 |
| 编号 | 用 **1 和 2**。0 留给 `__syncthreads` | 用 0；用 ≥16（CUTLASS 观测到会折回 0..15，16 会撞上 0） |
| 计数 | 参与线程数，**32 的倍数**。生产者 arrive、消费者 sync 的是**同一条** barrier 时，计数 = 两边线程数之和（1 warp + 1 warp → 64）。mbarrier 只计到达的那一侧，不要混用这个数 | 把计数写成 warp 个数；只写 32 就让 64 个线程来碰 |
| 谁执行 | 角色按 `threadIdx.x / 32` 切，整个 warp 走同一条 barrier | warp 内按 lane 分角色 |
| 重叠从哪来 | 写完 SMEM → `barrier.arrive` → 再跑 **tail**（消费者不读 tail）。消费者 `barrier.sync` 看见数据 → `barrier.arrive` 交还缓冲 → 再跑 **work** | 把长 FMA 放在 arrive **之前** 当主曲线（那是关键路径，叠不上） |
| 网格 | **每个 SM 一个 block**，block = 64（1 个生产者 warp + 1 个消费者 warp）。不超额投放第二波 block 去填「停在 barrier 上的 warp」 | 一上来铺满 GPU 再声称 named barrier 没收益；那是别的 block 把空档补上了 |
| 读数 | 主看 `cta/named`。一边 tail 或 work 为 0、或活在 arrive 之前，贴 1× 都合法 | 为赢改成两个 kernel（那是 C-08）或改成 mbarrier |

**和 mbarrier 的差别（正文必须写清，不做成第二套实验）**

| | Named barrier | `cuda::barrier` / mbarrier |
|---|---|---|
| 住哪 | CTA 硬件槽，编号 0..15，不占 SMEM 对象 | SMEM 里的 barrier 对象，要先 `init` |
| 等待 | `sync` = 到达并等待；`arrive` = 到达后继续。没有单独的 phase token | `arrive` 交回 token，`wait(token)` 等这一相 |
| 计数 | 这条 barrier 上 **所有** 执行 sync 或 arrive 的线程 | 每次 `arrive` 的更新；TMA 还可以再加字节事务 |
| 本章 | 主测 | 决策表一行，回到 B-07 / B-08 |

CuTe 的对照句可以进正文：两个 warp 做生产者/消费者时，named barrier 的计数是 64，mbarrier 的到达计数可以是 32。出处见文献池。

**默认夹具（禁止为赢而改）**

| 项 | 默认 | 扫描 |
|---|---|---|
| `block` | 64 | 不扫 |
| 角色 | warp 0 生产，warp 1 消费 | — |
| `count` | 64 | — |
| barrier | full = 1，empty = 2 | — |
| `phases` | 1024 | 若 median < 0.05 ms，加大 phases，不解释比值 |
| `tail` | 512（生产者 arrive **之后** 的依赖链 FMA） | `sweep_tail` |
| `work` | 512（消费者 arrive **之后** 的依赖链 FMA） | `sweep` |
| `grid` | `SM 数 × 1` 个 block | — |
| runs / warmup | 7 / 2 | — |
| burn | 与 C-08 相同的循环依赖 FMA，防 DCE | — |

依赖链 FMA 是故意的：打不满 FMA 流水时，两个 warp 的重叠才看得出。把 burn 换成吞吐型独立 FMA，SM 很容易先到屋顶，比值会贴 1×，那不是本章要的形状。

**TL;DR 目标**

1. **Named barrier 是 CTA 上 16 个硬件槽。** `__syncthreads` 用的是 0 号，并且要求未退出线程都到。子集或分角色用 `barrier.sync` / `barrier.arrive` 加参与线程数，编号从 1 起。
2. **单边 arrive 才腾得出重叠。** 生产者写完 SMEM 就 `arrive`，不必等消费者；消费者 `sync` 之后才能读。计数把两边都算进去。
3. **能叠的是 arrive 之后的 tail 和 work。** 活写在 arrive 之前，两边仍在关键路径上，墙钟接近 `__syncthreads` 串行。
4. **短的那组先跑完，不自动缩短 kernel。** 两组活互不依赖时，kernel 墙钟仍由慢组决定，`cta/named` 应贴 1×。
5. **判停**：主看 `cta/named` median。1× 有效。禁止 ncu 附着墙钟。禁止用 0 号、禁止 warp 内分裂、禁止把 mbarrier 字节计数写进这条曲线。

**建议正文结构**

1. 问题：C-04 的 block barrier 等的是整个 CTA。两个 warp 角色不同，能不能只让该等的人等？  
2. 物理模型：16 个槽；`sync` 与 `arrive`；ASCII 画出「arrive 之后 tail ∥ 对方 work」。对照一行 mbarrier（不展开）。  
3. 参与集：计数、32 的倍数、整 warp 收敛、sm_70 以下 unaligned 退回 aligned、编号 0 与 `__syncthreads` 冲突。  
4. 决策表：全体会合 → C-04；warp 内 mask → C-01 / `__syncwarp`；拷贝完成 → B-07/B-08；跨 kernel → C-08；CTA 内分角色且有 arrive 后的活 → 本章。  
5. 实验怎么设计 + `--mode sweep`。  
6. 实测表 + plot（无 5090 则待测）。  
7. §7：CudaDMA / CUTLASS warp-specialized GEMM → 文献与 D-05；不写成必须 C-10 才合法。  
8. SOP / 误区。

**可行性（实验矩阵）**

| 配置 | 裁决 |
|---|---|
| `cta`：同一对 tail/work，中间两道 `__syncthreads`，空闲角色不烧 FMA | **必做（基线）** |
| `named`：arrive 之后才 burn；双 barrier（full / empty） | **必做** |
| `sweep`：固定 `tail`，扫 `work` | **必做（主曲线）** |
| `sweep_tail`：固定 `work`，扫 `tail` | **必做（重叠轴）** |
| `modes`：定点 + `named_pre`（burn 在 arrive 之前）+ `indep`（两组互不依赖，kernel 比应贴 1×）+ 正确性 | **必做** |
| 正确性：消费者读到的 SMEM 令牌与阶段号一致，host 核对 | **必做** |
| CC < 7.0 清晰退出 | **必做** |
| `cuda::barrier` 对照臂 | **不做** |
| 缺 barrier 的数据竞争演示 | **不做进计时**（UB / 可能挂死；误区里写，不设 mode） |
| 编号折回、`bar.red`、cluster、GEMM | **不做** |
| NCU / SASS | 可选；默认不加 profile 壳 |

**主命令**：`./bin/03_compute_primitives_09_named_barrier --mode sweep`

| mode | 要回答的问题 |
|---|---|
| `cta` | 两道 `__syncthreads` 把 tail 和 work 串起来有多长？ |
| `named` | arrive 之后两边一起烧，墙钟短多少？ |
| `sweep` | 扫消费者 `work`：比值抬起来还是贴 1？ |
| `sweep_tail` | 扫生产者 `tail`：没有尾巴还能赢吗？ |
| `modes` | 定点、`named_pre`、`indep`、正确性、CC / SM / 编号 |

`named_pre`：生产者的 tail、消费者的 work 都放在各自的 `arrive` 之前。预期相对 `cta` 贴 1×。  
`indep`：block = 256，前 4 个 warp 与后 4 个 warp 各用自己的 `barrier.sync`（编号 1 与 2，计数 128），工作量故意差很多（例如 2048 vs 32）。两组都不读对方的 SMEM。预期 kernel 级 `cta/named` 贴 1×。这组不进主 sweep。

实现约定：

- 启动打印 GPU / `sm_XX` / `SMs` / `grid`（= SM 数）/ `block` / `count` / barrier 编号。  
- CC < 70：退出码非零，写明门槛。  
- 计时：event 与 kernel **同 stream**。  
- 防 DCE：令牌进 SMEM，消费者累加后写 `sink`。  
- 阶段 0 生产者不等 empty；从阶段 1 起先 `barrier.sync` empty 再写。消费者每一阶段都先等 full。两边 `phases` 必须相同。  
- 内联 PTX 用 `barrier.sync` / `barrier.arrive`，约束 `"memory"`。不要在子集路径上用 `bar.sync`。  
- 默认不加 profile shell。

**本机要验证的文献形状**

1. `tail` 与 `work` 同量级：`named` 短于 `cta`（叠的是 arrive 之后的两段依赖链）。上限大约是 2× 这一档，**不设达标线**；5090 上低于 2× 也照实写。  
2. `tail→0` 或 `work→0`：`cta/named` 收到 ~1×。  
3. `named_pre`：收到 ~1×。  
4. `indep`：kernel 墙钟收到 ~1×。短组先结束这件事，若要写，只能用该组自己的 `clock64` 旁注，**不进主比值**。  
5. 正确性失败就先修参与集，不解释加速比。

文献里的 CudaDMA（Fermi，SC’11）是专用 DMA warp 的算子级结果：合成微基准最高约 **1.37×**，若干科学 kernel **1.15×–3.2×**。那不是「barrier 指令快了这么多」。本章不复现 CudaDMA，也不把这些倍数写成 5090 应达到的线。

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 分角色 + arrive 后有活 → 可试 named barrier | TL;DR / 主曲线 |
| 活在 arrive 前 / 某一侧为 0 → 别指望 | TL;DR③ / `named_pre` / `sweep` 两端 |
| 两组互不依赖 → kernel 墙钟仍看慢组 | TL;DR④ / `indep` |
| 全体会合 → C-04；拷贝完成 → B-07/B-08；跨 kernel → C-08 | 决策表 |
| CudaDMA、Weft、CUTLASS GEMM | 文献 / §7；GEMM → D |

**参考文献池**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A | [PTX ISA — `bar` / `barrier`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier) | 每 CTA **16** 个 barrier（0..15）。`b` 为参与线程数且为 warp 大小的倍数。`sync` 等待其他参与 warp；`arrive` 不等待。`bar.sync` = `barrier.sync.aligned`。sm_6x 及以下无 `.aligned` 的 `barrier` 等价于 aligned。不同 warp 可以对同一名字一个 `sync`、一个 `arrive`（生产者/消费者）。完成即复位，可立刻再用 | §2～3；夹具 |
| A | [CUDA PG — Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html) | 全体 block / 全体 warp 仍建议 `__syncthreads` / `__syncwarp`。`cuda::barrier` 负责拆开的 arrive/wait 和「哪些线程参加」。warp specialization 例是 SMEM barrier + 双缓冲，不是这 16 个槽 | 边界表；**不进主测** |
| A | [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/)（`__syncthreads` / `bar.sync` 与未退出线程） | 有未退出线程到不了 barrier 时，程序要改，不能靠「它不会执行到」蒙混 aligned barrier | 误区 |
| B | [CUTLASS `cutlass/arch/barrier.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h) | 16 个槽；**不要用 0**（`__syncthreads` 会用）；参与线程数为 warp 大小的倍数。`bar.sync` 与 `barrier.sync` 分成 aligned / unaligned 两条 | §3；SOP |
| B | [CuTe DSL `NamedBarrier`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html) | 两个 warp 对接时，named barrier 计数是两边之和（例：64），mbarrier 可以只计一侧（例：32）。NamedBarrier 没有 mbarrier 那种单独 `wait` | TL;DR②；与 mbarrier 对照表 |
| B | [CUTLASS issue #2672](https://github.com/NVIDIA/cutlass/issues/2672) | 编号超出 0..15 会折回；16 会碰到 0 号槽 | 误区；禁止 ≥16 |
| C | Bauer, Cook, Khailany, **CudaDMA**, SC’11（[IEEE](https://ieeexplore.ieee.org/document/6114413)） | Fermi 上专用 DMA warp：合成最高约 **1.37×**，科学 kernel **1.15×–3.2×**。收益是访存与计算拆开，不是 barrier 微基准 | §1 动机；**不设为本机达标线** |
| C | Sharma, Bauer, Aiken, **Weft**, PLDI’15（[ACM](https://dl.acm.org/doi/10.1145/2813885.2737962)） | named barrier 上要查的三件事：不死锁、槽能安全复用、SMEM 访问被 barrier 盖住 | 正确性；误区。不进加速比 |
| D | CUTLASS warp-specialized GEMM：角色同步用 NamedBarrier，TMA 完成用 mbarrier | 两条机制会同时出现 | §7 → D-05 |
| D | [SIGARCH / Colfax GEMM pipeline](https://www.sigarch.org/efficient-gemm-kernel-designs-with-pipelining/) | 多 stage 缓冲的 full/empty 在文中是 **mbarrier** | §7；防止和本章槽位写混 |

**交付 checklist**

- [x] 用户确认本大纲（named barrier 主线）
- [x] `examples/03_compute_primitives/09_named_barrier.cu`
- [x] 5090 `--mode sweep` → `docs/results/C-09_*` 数字 + speedup plot
- [x] 正文 + 封面/原理图（墙钟已写入；图只画机制）
- [x] 回填规划 / Module-C / README；C-07、C-08 文末改为指向本章
