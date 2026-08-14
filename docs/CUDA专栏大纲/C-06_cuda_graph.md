# C-06 写作大纲：CUDA Graph 与 launch overhead

> 状态：✅ 文章+示例+RTX 5090 实测已落地。对照审稿用；以正文与 `docs/results/C-06_*` 为准。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) · 模块：[`Module-C.md`](Module-C.md) · 上章：[`C-05_kernel_fusion.md`](C-05_kernel_fusion.md)
>
> **裁决（已确认）**：host CPU chrono（大纲 G）= **可选旁证**；主证据 = GPU 端到端 median + instantiate 定点一行。
>
> **本机要点**：短核 `stream/graph`≈**3.7～4.1×**（@nodes≥2）；work=4096→**1.01×**；instantiate modes **0.24 ms**。
>
> **路线**：Microbench-first——主对照 **短核链：逐次 `<<<>>>` launch** vs **stream capture → instantiate → `cudaGraphLaunch` 回放**；主曲线扫 **节点数 `n_nodes`**；副线扫 **单核工作量**（核变长则 Graph 收益收窄）。
>
> **硬件门槛**：**不限 sm_90+**。CUDA Graph 自 CUDA 10 起常用；本机以 5090 / 当前驱动为准校准绝对 µs。
>
> **目录**：已有 `article/03_*` / `examples/03_*`；本章只增 `06_cuda_graph.cu` 等，**勿再建空壳**。  
> **读者用语**：正文 § 称「实验怎么设计」，勿写「MVP」。

**标题（拟定）**：`C-06. CUDA Graph 与 Launch Overhead：短核墙、capture-replay 与何时上图`

**与前后章的边界**

| 已有 / 规划章节 | 已覆盖 | C-06 应深化 / 避免重复 |
|---|---|---|
| **C-05** | 垂直融合省 GMEM；省 launch 只作次要一句 | **不重测** elementwise 融合曲线；正文钉死：**Graph = launch/提交层融合**，≠ kernel body 融合 |
| **C-04** | grid sync vs 两 kernel（`phases`≈1） | **不重测** device 同步分层；多核序列只作 launch 对照载体 |
| A-08 | Host Stream / CE overlap | **不重讲** stream 编程课；Graph 可 launch 进 stream，一句挂钩即可 |
| **C-07**（候选） | Persistent kernel | 不做 persistent；§7 钩子「另一种摊销 launch」 |
| **C-08**（候选） | 多 kernel 重叠 | 不做多 stream 重叠全家桶；可一句「图内依赖 ≠ 自动重叠」 |
| Module E | torch.compile / `reduce-overhead` / CUDAGraph Trees | **不做** PyTorch capture 正文；§7 钩子 |
| Module D | 算子实现 | 不做 |

**TL;DR 目标结论（写作时先写死；有实测后改成带本机数字）**

1. **短核会被 launch 税吃掉**：单次 kernel launch 有 CPU/驱动侧固定开销（论坛口径常约数 µs；框架/Python 路径更贵）；核本身只跑数 µs～数十 µs 时，串一串 launch 会抬高端到端墙钟（NVIDIA Getting Started with Graphs 形状）。
2. **Graph 把「多次提交」收成一次 `cudaGraphLaunch`**：定义与 instantiate 付一次成本，重复 replay 摊薄；短核链 + 高重复次数时收益最大。
3. **核变长 / 节点太少 → 收益收窄**：body 主导端到端时 Graph 几乎无感（论坛共识）；本机用 `work` 扫确认「何时别上图」。
4. **Capture 有纪律**：topology/参数在 replay 间需稳定；动态参数用 update / 重建（扩展阅读），本章主路径只做 **静态短核链 capture**。
5. **判停**：裸跑 event **median** 看 `graph/stream` 随 `n_nodes`（及 `work`）形状；instantiate 成本单独报一次，勿摊进热路径；**禁止**把 ncu 附着墙钟当结论。

**建议正文结构（8～9 节）**

1. **问题**：C-05 说少 launch 是次要收益——本章单独测清 launch 墙与 Graph。
2. **物理模型（短）**：stream 逐次提交 vs 图一次提交；ASCII：CPU launch 间隙 vs GPU 短核；对照「body fusion vs launch fusion」。
3. **API 清单**：stream capture（`Begin/EndCapture`）→ `Instantiate` → `Launch`；与显式 `cudaGraphAddKernelNode` 一句对照（主路径用 capture）。
4. **决策表**：短核高重复 / 长核 / 拓扑常变 / 该退回 stream / 框架侧 → E。
5. **实验怎么设计** + 主命令 `--mode sweep`。
6. **实测**：表 + 曲线；口径 median。
7. **旁证（可选）**：NSYS 看 CPU launch 轨迹；或 host chrono 定点。
8. **扩展阅读**：`cudaGraphExecUpdate`；constant-time launch（CUDA 12.6+ blog）；PDL；torch CUDA Graph → E；钩子 → C-07/C-10。
9. **误区 + SOP + 下一章钩子**。

**写作路线**

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Microbench-first**：空/短核链，stream 循环 launch vs Graph replay；扫 `n_nodes` | 与 C-01～C-05 同构；直接回答「何时上图」 |
| 2 | 教程复述 Getting Started with Graphs | 易与官方重复；**不推荐**作主线 |
| 3 | PyTorch / LLM 推理 Graph | 抢 Module E；**仅扩展阅读** |

**可行性评估（实验矩阵）**

| 编号 | 配置 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | Stream：循环 launch `n_nodes` 个短核（同 stream）+ 最终 sync | ✅ | **必做（基线）** |
| B | Graph：capture 同序列 → instantiate → 循环 `cudaGraphLaunch` | ✅ | **必做（主对照）** |
| C | `sweep`：扫 `n_nodes∈{1,2,4,8,16,32,64}`（固定极短核）上 A/B | ✅ | **必做（主曲线）** |
| D | `sweep_work` 或 modes 点：固定 `n_nodes`，扫单核 `work`（空转/FMA iters） | ✅ | **必做（收益收窄轴）** |
| E | 定点：报告 **instantiate 一次** 成本（host 或单独 event 外墙钟）；热路径不含 instantiate | ✅ | **定点必做** |
| F | 正确性：短核写累加/pattern，stream 与 graph 结果一致 | ✅ | **必做** |
| G | 可选：host chrono 测单次 `cudaLaunchKernel` vs `cudaGraphLaunch` 返回时延 | ⚪ | **可选旁证**（与 GPU e2e 分开报） |
| — | torch.compile Graph；多 stream 复杂 DAG 主线；`ExecUpdate` 全家桶；PDL 深挖 | ❌ | **不做**（→ E / C-08 / 扩展） |

**最小可复现实验（`06_cuda_graph.cu`）**

| mode | 要回答的问题 | 进主结论？ |
|---|---|---|
| `stream` | 短核链逐次 launch 端到端？ | 基线 |
| `graph` | 同链 Graph replay 端到端？ | 对照 |
| `sweep` | 扫 `n_nodes`：`stream/graph` 形状？ | **主曲线** |
| `sweep_work` | 扫 `work`：收益是否收窄？ | **副曲线** |
| `modes` | 定点全表 + instantiate 成本一行 | 写结果用 |

实现约定：

- **主证据**：CUDA event **median**，量 **整条链端到端**（含多次 launch 的 GPU 可见完成）；warmup 后计时；Graph 路径：**instantiate 在计时外**，热循环只 `cudaGraphLaunch` + 必要时一次 sync。
- 短核：默认极轻（如空核或极少 FMA），保证 launch 税可见；`work` 扫时再加重。
- Capture：`cudaStreamBeginCapture` / `EndCapture`；失败清晰退出。
- 防 DCE：核写 `out` 或 atomic sink。
- 启动打印 GPU / `sm_XX`。
- 默认 **不加** profile shell；用户要 NSYS 再补。
- 正文必须写清：本机 µs/加速比是 **短核微基准形状**，不直接对齐 V100 blog 的 3.8→3.4 µs 绝对数；也不替代框架 Graph 数字。

**本机要验证的「文献形状」假设（1～3）**

1. **极短核 + 多节点**：`graph` 相对 `stream` **明显更快**，加速比随 `n_nodes` 抬升（或端到端差距拉开）。
2. **增大 `work`**：加速比收窄 →1（launch 税占比下降）。
3. **instantiate 一次性成本** 明显高于单次 `cudaGraphLaunch`（数量级差）；热路径必须排除它。

**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA PG — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) | 定义与执行分离；摊销 setup；`Instantiate`/`Launch` | §2～3；实验 |
| A 官方 | Runtime：`cudaStreamBeginCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch` | capture 与回放 API | 实验 |
| B 工程 | NVIDIA, [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) | 20 短核示例；有效时间/核从 ~3.8µs→~3.4µs；instantiate ~400µs 量级需重复摊薄 | TL;DR①②；假设①③ |
| B 工程 | NVIDIA, [Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)（12.6） | 直线图 repeat launch ≈ **2.5µs + ~1ns/node**（测例平台） | §2 旁证；勿当本机绝对数 |
| B 工程 | NVIDIA 论坛：launch ~3–5µs；ms 级核上 Graph 几乎无关 | 决策表「核够长别上图」 | TL;DR③ |
| B 工程 | [Employing CUDA Graphs in a Dynamic Environment](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/) | `ExecUpdate`；动态参数 | §7 |
| C 实证 | C-05：融合省流量为主，launch 次要 | 承接分层 | §1 |
| C 实证 | C-04：`phases`≈1 | 「少同步/少核」不自动更快同族 | §1 |
| C 实证 | Zhang et al. arXiv:2004.05371（launch / sync 微基准旁证） | launch 与 grid sync 成本对照 | §2 可选 |
| D 前沿 | Programmatic Dependent Launch（sm_90+） | 跨核依赖另一条路 | §7 |
| D 前沿 | torch.compile `reduce-overhead` / CUDAGraph Trees | 框架自动 Graph | §7 → **Module E** |

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 短核多节点 → Graph replay | TL;DR / 主曲线 |
| 长核 / 少节点 → 别上图 | TL;DR③ / `sweep_work` |
| instantiate 一次性、热路径排除 | TL;DR⑤ / 误区 |
| ExecUpdate / 动态图 / torch Graph | **仅** §扩展 → E |
| Persistent / 多 stream 重叠 | **仅** → C-07 / C-08 |

**交付进度**

- [x] 用户确认本大纲（G=可选旁证）
- [x] `examples/03_compute_primitives/06_cuda_graph.cu`
- [x] 本机实测 + `docs/results/C-06_*` 填数 + plot PNG
- [x] 正文 + 封面/原理图 + plot 脚本
- [x] 回填规划总表 / README / examples README
