# B-03 写作大纲：寄存器（Spilling / Occupancy / launch_bounds）

> 状态：✅ **已收口**（5090：`highreg` **0.308×**；regs/occ 三档相同，墙钟跟 `localB` 128→1024 走）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §4.2 · [Module-B.md](Module-B.md)

**路线**：**Microbench-first**——同一压力核三档：`baseline` / `highreg` / `launch_bounds` + 相对 `baseline` 加速比；运行时打印 `numRegs` / `localSizeBytes` / occupancy。  
**硬件门槛**：不限 sm_90+。  
**证据口径**：裸跑 CUDA event **median**（不用 ptxas 日志当墙钟结论）。

**标题**：`B-03. 寄存器：Spilling、Occupancy 与 launch_bounds`

**边界**

| 已有章节 | 本章 |
|---|---|
| B-02 | 不重讲 bank；承接「padding 多占的那一列可能变成 occupancy 账单」 |
| B-01 / B-09 | 不讲合并 / AoS |
| B-04 | **不做** L2 residency 主测 |
| B-07 / B-08 | **不做** `cp.async` / TMA；不把 RF 写成「为 Tensor Core 铺路」 |
| D-05 | ILP / 累加器寄存器只留钩子 |

**TL;DR 目标**

1. RF 是 SM 上固定的 32-bit 池；每线程多拿寄存器，能同时驻留的 warp 就少。
2. Spill = 活变量进 **Local Memory**（逻辑私有，物理走 GMEM 层级，最坏到 HBM）。
3. Occupancy 不是 KPI：藏延迟可以用更多 warp，也可以用更高 ILP；低 occupancy 可以更快（Volkov）。
4. `__launch_bounds__` 是驻留契约：为了塞下 `minBlocks`，编译器可能减寄存器、加 spill。
5. 主看 `highreg` / `launch_bounds` 相对 `baseline` 的墙钟 + `localSizeBytes`；不要只看 occupancy 百分比。

**MVP**

| 配置 | 裁决 |
|---|---|
| `baseline`（REGS=32） | 必做（低压力对照） |
| `highreg`（REGS=256） | 必做（推 spill / 掉 occupancy） |
| `launch_bounds`（REGS=256 + min 2 blocks/SM） | 必做（契约对照） |
| 运行时 `cudaFuncGetAttributes` + occupancy API | 必做（regs / local / occ） |
| `-Xptxas=-v` | 可选旁证 |
| L2 lock / TMA / SMEM-spill 新路径 | **不做** |

**主命令**：`./bin/02_memory_optim_03_register_spill --mode modes`

**参考文献池**

| 层 | 条目 | 用途 |
|---|---|---|
| A | [CUDA PG — Local Memory / Occupancy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | local 在 GMEM 空间；spill；occupancy 定义 |
| A | [CUDA PG — `__launch_bounds__`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) | 超上限则减寄存器、加 local / 指令 |
| A | [Best Practices — Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) | 算 occupancy；regs 与驻留互挤 |
| A | [Runtime — Occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html) | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| B | Volkov, *Better Performance at Lower Occupancy*（GTC 2010） | 低 occupancy + 高 ILP 可以更快 |
| C | Jia et al., [arXiv:1804.06826](https://arxiv.org/abs/1804.06826) | RF / 延迟对照；本机仍以 median 为准 |
| D | RegDem [arXiv:1907.02894](https://arxiv.org/abs/1907.02894) | 溢到 SMEM 的研究路径；不进 MVP |

**交付 checklist**

- [x] 大纲（边界 + modes）
- [x] 重写 `.cu`（`--mode` + event median + attrs）
- [x] 重写正文（文件名与 H1 对齐；CTA）
- [x] 5090 `--mode modes` → `docs/results/B-03_*` + plot
- [x] 数字回填后改规划 🟡→✅
