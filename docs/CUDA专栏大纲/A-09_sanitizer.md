# A-09 写作大纲：Compute Sanitizer 与异步错误归因

> 状态：✅ 正文+示例已落地。本机 RTX 5090 / `sm_120`：memcheck / racecheck / synccheck **三 PASS**（无 `docs/results/` CSV）。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)
>
> **已交付**：
> - 正文：`article/01_cuda_basic/A-09. 调试与错误诊断：Compute Sanitizer 实战与 CUDA 13 增强特性.md`（H1：Compute Sanitizer：异步错误怎么归因）
> - 示例：`examples/01_cuda_basics/09_debug_and_sanitizer.cu`（mode 0/1/2；initcheck 未做代码）
> - 脚本：`09_run_sanitizer.sh`
> - 图：`assets/A-09-sanitizer-cover.png` + `A-09-fig1-async-lag.png`
> - **本机要点**：OOB Invalid write；race Hazard；syncwarp Invalid arguments（半 Block syncthreads 本机曾 0 errors）
>
> **路线**：Module A 概念章。主证据 = Sanitizer 报告对上 planted bug。

**标题（H1）**：`A-09. Compute Sanitizer：异步错误怎么归因`

---

## MVP（已实现）

| 编号 | mode | tool | 本机 |
|---|---|---|---|
| A | 0 OOB | memcheck | ✅ PASS |
| B | 1 SMEM race | racecheck | ✅ PASS |
| C | 2 illegal syncwarp mask | synccheck | ✅ PASS |
| D | initcheck | — | 不做代码 |

---

## 交付 checklist

- [x] 用户确认大纲（D 可选 → 未做代码）
- [x] 重写 `09_debug_and_sanitizer.cu`（含 OOB/syncwarp 种因修正）
- [x] 校对 `09_run_sanitizer.sh` + README 第 9 章
- [x] 正文 + 本地图 + GitHub 绝对链
- [x] 规划表 A-09 → ✅；焦点 → A-10
- [x] 用户贴 sanitizer 输出后写入 TL;DR（三 PASS）
