# A-08 写作大纲：Stream / Event / 三级流水线

> 状态：✅ 正文+示例已落地。本机 RTX 5090：`sm_120`、`asyncEngineCount=2`；A/B≈**8.40×**、C/B≈**1.30×**（无 `docs/results/` CSV）。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)
>
> **已交付**：
> - 正文：`article/01_cuda_basic/A-08. 异步执行模型：Stream, Event 与流水线并发.md`（H1：Stream 与 Event：让拷贝和计算重叠）
> - 示例：`examples/01_cuda_basics/08_async_pipeline.cu`（A/B/C 必做；D WaitEvent 仅正文）
> - 图：`assets/A-08-async-stream-cover.png` + `A-08-fig1-issue-order.png` + `A-08-fig2-pipeline.png`
> - **本机要点**：A 11.825 ms / B 1.408 ms / C 1.835 ms；图 `A-08-mode-median-bars.png`
> - NSYS：`08_profile_nsys.sh` 可选旁证，不进 TL;DR
>
> **路线**：Module A 概念章。主证据 = serial vs depth-first vs breadth-first 的 **CUDA event median 形状**；不是 pinned GB/s（B-06）。
>
> **硬件门槛**：不限 sm_90+；打印 `asyncEngineCount`。

**标题（H1）**：`A-08. Stream 与 Event：让拷贝和计算重叠`

---

## 与前后章的边界

（落地后仍有效；详见正文 §1。）

| 已有章节 | A-08 深化 / 禁止 |
|---|---|
| A-07 | 承接空间课；mapped ≠ overlap 前置 |
| B-06 | Pinned 硬前置一句；**不扫** size / GB/s / CE |
| B-07 / B-08 | Host CE ≠ 设备内 async |
| C-06 | Event Wait 机制课；不 Graph |
| A-09 | 钩子：Sanitizer |

---

## TL;DR

见正文。本机：`sm_120` / `asyncEngineCount=2` / A/B **8.40×** / C/B **1.30×**。

---

## MVP（已实现）

| 编号 | 配置 | 状态 |
|---|---|---|
| A | pageable + default stream | ✅ |
| B | pinned + NonBlocking + depth-first | ✅ |
| C | breadth-first 对照 | ✅ |
| D | StreamWaitEvent demo | 正文 ASCII；无独立代码 |

计时：event + device sync 后再 record stop（NonBlocking 安全）。

**主命令**：

```bash
cmake --build . --parallel --target 01_cuda_basics_08_async_pipeline
./bin/01_cuda_basics_08_async_pipeline
```

---

## 参考文献池

见正文 §10（与大纲阶段池对齐）。

---

## 交付 checklist

- [x] 用户确认大纲（C 必做）
- [x] 重写 `08_async_pipeline.cu`
- [x] 改 `examples/01_cuda_basics/README.md` 第 8 章
- [x] 正文 + 本地图（无 `csdnimg`）+ GitHub 绝对链
- [x] 规划表 A-08 → ✅；焦点 → A-09
- [x] 用户贴本机输出后写入 TL;DR（5090：A/B 8.40×、C/B 1.30×）
