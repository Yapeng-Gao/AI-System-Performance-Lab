# Module A 大纲：映射与度量（1–10）

> 状态：✅ 已收束。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §2 全景 · §4.1

## 模块目标

逻辑并行模型 → 硬件执行实体的物理映射；养成可复现性能分析习惯（SASS / NSYS / NCU / Roofline）。

## 教法弧（锁定）

```text
模型 A-01 → 硬件 A-02 → 映射 A-03 → SIMT A-04 → ABI/SASS A-05
  → 工具链 A-06 → 空间 A-07 → Host 异步 A-08 → 正确性 A-09 → Roofline A-10
```

A-10 是度量语言终点，**不是** Checklist。访存排障出口在 B-10。

## 章节总表（1–10 ↔ A-01～A-10）

| 总序 | 章 | 主题 | 状态 | 大纲 |
|---|---|---|---|---|
| 0 | 导读 | 怎么读、怎么跑、本机数字 | ✅ | — |
| 1 | A-01 | CUDA 核心概念总览 | ✅ | — |
| 2 | A-02 | GPU 硬件架构 | ✅ | — |
| 3 | A-03 | Grid / Block / Warp 物理映射 | ✅ | — |
| 4 | A-04 | SIMT / Divergence / Replay | ✅ | — |
| 5 | A-05 | Kernel 结构与 ABI / SASS | ✅ | — |
| 6 | A-06 | NVCC / NVRTC | ✅ | — |
| 7 | A-07 | 内存空间 / UVA | ✅ | [A-07](A-07_memory_spaces.md) |
| 8 | A-08 | Host Stream / Event / 流水线 | ✅ | [A-08](A-08_async_stream.md) |
| 9 | A-09 | Compute Sanitizer | ✅ | [A-09](A-09_sanitizer.md) |
| 10 | A-10 | Roofline / SOL | ✅ | [A-10](A-10_roofline.md) |

路径：`article/01_cuda_basic/` · `examples/01_cuda_basics/`。已落地章以正文 + `docs/results/` 为准；A-01～A-06 不补空大纲。

## 硬边界

| 对象 | Module A 原则 |
|---|---|
| B-01 | 不讲合并 / sector 处方；空间章只点名 |
| B-02 / B-03 | 不讲 bank / spilling 处方 |
| B-07 / B-08 | 不讲设备 `cp.async` / TMA |
| A-08 vs C | Host CE overlap 在 A-08；设备侧重叠候选在 C-08 |

## 开写交接

已收束。下一 Module 见规划 §4.2（B 教法弧）。
