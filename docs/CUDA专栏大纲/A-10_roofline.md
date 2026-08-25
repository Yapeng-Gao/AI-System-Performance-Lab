# A-10 写作大纲：Roofline 与 Speed-of-Light

> 状态：✅ 正文+示例已落地。本机 RTX 5090：`sm_120`，BW≈**1954 GB/s**，FP32≈**49 TFLOPS**，ridge≈**25 FLOP/byte**（无强制 CSV）。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)
>
> **已交付**：
> - 正文：`article/01_cuda_basic/A-10. 性能建模第一性原理：Roofline Model 与 Speed-of-Light 分析.md`
> - 示例：`examples/01_cuda_basics/10_roofline_demo.cu`（A/B/C；NCU 可选未跑）
> - 图：`assets/A-10-roofline-cover.png` + `A-10-fig1-roofline.png`
> - **本机要点**：copy memory 侧、FMA compute 侧；图 `A-10-measured-roofline.png`

**标题（H1）**：`A-10. Roofline：先分清带宽墙还是算力墙`

---

## MVP

| 编号 | 内容 | 本机 |
|---|---|---|
| A | float4 copy BW | ✅ 1953.56 GB/s |
| B | FMA FP32 | ✅ 49.01 TFLOPS |
| C | ridge + 相对位置 | ✅ 25.09 FLOP/byte |
| NCU | 可选 | 未跑 |

---

## 交付 checklist

- [x] 用户确认（A/B/C 必做，NCU 可选）
- [x] 重写示例 / 正文 / 导航
- [x] 用户贴本机输出后写入 TL;DR
