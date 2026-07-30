# Module D 大纲：计算原语与高级算子实现（31–40）

> 状态：⏳ 规划中。无代码目录。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

## 模块目标

从「会写 kernel」走向「能写出接近库级质量的 kernel」，形成算子实现范式。

## 落地策略

1. 先做 **Reduction / Softmax / LayerNorm**（易评测、易验证正确性）。
2. 再做 **GEMM / Tensor Core**（矩阵分块、布局、与 B-08/B-09 布局钩子汇合）。
3. Attention / FA 类生产叠法作扩展或后置章，避免过早抢 Module E。

## 建议主题骨架

| 阶段 | 主题 | 备注 |
|---|---|---|
| 通用算子 | Reduce / Softmax / LayerNorm / RMSNorm | 数值稳定性 + 带宽/算力对照 |
| 矩阵 | Naive GEMM → tiling → Tensor Core | CuTe/CUTLASS 可作对照，不必复刻生产库 |
| 进阶 | 融合算子 / epilogue | 与 C 的 fusion、B 的布局交叉 |

开写第一章时再拆独立 `D-0N_*.md` 大纲文件。
