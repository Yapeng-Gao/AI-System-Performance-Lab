# Global Memory / Pipeline / Attention Benchmark Comparison

## 1. HBM Saturation (Pointer Chasing)

| Arch       | BW / Latency 特征 | SASS 关键指令 | 结论 |
|------------|------------------|---------------|------|
| Ampere     | HBM-bound        | LDG.E         | 真实 global 下限 |
| Hopper     | HBM-bound        | LDG.E + DEPBAR| 几乎无提升 |
| Blackwell  | HBM-bound        | LDG.E         | 不可隐藏 |

> 说明：强依赖 pointer chasing 彻底击穿 L2 / prefetch。

---

## 2. Async Pipeline (cp.async)

| Arch       | 性能表现 | SASS 证据 | 结论 |
|------------|----------|-----------|------|
| Ampere     | ↑ 明显   | CP.ASYNC  | pipeline 开始有用 |
| Hopper     | ↑↑       | CP.ASYNC + DEPBAR | latency hiding 强 |
| Blackwell  | ↑↑↑      | LDGSTS / TMA | 架构级优势 |

---

## 3. Attention / GEMM-style Kernel

| Arch       | L2 Reuse | Compute Overlap | 结论 |
|------------|----------|-----------------|------|
| Ampere     | 有限     | 一般            | baseline |
| Hopper     | 强       | 明显            | AI 优化开始显现 |
| Blackwell  | 极强     | TMA + 大 L2     | AI 代际差距来源 |

---

## 总结

- memcpy / streaming：架构差异被隐藏
- pointer chasing：架构无能为力
- tiled + pipeline：Blackwell 真正拉开差距
