# C-03 Atomics 与 Contention — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 正文：`article/03_compute_primitives/C-03*.md`

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CC | sm_120 |
| SMs | 170 |
| n | 16777216 |
| block | 256 |
| grid | 1360（auto = SMs×8） |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median** |

## 复现命令

```bash
./bin/03_compute_primitives_03_atomics_contention --mode sweep
./bin/03_compute_primitives_03_atomics_contention --mode modes
```

> 工作负载：`in[i]=i%1000`，`thresh=round(hit_rate*1000)`，命中则对**同一全局计数器** +1。

## Sweep（主曲线：agg/naive、smem/naive vs hit_rate）

verify naive OK（各档 hits 与期望一致）。

| hit_rate | naive_ms | smem_ms | agg_ms | agg/naive | smem/naive |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0379 | 0.0263 | 0.0358 | **1.060** | **1.443** |
| 0.125 | 0.0521 | 0.0267 | 0.0514 | **1.015** | **1.953** |
| 0.25 | 0.0790 | 0.0278 | 0.0790 | **1.000** | **2.839** |
| 0.5 | 0.1321 | 0.0308 | 0.1330 | **0.993** | **4.282** |
| 1.0 | 0.2335 | 0.0370 | 0.2335 | **1.000** | **6.312** |

### 怎么读

1. **手写 warp-agg ≈ naive（全程 ~1.0×）**：同 warp、同地址、相同 +1 的 `atomicAdd`，5090/现代工具链上 **NVCC/硬件很可能已自动聚合**；手写 `coalesced_threads` 几乎不再加分。这是大纲 TL;DR④ 的本机实证，不是实现 bug（verify 全过）。
2. **SMEM staging 才是主曲线英雄**：`smem/naive` 随 hit_rate **单调抬升**——0.05→1.44×，1.0→**6.31×**。争用越重，把原子压进 block SMEM、每 block 一次 global 越值。
3. **与 Kepler blog 形状相反**：Pro Tip 时代「聚合 global ≫ SMEM」；本机是 **SMEM staging ≫（已等价的）naive/agg global**。正文必须写清「以本机为准，勿硬套 20×」。
4. naive 墙钟随 hit_rate 近似线性涨（0.038→0.234 ms）；smem 几乎平坦（0.026→0.037）——争用被挡在 SM 内。

![C-03：加速比随 hit_rate](../../article/03_compute_primitives/assets/C-03-speedup-vs-hitrate.png)

## Modes（定点，hit_rate=1.0）

| tag | median_ms | 相对 naive |
|---|---:|---|
| naive | 0.2319 | 基线 |
| smem | 0.0354 | **6.557×** |
| agg | 0.2310 | **1.004×** |
| agg_smem | 0.0354 | **6.546×** |

### 怎么读

1. 与 sweep@1.0 同量级，重复性正常。
2. **`agg_smem ≈ smem`**：在已有 block staging 时，再套 coalesced 聚合几乎无额外收益（SMEM atomic 路径已够便宜）。
3. 工程处方（本机）：同址高争用计数 → **优先 SMEM staging**；手写 warp-agg 作可读性/可移植写法，别预期数量级加速。

## 对大纲「文献形状」假设的校准

| 假设 | 本机 |
|---|---|
| 高争用：手写 agg ≫ naive | **不成立（≈1.0×）**——自动聚合/硬件掩盖手写差价 |
| 低争用：agg 收益收窄 | agg 全程无收益；**smem** 收益随争用**升高**（非收窄） |
| smem staging 不总赢过 agg+global | **本机 smem 全面赢**（至 ~6.5×）；与 Kepler「聚合 global 最优」叙事相反 |

## 写作要点（给正文）

- TL;DR 主数字改成：**smem/naive @hit=1 → ~6.3×**；**agg/naive ≈ 1.0×**。
- 决策表：5090 同址计数默认 **smem staging**；warp-agg 保留为安全写法与旧架构/编译器差异的保险。
- ARC / Kepler 20× 只作形状/动机，不替代上表。
