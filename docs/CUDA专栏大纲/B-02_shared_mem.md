# B-02 写作大纲：Shared Memory（Bank / Padding / Swizzle）

> 状态：✅ **已收口**（5090：padding **14.12×** / swizzle **12.41×**）。  
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md) §4.2 · [Module-B.md](Module-B.md)

**路线**：**Microbench-first**——同一 32×32 tile **列访问**：`naive` / `padding` / `swizzle` + 相对 `naive` 加速比。  
**硬件门槛**：不限 sm_90+。  
**证据口径**：裸跑 CUDA event **median**（不用 `clock64` 当主结论）。

**标题**：`B-02. Shared Memory：Bank Conflict、Padding 与 XOR Swizzle`

**边界**

| 已有章节 | 本章 |
|---|---|
| B-01 | 不重讲 sector / 合并；承接「GMEM 形状修好后落地 SMEM」 |
| B-03 | 不讲 spilling / occupancy 处方 |
| B-07 / B-08 | **不做** `cp.async` / TMA 主测；swizzle **不是** TMA 前奏 |
| B-09 / D-05 | 不讲 AoS、不讲 `ldmatrix` / TC 全套；一句钩子 |

**TL;DR 目标**

1. Bank = 32 条 32-bit 通道；同 bank 不同地址 → 串行（n-way）；同地址 → 广播。
2. 行长 = 32 float 时，列访问 `tile[tid][col]` 是 32-way。
3. Padding `[32][33]`：行跨度与 32 互质，列访问错开 bank。
4. XOR swizzle：紧凑 `[32][32]`，`phys = col ^ row` 把同列映射到不同 bank。讲 bank，不讲 TC。
5. 主看相对 `naive` 加速比；clock64 只作旧示例对照，不当结论。

**MVP**

| 配置 | 裁决 |
|---|---|
| `naive` 列访问 `[32][32]` | 必做（基线） |
| `padding` `[32][33]` | 必做 |
| `swizzle` `col ^ row` | 必做 |
| NCU shared bank conflicts | 可选旁证 |
| TMA / ldmatrix / carveout | **不做** |

**主命令**：`./bin/02_memory_optim_02_shared_mem_bank_conflict --mode modes`

**参考文献池**

| 层 | 条目 | 用途 |
|---|---|---|
| A | [CUDA PG — Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | 32 bank；padding `+1` 官方处方 |
| A | [Best Practices — Shared Memory and Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 同时访问 n bank → n 倍带宽；冲突则拆请求 |
| B | [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) | tile / 冲突现场 |
| D | CUTLASS / `ldmatrix` swizzle | 仅钩子 → D-05 / B-08 |

**交付 checklist**

- [x] 大纲（边界 + modes）
- [x] 重写 `.cu`（`--mode` + event median）
- [x] 5090 `--mode modes` → `docs/results/B-02_*` + plot
- [x] 重写正文（去 TMA 开篇；去 csdnimg；CTA）
- [x] 数字回填后改规划 🟡→✅
