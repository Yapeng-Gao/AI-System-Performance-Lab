# C-01 Warp Primitives — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 正文：`article/03_compute_primitives/C-01*.md`

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CC | sm_120 |
| SMs | 170 |
| n | 16777216 |
| grid | 1360（auto = SMs×8） |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median** |

## 复现命令

```bash
./bin/03_compute_primitives_01_warp_primitives --mode sweep
./bin/03_compute_primitives_01_warp_primitives --mode modes
```

## Sweep（主曲线：`shfl/smem` vs `nwarps`）

verify float-reduce OK（sum=142606336）。

| nwarps | block | smem_ms | shfl_ms | shfl/smem |
|---:|---:|---:|---:|---:|
| 1 | 32 | 0.0283 | 0.0278 | **1.016** |
| 2 | 64 | 0.0215 | 0.0216 | **0.996** |
| 4 | 128 | 0.0209 | 0.0209 | **1.002** |
| 8 | 256 | 0.0248 | 0.0238 | **1.043** |
| 16 | 512 | 0.0240 | 0.0209 | **1.147** |
| 32 | 1024 | 0.0304 | 0.0239 | **1.273** |

### 怎么读

1. **nwarps≤4：加速比 ≈1**。两端都要 grid-stride 读 16M float，墙钟被 **GMEM** 吃掉，warp 内用 SMEM 树还是 shfl 几乎看不出。
2. **nwarps↑ → shfl 拉开**：16→**1.15×**，32→**1.27×**。更大 block 上 SMEM 树的 `__syncthreads` / 共享往返更显眼，shfl「每 warp 一份 SMEM」更省。
3. **不要拿本机 1.27× 去对齐** IPDPS’17 应用侧 1.2～2.1×——那是通信密集内核；本 microbench 诚实标出「访存墙下的原语差价」。
4. p10/p90 在个别点（如 smem@16）略散，结论仍看 **median 形状**：随 nwarps 单调抬升。

## Modes（定点，`nwarps=8`）

| tag | median_ms | 备注 |
|---|---:|---|
| smem | 0.0232 | float；verify OK |
| shfl | 0.0220 | float；vs smem **1.054×** |
| shfl_i | 0.0225 | int |
| redux | 0.0213 | `__reduce_add_sync`；vs shfl_i **1.055×** |
| ballot | 0.0252 | odd_count=8388608；verify OK |

### 怎么读

1. 定点与 sweep@8 同量级（~1.05×），重复性正常。  
2. **redux 略快于手写 shfl-int（~1.06×）**：有收益但不是数量级；float 仍只能走 shfl。  
3. ballot 正确性过；时延含 atomic 聚合写回，不与 reduce 加速比横比。

## NCU 旁证（RTX 5090 / sm_120，nwarps=32）

```bash
DO_NCU=1 bash examples/03_compute_primitives/01_profile_warp.sh ncu-only
```

> **禁止**把 ncu 附着时程序自打印的 median ms（数百～上千 ms）当结论——那是 replay 放大。主证据仍是裸跑 sweep。

### smem vs shfl（float）

| Metric | smem | shfl | shfl/smem |
|---|---:|---:|---:|
| `smsp__inst_executed.sum` | 9701664 | 5631184 | **0.58** |
| `l1tex__…wavefronts_mem_shared.sum` | 1412446 | 314780 | **0.22** |
| `sm__sass_inst_executed_op_shared_ld.sum` | 99280 | 1360 | **0.014** |
| `sm__sass_inst_executed_op_shared_st.sum` | 92480 | 43520 | **0.47** |
| `smsp__warps_issue_stalled_barrier.sum` | 75167830 | 48163861 | **0.64** |
| `smsp__warps_issue_stalled_long_scoreboard.sum` | 375421081 | 418482579 | 1.11 |

### shfl_int vs redux（int）

| Metric | shfl_int | redux | redux/shfl_i |
|---|---:|---:|---:|
| `smsp__inst_executed.sum` | 5757648 | 5790288 | **1.01** |
| `l1tex__…wavefronts_mem_shared.sum` | 314643 | 90991 | **0.29** |
| `shared_ld` / `shared_st` | 1360 / 43520 | 1360 / 43520 | 1.00 |
| barrier stall | 44993609 | 63077095 | 1.40 |
| long_scoreboard | 426923362 | 407143368 | 0.95 |

### 怎么读

1. **shfl vs smem**：指令约 **-42%**，shared wavefront 约 **-78%**，shared load 近乎砍光——与裸跑 nwarps=32 的 **1.27×** 同向；墙钟仍被 GMEM 压着，加速比不是数倍。
2. **barrier stall** shfl 更低（0.64×），符合少 `__syncthreads` 树。
3. **long_scoreboard** shfl 略高：更多时间暴露在等 GMEM，不是 shfl「更慢」。
4. **redux vs shfl_int**：整 kernel `inst_executed` ≈持平（规约尾巴相对 16M load 太小）；裸跑定点仍是 ~1.06×，勿用附着 ms 横比。

## 对大纲「文献形状」假设的校准

| 假设 | 本机 |
|---|---|
| shfl 相对 naive SMEM「显著更快」 | **有条件成立**：小 block ≈持平；大 block **1.15～1.27×** |
| shuffle 少走 SMEM | NCU：shared/inst 大降，支持 |
| int `__reduce_add_sync` ≤ shfl 树 | 裸跑 **~1.06×**；整 kernel NCU inst ≈持平 |

