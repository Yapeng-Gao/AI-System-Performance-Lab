# C-04 同步分层 — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 正文：`article/03_compute_primitives/C-04*.md`  
> 示例：`examples/03_compute_primitives/04_sync_layers.cu`

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CC | sm_120 |
| SMs | 170 |
| CooperativeLaunch | yes |
| coop_max_grid | ≈1020（blockDim=256 → 约 6 blocks/SM） |
| iters | 256 |
| nwarps（定点） | 8 |
| n（phases） | 1048576 |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median** |

## 复现命令

```bash
./bin/03_compute_primitives_04_sync_layers --mode sweep
./bin/03_compute_primitives_04_sync_layers --mode sweep_grid
./bin/03_compute_primitives_04_sync_layers --mode modes
```

## Sweep（主曲线：block/warp vs nwarps）

nblocks=170（=SMs）。空同步 ×256。

| nwarps | warp_ms | block_ms | block/warp |
|---:|---:|---:|---:|
| 1 | 0.0156 | 0.0124 | **0.799** |
| 2 | 0.0122 | 0.0133 | **1.097** |
| 4 | 0.0116 | 0.0133 | **1.152** |
| 8 | 0.0116 | 0.0135 | **1.160** |
| 16 | 0.0133 | 0.0161 | **1.209** |
| 32 | 0.0150 | 0.0185 | **1.230** |

![C-04：block/warp vs nwarps](../../article/03_compute_primitives/assets/C-04-ratio-vs-nwarps.png)

### 怎么读

1. **block/warp 只有约 1.1×～1.23×**（nwarps≥2）：5090 上空同步微基准里，warp 与 block barrier **并未拉开数量级**；主叙事不要写成「warp ≪ block」。
2. **nwarps=1 的 0.80×** 落在 ~0.01 ms 量级，受事件计时噪声/启动底噪影响大，**不采信为「block 比 warp 快」**。
3. 随 nwarps 增加，ratio 略抬升（1.10→1.23），与「更多 warp 等会合」方向一致，但幅度有限。

## Sweep grid（主曲线：grid sync vs nblocks）

nwarps=8，空 `this_grid().sync` ×256。

| nblocks | ≈blocks/SM | grid_ms | vs nblocks=1 |
|---:|---:|---:|---:|
| 1 | ~0 | 0.1593 | 1.00× |
| 170 | 1 | 0.1695 | 1.06× |
| 340 | 2 | 0.1840 | 1.16× |
| 1020（coop_max） | 6 | **0.6318** | **3.97×** |

![C-04：grid sync vs nblocks](../../article/03_compute_primitives/assets/C-04-grid-vs-nblocks.png)

### 怎么读

1. **1→340 几乎平坦**（0.16→0.18 ms）；**塞满 coop_max（6/SM）后跳到 0.63 ms**——对齐文献「grid 延迟更相关 blocks/SM」。
2. 相对同配置 block 空同步（modes ~0.01 ms），grid 仍贵一个数量级以上（见下）。

## Modes / phases（定点，nwarps=8，nblocks=170）

`correctness`：with `__syncthreads` **OK**。

| tag | median_ms | 备注 |
|---|---:|---|
| warp | 0.0089 | |
| block | 0.0098 | block/warp = **1.105×** |
| grid | 0.1686 | grid/block = **17.22×** |
| phases_grid | 0.0197 | verify OK |
| phases_two_k | ~0.02 | verify OK（终端行曾截断在 `0.02`；与 grid 同量级） |
| phases ratio | **≈1.0** | grid/two_kernel；不自动更快 |

### 怎么读

1. **本章主数字 = grid/block ≈ 17×**（空同步、同 nwarps/nblocks）：跨 SM 会合才是层间「断层」。
2. **phases ≈ 1.0**：同载荷下 grid 单核与两 kernel 墙钟持平量级；选型理由是状态复用，不是加速。
3. modes 与 sweep@8 的绝对 ms 略有差别（0.009 vs 0.012），同属短 kernel 波动；**主看相对比**。
4. **`sweep`（block/warp）为次级**：本机仅 ~1.1×～1.23×；主曲线看 `sweep_grid` + modes 的 `grid/block`。

## CSV

- `docs/results/C-04_sweep.csv`
- `docs/results/C-04_sweep_grid.csv`

```bash
python scripts/plot_c04_sync_layers.py
```
