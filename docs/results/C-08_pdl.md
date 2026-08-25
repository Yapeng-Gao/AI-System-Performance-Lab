# C-08 PDL — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 正文：`article/03_compute_primitives/C-08*.md`  
> 示例：`examples/03_compute_primitives/08_pdl.cu`

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CC | sm_120 |
| SMs | 170 |
| blocks_per_sm | **6** |
| half_grid / full_grid | **510** / **1020**（occ=1020，与 C-07 同口径） |
| n | 1048576 |
| work / tail 默认 | 512 / 512 |
| block | 256 |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median**；**同 stream 记 event** |

## 复现命令

```bash
./bin/03_compute_primitives_08_pdl --mode sweep
./bin/03_compute_primitives_08_pdl --mode sweep_tail
./bin/03_compute_primitives_08_pdl --mode modes
```

Windows：`.\bin\03_compute_primitives_08_pdl.exe --mode sweep`

## Sweep（主曲线：固定 tail=512，扫 work）

| work | serial_ms | pdl_ms | serial/pdl |
|---:|---:|---:|---:|
| 0 | 0.0205 | 0.0215 | **0.952×** |
| 1 | 0.0225 | 0.0215 | **1.048×** |
| 8 | 0.0225 | 0.0205 | **1.100×** |
| 64 | 0.0235 | 0.0225 | **1.044×** |
| 512 | 0.0338 | 0.0317 | **1.066×** |
| 4096 | 0.1260 | 0.1239 | **1.017×** |

![C-08：serial/pdl vs work](../../article/03_compute_primitives/assets/C-08-speedup-vs-work.png)

### 怎么读

1. **整条曲线贴 1×。** 峰值 1.10×（work=8），默认点 1.07×，work=4096 回到 **1.02×**。不是「work 越大越能叠」。
2. **work=0 时 PDL 更慢（0.95×）。** 没有独立 prologue，只剩 attribute / wait 税。
3. 多数点墙钟只有 **20～35µs**，p10–p90 已经有 2～3µs。0.95×～1.10× 落在这条缝里，不要读成第二套 GPU。
4. work=4096 时省下的 ~2µs 被 0.12 ms 的 body 淹没。固定 `tail=512` 叠不上更长的 K2。

## Sweep tail（副曲线：固定 work=512）

| tail | serial_ms | pdl_ms | serial/pdl |
|---:|---:|---:|---:|
| 0 | 0.0215 | 0.0236 | **0.913×** |
| 1 | 0.0215 | 0.0215 | **1.001×** |
| 8 | 0.0215 | 0.0215 | **1.000×** |
| 64 | 0.0215 | 0.0235 | **0.914×** |
| 512 | 0.0338 | 0.0297 | **1.138×** |
| 4096 | 0.1249 | 0.1137 | **1.099×** |

![C-08：serial/pdl vs tail](../../article/03_compute_primitives/assets/C-08-speedup-vs-tail.png)

### 怎么读

1. **没有尾巴就叠不上。** tail=0/1/8 是 0.91×～1.00×；tail=0 时 PDL 更慢。
2. 本机最大点在 **tail=512：1.14×**（0.0338→0.0297 ms，大约 4µs）。同配置在 `sweep` / `modes` 里是 1.07× / 1.13×，差 1µs 级，当同一档。
3. tail=4096：1.10×（0.125→0.114 ms）。尾巴变长，加速比没有继续抬。
4. 禁止为了「看见 2×」去加长 burn 或改成双 stream。

## Modes（定点 work=512，tail=512）

| tag | grid | median_ms | 备注 |
|---|---:|---:|---|
| serial | 510 | 0.0348 | 同流无 attribute |
| pdl | 510 | 0.0307 | Occupancy/2；`serial/pdl`=**1.134×** |
| pdl_full | 1020 | 0.0307 | `pdl_full/pdl`=**0.999×** |

满网格没有多赢，也没有系统性更慢。n 固定时 grid-stride 总工作量一样；本机省下的仍是 ~4µs 调度缝，不是「再来一块 SM」。

## 读数纪律

- 禁止 ncu 附着墙钟。
- 禁止为赢拆依赖或改成 Host 双 stream。
- **1× 有效。** 本夹具的结论就是 1× 量级，不是测废了。
- `pdl_full` 贴 `pdl` 说明占满 occupancy 叠不出更多计算。
