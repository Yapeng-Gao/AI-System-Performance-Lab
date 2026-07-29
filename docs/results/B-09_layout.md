# B-09 数据布局（AoS/SoA/Transpose）— 参考结果

> 口径：裸跑 CUDA event **median**；带宽为 **useful payload**（触达字段或整矩阵的 R+W）。  
> useful GB/s 可因 L2 命中高于 DRAM 总线标称带宽——**主看加速比与相对形状**，勿当总线利用率。

## 平台

- GPU：NVIDIA GeForce RTX 5090，`sm_120`
- 载荷：`n=4194304` particles，`dim=4096`，`touch_fields` 见 sweep，`block=256`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_09_layout_transform`
- AoS：`sizeof(Particle)=32 B`（8×float）

## Touch-fields sweep（主结论）

```bash
./bin/02_memory_optim_09_layout_transform --mode sweep
```

CSV：[`B-09_sweep.csv`](B-09_sweep.csv)

| touch_fields | aos_ms | soa_ms | aos_gbps | soa_gbps | speedup_soa |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.177024 | 0.013024 | 189.55 | 2576.35 | **13.59** |
| 2 | 0.174656 | 0.016384 | 384.23 | 4096.00 | **10.66** |
| 4 | 0.169856 | 0.027264 | 790.19 | 4922.89 | **6.23** |
| 8 | 0.322688 | 0.178784 | 831.87 | 1501.45 | **1.80** |

## Modes 全表（layout touch=1 + transpose）

```bash
./bin/02_memory_optim_09_layout_transform --mode modes
```

CSV：[`B-09_modes.csv`](B-09_modes.csv)。正确性检查：copy / transpose_* 均为 OK。

| mode | median_ms | useful_gbps | 相对 copy |
|---|---:|---:|---:|
| aos | 0.176736 | 189.86 | — |
| soa | 0.013152 | 2551.28 | — |
| copy | 0.068576 | 1957.21 | 1.00× |
| transpose_naive | 0.174656 | 768.47 | **0.39×** |
| transpose_tiled | 0.075456 | 1778.75 | **0.91×** |
| transpose_pad | 0.074432 | 1803.23 | **0.92×** |

## 怎么读（本机形状）

1. `touch_fields=1`：SoA **~13.6×** 快于 AoS（少字段热读红利最大）。  
2. `touch_fields→8`：加速比 **13.6 → 1.80** 收窄，但仍 SoA 更快（未回到 ~1）。  
3. `transpose_tiled` / `pad` ≫ `naive`，并达 copy 的 **~91% / ~92%**；pad 略优于 tiled（bank）。  
4. SoA 在 touch=2/4 报出极高 useful GB/s，主要是工作集吃 L2；解读以加速比曲线为准。

## 重画

```bash
python scripts/plot_b09_layout.py
```

→ `article/02_memory_optim/assets/B-09-speedup-vs-touch.png`  
→ `article/02_memory_optim/assets/B-09-transpose-gbps-bars.png`
