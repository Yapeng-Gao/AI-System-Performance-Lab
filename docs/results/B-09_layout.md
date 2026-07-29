# B-09 数据布局（AoS/SoA/Transpose）— 参考结果

> 待目标卡落盘。跑通后把下方占位表替换为真实 median。  
> 口径：裸跑 CUDA event **median**；带宽为 **useful payload**（触达字段或整矩阵的 R+W）。

## 平台（填写）

- GPU：_（如 RTX 5090）_，`sm_XX`
- 载荷：`n=4194304` particles，`dim=4096`，`touch_fields` 见 sweep，`block=256`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_09_layout_transform`

## Touch-fields sweep（主结论）

```bash
./bin/02_memory_optim_09_layout_transform --mode sweep
```

将 CSV 保存为 `docs/results/B-09_sweep.csv`，表头：

```text
touch_fields,aos_ms,soa_ms,aos_gbps,soa_gbps,speedup_soa
```

| touch_fields | aos_ms | soa_ms | aos_gbps | soa_gbps | speedup_soa |
|---:|---:|---:|---:|---:|---:|
| 1 | _ | _ | _ | _ | _ |
| 2 | _ | _ | _ | _ | _ |
| 4 | _ | _ | _ | _ | _ |
| 8 | _ | _ | _ | _ | _ |

## Modes 全表（layout touch=1 + transpose）

```bash
./bin/02_memory_optim_09_layout_transform --mode modes
```

程序末尾会打印 `mode,median_ms,useful_gbps` 块；保存为 `docs/results/B-09_modes.csv`。

| mode | median_ms | useful_gbps |
|---|---:|---:|
| aos | _ | _ |
| soa | _ | _ |
| copy | _ | _ |
| transpose_naive | _ | _ |
| transpose_tiled | _ | _ |
| transpose_pad | _ | _ |

## 怎么读（预期形状）

1. `touch_fields=1`：`speedup_soa` 应明显 >1。  
2. `touch_fields→8`：加速比**通常收窄**（不要求回到 ~1）；看形状，勿只看绝对 GB/s。  
3. `transpose_tiled` / `pad` ≫ `naive`，并靠近 `copy`。  
4. 程序会对 transpose/copy 做轻量正确性检查，失败则非 0 退出。

## 重画

```bash
python scripts/plot_b09_layout.py
```

→ `article/02_memory_optim/assets/B-09-speedup-vs-touch.png`  
→ `article/02_memory_optim/assets/B-09-transpose-gbps-bars.png`
