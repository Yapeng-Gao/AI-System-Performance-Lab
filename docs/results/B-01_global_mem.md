# B-01 Global Memory（合并 / 对齐 / float4）— 参考结果

> 口径：裸跑 CUDA event **median**；带宽为 **R+W useful payload**（`2 * n * sizeof(float)`，GiB/s = /1024³）。  
> 绝对 GB/s 可因 L2 命中高于 DRAM 总线标称带宽——**主看相对 `aligned` 的加速比与形状**，勿当总线利用率。  
> **禁止**把 `ncu` 附着时程序自打印的 ms/GB/s 当结论。

## 平台

- GPU：**NVIDIA GeForce RTX 5090**，`sm_120`
- 载荷：`n=16777216`（64.00 MiB）floats，`offset=1`，`block=256`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_01_global_mem_bandwidth`
- 主命令：`--mode modes`

## Modes 全表（主结论 = 最近一次裸跑）

CSV：[`B-01_modes.csv`](B-01_modes.csv)

| mode | median_ms | gbps (R+W) | vs aligned |
|---|---:|---:|---:|
| `misaligned` | 0.0780 | 1603.55 | **0.988×** |
| `aligned` | 0.0770 | 1622.87 | **1.000×** |
| `float4` | 0.0742 | 1683.73 | **1.038×** |

同机先前一轮（已替换，仅对照抖动）：`misaligned` 0.973× / `float4` 1.001×。两轮都在 **0.97～1.04** 带内。

`float4` 本轮 first=0.0791、median=0.0742、p95=0.0782——median 比 first 快，**不要**把 1.038× 读成稳定数倍红利。

实测图：`article/02_memory_optim/assets/B-01-mode-gbps-bars.png`、`B-01-speedup-vs-aligned.png`（`python scripts/plot_b01_global_mem.py`）。

## 怎么读（本机形状）

1. **三档几乎贴齐**：`offset=1` 流式 copy 在大 L2 的 5090 上，小错位和显式 `float4` 都拉不开墙钟。这是有效结论，不是测坏了。  
2. **和旧 B-01 稿的「数倍」不是同一命题**。旧稿把 stride 打散（useful/transferred≈12.5%）、HBM3e 8 TB/s 营销、TMA/L2 叠进一章，并承诺「亲眼见证数倍」。本章只测**同一 useful payload、读侧平移 1 float**。该差的是实验定义，不是这台 5090 失灵。  
3. **和 A-10 copy ≈1954 GB/s 也不横比绝对数**。A-10 用十进制 GB（/1e9）、纯 `float4` copy、无 `+1`；本章用 GiB（/1024³）且每元素 `+1` 写回。主看相对 `aligned`。  
4. **`offset=1` ≠ 经典跨步灾难**：warp 仍大体连续。要看合并崩盘，换跨步/打散或 NCU `sectors/request`。真正的形状差在 B-09（touch=1 时 SoA/AoS ≈13.6×）。  
5. 出口：形状已修仍 latency-bound → B-07；要管 L2 → B-04；布局 → B-09。

## 复现命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel --target 02_memory_optim_01_global_mem_bandwidth
./build/bin/02_memory_optim_01_global_mem_bandwidth --mode modes
python scripts/plot_b01_global_mem.py
```

## NCU 旁证（可选；默认不做）

若要验证「小 offset 是否仍接近理想 sector」，可对三档采 `sectors/request`；**忽略**附着墙钟。
