# B-02 Shared Memory（Bank / Padding / Swizzle）— 参考结果

> 口径：裸跑 CUDA event **median**；加速比 = `naive_median / mode_median`。  
> **禁止**把 `ncu` 附着时程序自打印的 ms 当结论。

## 平台

- GPU：**NVIDIA GeForce RTX 5090**，`sm_120`
- 默认：`grid=2048`，`block=32`，`iters=8192`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_02_shared_mem_bank_conflict`
- 主命令：`--mode modes`

## Modes 全表（主结论 = 最近一次裸跑）

CSV：[`B-02_modes.csv`](B-02_modes.csv)

| mode | first_ms | median_ms | p95_ms | mean_ms | vs naive |
|---|---:|---:|---:|---:|---:|
| `naive` | 1.2538 | 1.2576 | 1.2598 | 1.2559 | **1.000×** |
| `padding` | 0.0894 | 0.0891 | 0.0897 | 0.0885 | **14.121×** |
| `swizzle` | 0.1005 | 0.1014 | 0.1042 | 0.1012 | **12.405×** |

`probe_out0=126976.0`（tid=0 列扫累加；防 DCE 有效）。

三档 first ≈ median，**不是** B-01 `float4` 那种 0.07 ms 核上的抖动。

实测图：`article/02_memory_optim/assets/B-02-mode-ms-bars.png`、`B-02-speedup-vs-naive.png`（`python scripts/plot_b02_shared_mem.py`）。

## 怎么读（本机形状）

1. **列扫冲突是真墙钟，不是教科书恐吓。** `naive` 1.26 ms，padding / swizzle 收到 0.089 / 0.101 ms。和 B-01 `offset=1` 三档贴齐（0.97～1.04×）不是同一命题：这里行长=32 的列访问是 32-way。  
2. **加速比小于 32 仍正常。** 本机 14.1× / 12.4×。event 包了 tile init + `__syncthreads` + 列扫；流水线也会掩盖一部分 replay。不要用「不到 32× 所以处方无效」否定这两档。  
3. **两档处方同向。** padding 略快（14.1× vs 12.4×，约 1.14×）。XOR 多一次地址计算，且读的是置换后的列。选 padding 还是 swizzle 看你是否必须紧凑，不要看谁「更现代」。  
4. **旧稿 `clock64` 单 warp「20×」不当结论。** 主证据是这条 event median。  
5. 出口：处方有效仍慢 → B-03（occupancy / spill）；GMEM→SMEM 异步 → B-07；描述符 / `ldmatrix` → B-08 / D-05。

## 复现命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel --target 02_memory_optim_02_shared_mem_bank_conflict
./build/bin/02_memory_optim_02_shared_mem_bank_conflict --mode modes
python scripts/plot_b02_shared_mem.py
```

## NCU 旁证（可选；默认不做）

若要验证 naive 的 shared wavefront ≈ 32、padding/swizzle ≈ 1，可跑 `examples/02_memory_optim/02_profile_banks.sh`；**忽略**附着墙钟。
