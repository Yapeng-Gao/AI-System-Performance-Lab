# B-03 寄存器（Spilling / Occupancy / launch_bounds）— 参考结果

> 口径：裸跑 CUDA event **median**；加速比 = `baseline_median / mode_median`。  
> 同一次运行打印 `numRegs` / `localSizeBytes` / `occ_blocks`。  
> **禁止**把 `ncu` 附着时程序自打印的 ms 当结论。

## 平台

- GPU：**NVIDIA GeForce RTX 5090**，`sm_120`
- RF：`regsPerSM=65536`，`maxThreadsPerSM=1536`，`regsPerBlock=65536`
- 默认：`n=1048576`，`block=256`，`iters=256`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_03_register_spill`
- 主命令：`--mode modes`

`6 × 256 = 1536` = 本机 `maxThreadsPerSM`，三档都已经贴线程上限。5090 不是教科书里的 2048 threads / 64 warps。

## Modes 全表（主结论 = 最近一次裸跑）

CSV：[`B-03_modes.csv`](B-03_modes.csv)

| mode | first_ms | median_ms | p95_ms | vs baseline | num_regs | local_bytes | occ_blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 4.5558 | 4.5550 | 4.5660 | **1.000×** | 19 | 128 | 6 |
| `highreg` | 14.7762 | 14.7956 | 14.9074 | **0.308×** | 19 | 1024 | 6 |
| `launch_bounds` | 14.9015 | 14.8997 | 14.9180 | **0.306×** | 19 | 1024 | 6 |

`probe_out0` 很大（累加链），循环没被 DCE 吃掉。三档 first ≈ median。

实测图：`article/02_memory_optim/assets/B-03-mode-ms-bars.png`、`B-03-speedup-vs-baseline.png`（`python scripts/plot_b03_register.py`）。

## 怎么读（本机形状）

1. **慢的是 local 足迹，不是 occupancy 台阶。** `numRegs` 三档都是 **19**，`occ_blocks` 都是 **6**。`volatile` + 动态下标让数组没进 RF：`localB` 128 = 32×4 B，1024 = 256×4 B。  
2. **墙钟跟 local 走。** `highreg` **0.308×**（约 3.25× 更慢）。local 字节 8×，时间不是 8×——L1/L2 会藏一块。  
3. **`launch_bounds` 在这台机器上没改分配。** regs / local / occ 与 `highreg` 相同，墙钟 14.90 vs 14.80 ms，抖动带内。要履约，得先出现「regs 太多、住不下 2 block」；本夹具没把数组放进寄存器，契约无物可压。  
4. **不要读成「256 个寄存器 → occupancy 12.5%。」** 那是 §2.5 的阶梯示意。本机 RF 限制器没亮：19×1536 ≈ 29K，远低于 64K。  
5. **和旧稿 4.63 / 16.42 ms 是同一命题。** 旧稿也是 local 变大变慢；缺的是把 `numRegs`/`occ` 打出来，于是容易误读成 occupancy 故事。  
6. 出口：热路径 local → 先缩数组 / 改成能寄存器化的标量；真要看 regs 升、occ 掉台阶，需要另一套夹具（本章未改 binary）。L2 → B-04。

## 复现命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel --target 02_memory_optim_03_register_spill
./build/bin/02_memory_optim_03_register_spill --mode modes
python scripts/plot_b03_register.py
```

## ptxas / NCU（可选；默认不做）

`-Xptxas=-v` 对照 `spill loads/stores` 与 `stack frame`。本机更可能看到大 stack / local，而不是 256 个寄存器。忽略 ncu 附着墙钟。
