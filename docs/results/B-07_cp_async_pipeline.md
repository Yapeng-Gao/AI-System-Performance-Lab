# B-07 Async Copy / Pipeline — RTX 5090 参考结果

> 摘自正文 `article/02_memory_optim/B-07*.md` §5；便于仓库内对照。  
> 口径：裸跑 CUDA event **median**。`ncu` 附着时程序打印的 ms/GB/s 不作证据。

## 平台

- GPU：RTX 5090，`sm_120`，`sharedMemPerBlock=48 KB`
- 载荷：`n=4194304`，`tiles/block=64`，`block=256`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_07_cp_async_pipeline`
- 读数：`speedup = sync_ms / pipe_ms`（>1 赚，≈1 平，<1 亏）
- 封面：`article/02_memory_optim/assets/B-07-cp-async-pipeline-cover.png`
- SASS：`docs/sass/blackwell/07_cp_async_pipeline.sass`（本机已 dump）

## Intensity sweep（裸跑，完整）

```text
fma_iters,sync_ms,pipe2_ms,pipe4_ms,speedup_pipe2,speedup_pipe4
1,0.016864,0.013248,0.013952,1.2729,1.2087
2,0.016288,0.014208,0.013120,1.1464,1.2415
4,0.017376,0.013216,0.015008,1.3148,1.1578
8,0.019328,0.017280,0.017248,1.1185,1.1206
16,0.024384,0.022432,0.023104,1.0870,1.0554
32,0.033472,0.031584,0.031616,1.0598,1.0587
64,0.051872,0.050048,0.052736,1.0364,0.9836
128,0.088960,0.089824,0.093056,0.9904,0.9560
256,0.163488,0.170752,0.173792,0.9575,0.9407
```

| fma_iters | sync_ms | pipe2_ms | pipe4_ms | speedup_pipe2 | speedup_pipe4 |
|----------|---------|----------|----------|---------------|---------------|
| 1 | 0.016864 | 0.013248 | 0.013952 | **1.273** | **1.209** |
| 2 | 0.016288 | 0.014208 | 0.013120 | **1.146** | **1.241** |
| 4 | 0.017376 | 0.013216 | 0.015008 | **1.315** | **1.158** |
| 8 | 0.019328 | 0.017280 | 0.017248 | **1.119** | **1.121** |
| 16 | 0.024384 | 0.022432 | 0.023104 | **1.087** | **1.055** |
| 32 | 0.033472 | 0.031584 | 0.031616 | **1.060** | **1.059** |
| 64 | 0.051872 | 0.050048 | 0.052736 | **1.036** | 0.984 |
| 128 | 0.088960 | 0.089824 | 0.093056 | 0.990 | 0.956 |
| 256 | 0.163488 | 0.170752 | 0.173792 | 0.958 | 0.941 |

```text
speedup_pipe2

 1.32 ┤    ●4
 1.27 ┤ ●1
 1.15 ┤  ●2
 1.12 ┤     ●8
 1.09 ┤      ●16
 1.06 ┤       ●32
 1.04 ┤        ●64
 1.00 ┤──────────●128── 盈亏线
 0.96 ┤             ●256
      └──────────────────► fma_iters
```

## 固定 mode（裸跑，`fma_iters=8`）

| mode | median (ms) | ~GB/s | 相对 sync | 备注 |
|------|-------------|-------|-----------|------|
| `sync` | 0.0213 | 733.16 | 1.00× | 基线 |
| `async1` | 0.0231 | 677.23 | **0.92×** | 立刻 wait，无 overlap → 更慢 |
| `pipe2` | 0.0193 | 811.10 | **1.10×** | 本组最佳 |
| `pipe4` | 0.0202 | 773.82 | **1.05×** | 快于 sync，差于 pipe2 |
| `pipe2_blk` | 0.0207 | 754.69 | **1.03×** | shared pipeline 税 |

## NCU 旁证（WarpStateStats，`fma_iters=4`，`kernel_sync`）

- 主导 stall：**fixed latency execution dependency**（短依赖）  
- Est. Local Speedup ≈ **40.3%**  
- `launch__waves_per_multiprocessor` = 0.25  
- 解读：高强度段短依赖变重时，再叠 pipeline 不划算

## SASS 旁证（sm_120 / Blackwell，已 dump）

```bash
bash examples/02_memory_optim/07_dump_sass.sh
grep -nE 'LDGSTS|CP\.ASYNC|LDG\.E|STS' docs/sass/blackwell/07_cp_async_pipeline.sass | head
```

本机命中摘要：

- 多处 `LDGSTS.E ...` + `ARRIVES.LDGSTSBAR.64.TRANSCNT` → async/pipeline 路径生效  
- 同文件可见 `LDG.E` + `STS` → sync 对照仍在二进制中（预期）

## 结论

1. 极低～中低强度（`fma=1~32`）：`pipe2` 约 **1.06～1.31×**，优先 2-stage thread-local。  
2. 高强度（`fma≥128`）：加速比 ≤1，不必强上 async pipeline。  
3. `async1` 无 overlap 会亏；`pipe4` / `pipe2_blk` 通常不优于 `pipe2`。  
4. SASS 已确认 `LDGSTS`，不是“只换了 API 名”。

## 复现

```bash
./bin/02_memory_optim_07_cp_async_pipeline --mode sweep

for m in sync async1 pipe2 pipe4 pipe2_blk; do
  ./bin/02_memory_optim_07_cp_async_pipeline --mode $m --fma-iters 8
done

ncu --launch-skip 2 --launch-count 1 --section WarpStateStats --section MemoryWorkloadAnalysis -o cp_async_sync ./bin/02_memory_optim_07_cp_async_pipeline --mode sync --fma-iters 4 --runs 1 --warmup 2
bash examples/02_memory_optim/07_dump_sass.sh
```
