# B-07 Async Copy / Pipeline — RTX 5090 参考结果

> 摘自正文 `article/02_memory_optim/B-07*.md` §5；便于仓库内对照。  
> 口径：裸跑 CUDA event **median**。`ncu` 附着时程序打印的 ms/GB/s 不作证据。

## 平台

- GPU：RTX 5090，`sm_120`，`sharedMemPerBlock=48 KB`
- 载荷：`n=4194304`，`tiles/block=64`，`block=256`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_07_cp_async_pipeline`
- 读数：`speedup = sync_ms / pipe_ms`（>1 赚，≈1 平，<1 亏）
- 封面：`article/02_memory_optim/assets/B-07-cp-async-pipeline-cover.png`

## Intensity sweep（裸跑）

`fma=8` 取自同日固定 mode 的 sync/pipe2/pipe4（口径相同）。`1/2/4` 若完整 `sweep` 终端仍在，用文末命令抽出后可贴入下表。

```text
fma_iters,sync_ms,pipe2_ms,pipe4_ms,speedup_pipe2,speedup_pipe4
8,0.0213,0.0193,0.0202,1.1036,1.0545
16,0.024384,0.022432,0.023104,1.0870,1.0554
32,0.033472,0.031584,0.031616,1.0598,1.0587
64,0.051872,0.050048,0.052736,1.0364,0.9836
128,0.088960,0.089824,0.093056,0.9904,0.9560
256,0.163488,0.170752,0.173792,0.9575,0.9407
```

| fma_iters | sync_ms | pipe2_ms | pipe4_ms | speedup_pipe2 | speedup_pipe4 |
|----------|---------|----------|----------|---------------|---------------|
| 1 | （待完整 sweep） | | | | |
| 2 | （待完整 sweep） | | | | |
| 4 | （待完整 sweep） | | | | |
| 8 | 0.0213 | 0.0193 | 0.0202 | **1.104** | **1.054** |
| 16 | 0.024384 | 0.022432 | 0.023104 | **1.087** | **1.055** |
| 32 | 0.033472 | 0.031584 | 0.031616 | **1.060** | **1.059** |
| 64 | 0.051872 | 0.050048 | 0.052736 | **1.036** | 0.984 |
| 128 | 0.088960 | 0.089824 | 0.093056 | 0.990 | 0.956 |
| 256 | 0.163488 | 0.170752 | 0.173792 | 0.958 | 0.941 |

```text
speedup_pipe2

 1.10 ┤ ●8
 1.09 ┤  ●16
 1.06 ┤   ●32
 1.04 ┤    ●64
 1.00 ┤─────────●128── 盈亏线
 0.96 ┤            ●256
      └──────────────────► fma_iters
```

- `fma=8~32`：pipe2 约 **+6%～+10%**（能藏延迟）  
- `fma≥128`：掉到 **≤1**；256 约 **0.96×**（会变慢）  
- 从 64 起 `pipe4` 已弱于 `pipe2`，且先进入亏损

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
- sm_120 上部分 legacy named metrics 曾为 **n/a**  
- 解读：高强度段短依赖变重时，再叠 pipeline 不划算

## SASS 旁证

```bash
bash examples/02_memory_optim/07_dump_sass.sh
grep -nE 'LDGSTS|CP\.ASYNC|LDG\.E|STS' docs/sass/blackwell/07_cp_async_pipeline.sass | head
```

期望：`pipe`/`async` 路径出现 `LDGSTS` 或 `CP.ASYNC`；`sync` 对照以 `LDG`/`STS` 为主。产物目录：`docs/sass/{ampere,blackwell}/`。

## 结论

1. 中低强度（约 `fma=8~32`）：优先 **2-stage thread-local**（约 1.06～1.10×）。  
2. 高强度（`fma≥128`）：加速比 ≤1，不必强上 async pipeline。  
3. `async1` 无 overlap 会亏；`pipe4` / `pipe2_blk` 通常不如 `pipe2`。

## 复现

```bash
# 主证据：裸跑（在 build/）
./bin/02_memory_optim_07_cp_async_pipeline --mode sweep
./bin/02_memory_optim_07_cp_async_pipeline --mode sweep | awk -F, 'NR==1 || ($1+0>0 && $1+0<=8)'

for m in sync async1 pipe2 pipe4 pipe2_blk; do
  ./bin/02_memory_optim_07_cp_async_pipeline --mode $m --fma-iters 8
done

# 旁证
ncu --launch-skip 2 --launch-count 1 --section WarpStateStats --section MemoryWorkloadAnalysis -o cp_async_sync ./bin/02_memory_optim_07_cp_async_pipeline --mode sync --fma-iters 4 --runs 1 --warmup 2
bash examples/02_memory_optim/07_dump_sass.sh
```
