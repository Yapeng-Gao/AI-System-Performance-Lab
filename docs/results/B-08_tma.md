# B-08 Hopper TMA — RTX 5090 参考结果

> 摘自 `./bin/02_memory_optim_08_tma_intro --mode sweep`（Ubuntu / Release）。  
> 口径：裸跑 CUDA event **median**。

## 平台

- GPU：RTX 5090，`sm_120`，`sharedMemPerBlock=48 KB`
- 载荷：`n=4194304`（4096 tiles），`tiles/block=64`，`grid=64`，`block=256`，`runs=7`（默认），`warmup=2`
- 2D 视图：`2048×2048`，`tiles_x=64`
- 可执行文件：`02_memory_optim_08_tma_intro`
- 封面：`article/02_memory_optim/assets/B-08-tma-cover.png`

## Intensity sweep（裸跑，完整）

```text
fma_iters,sync_ms,bulk1d_ms,tensor2d_ms,pipe2_ms,speedup_bulk1d,speedup_tensor2d,speedup_pipe2
1,0.036832,0.035008,0.040352,0.021824,1.0521,0.9128,1.6877
2,0.036160,0.037280,0.042112,0.023776,0.9700,0.8587,1.5209
4,0.040864,0.042912,0.047520,0.028800,0.9523,0.8599,1.4189
8,0.050528,0.050944,0.056608,0.040160,0.9918,0.8926,1.2582
16,0.069536,0.071040,0.075424,0.058784,0.9788,0.9219,1.1829
32,0.107424,0.109472,0.114688,0.096448,0.9813,0.9367,1.1138
64,0.183552,0.185248,0.190336,0.173184,0.9908,0.9644,1.0599
128,0.336320,0.337600,0.343200,0.325376,0.9962,0.9800,1.0336
256,0.641760,0.643264,0.647616,0.629728,0.9977,0.9910,1.0191
```

| fma_iters | sync_ms | bulk1d | tensor2d | pipe2 | sp_bulk1d | sp_tensor2d | sp_pipe2 |
|----------|---------|--------|----------|-------|-----------|-------------|----------|
| 1 | 0.0368 | 0.0350 | 0.0404 | 0.0218 | **1.052** | 0.913 | **1.688** |
| 2 | 0.0362 | 0.0373 | 0.0421 | 0.0238 | 0.970 | 0.859 | **1.521** |
| 4 | 0.0409 | 0.0429 | 0.0475 | 0.0288 | 0.952 | 0.860 | **1.419** |
| 8 | 0.0505 | 0.0509 | 0.0566 | 0.0402 | 0.992 | 0.893 | **1.258** |
| 16 | 0.0695 | 0.0710 | 0.0754 | 0.0588 | 0.979 | 0.922 | **1.183** |
| 32 | 0.1074 | 0.1095 | 0.1147 | 0.0964 | 0.981 | 0.937 | **1.114** |
| 64 | 0.1836 | 0.1852 | 0.1903 | 0.1732 | 0.991 | 0.964 | **1.060** |
| 128 | 0.3363 | 0.3376 | 0.3432 | 0.3254 | 0.996 | 0.980 | 1.034 |
| 256 | 0.6418 | 0.6433 | 0.6476 | 0.6297 | 0.998 | 0.991 | 1.019 |

CSV：`docs/results/B-08_sweep.csv`。重画：`python scripts/plot_b08_tma.py`。

## 固定 mode（取 sweep 的 `fma_iters=8` 行）

| mode | median (ms) | 相对 sync | 一句话 |
|------|-------------|-----------|--------|
| `sync` | 0.0505 | 1.00× | 协作 sync load 基线 |
| `bulk1d` | 0.0509 | **0.99×** | 1D TMA 立刻 wait ≈ 打平 |
| `tensor2d` | 0.0566 | **0.89×** | 2D 路径立刻 wait 更慢 |
| `pipe2` | 0.0402 | **1.26×** | 2-stage prefetch 才赚 |

## 怎么读

1. **立刻 wait ≠ 加速**：`bulk1d` 全程约 0.95～1.05×；`tensor2d` 全程约 0.86～0.99×。  
2. **重叠才是产品**：`pipe2` 在 `fma=1` 达约 **1.69×**，随后随 AI 回落到 ~1.02×。  
3. 与正文标题一致：TMA 引擎本身不是免费午餐；**prefetch ∥ compute** 才藏延迟。

## NCU 旁证（`fma_iters=1`，RTX 5090 / sm_120）

```bash
DO_NCU=1 bash examples/02_memory_optim/08_profile_tma.sh ncu-only
```

> **不要**把 ncu 附着时程序自打印的 ms/GB/s 当结论（会被 replay 放大到秒级）。下表来自 `ncu --import … --page details` / CLI metrics。

### WarpState + MemoryWorkload

| mode | Mem Throughput | Max Bandwidth | Warp cycles / issued | 主导 stall（NCU OPT） | Est. Local Speedup |
|------|----------------|---------------|----------------------|----------------------|--------------------|
| `sync` | 351 GB/s | 19.9% | 14.12 | **L1TEX scoreboard** ~53%（等全局/TEX 数据） | 53.5% |
| `bulk1d` | 338 GB/s | 19.2% | 12.08 | **branch target** ~30%（控制流 / elect·barrier） | 30.2% |
| `pipe2` | **742 GB/s** | **42.1%** | **8.39** | **fixed latency** ~34%（短依赖） | 34.2% |

### 指令量（`smsp__inst_executed.sum`，同 launch：512 warps）

| mode | inst_executed | vs sync |
|------|---------------|---------|
| `sync` | 4 124 160 | 1.00× |
| `bulk1d` | 4 668 448 | **1.13×**（更多，不是更少） |

### 怎么读（与裸跑 `sweep` 对齐）

1. **`pipe2` 才像产品**：吞吐约 **2.1×** sync、issue 间隔从 14.1→8.4 cycle；主导 stall 从「等 L1TEX」转到「短依赖」——和 event 低 AI ~**1.69×** 同向。  
2. **`bulk1d` 立刻 wait ≈ 换皮**：Mem Throughput 与 sync 同量级（338 vs 351）；stall 从 L1TEX 换成 branch，**没有**带宽翻倍。  
3. **本 microbench 未证明「换 TMA 就少指令」**：4 KiB tile + 全 block `mbarrier` 协议下，`bulk1d` 的 `inst_executed` 反而多 ~13%。卸地址/发令压力更适合大 tile / 多维 / warp-spec；本章主结论仍是 **overlap**。

## SASS 旁证（sm_90 + sm_120）

```bash
bash examples/02_memory_optim/08_dump_sass.sh
```

落盘：`docs/sass/{hopper,blackwell}/08_tma_intro.sass`。

| 观察 | 含义 |
|---|---|
| **`UTMALDG.2D`**（hopper / blackwell 均有） | **tensor-map 2D TMA load** 已进二进制（`kernel_tensor2d` 路径） |
| 多处 **`ELECT`** | `ptx::elect_sync` / 选举 issue 线程生效 |
| **`EIATTR_MBARRIER_*`** | G2S 完成模型走 **mbarrier**（与正文同步节一致） |
| `kernel_bulk1d` `.text` 仍可见 **`LDG.E` + `STS`** | 1D `memcpy_async_tx` 反汇编不如 2D 干净；与 NCU「bulk1d 吞吐≈sync」同向——**立刻 wait 的 1D 路径不是本章英雄** |
| `kernel_sync` | 预期以 LDG+STS 为主（协作 sync load） |

> 解读优先级：裸跑 `sweep` > NCU stall/吞吐 > SASS「路径是否存在」。SASS 证明 **TMA/elect/mbarrier 没写空**；不单独用 SASS 报加速比。

## 复现命令

```bash
./bin/02_memory_optim_08_tma_intro --mode sweep
python scripts/plot_b08_tma.py
DO_NCU=1 bash examples/02_memory_optim/08_profile_tma.sh ncu-only
bash examples/02_memory_optim/08_dump_sass.sh
```
