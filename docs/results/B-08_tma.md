# B-08 Hopper TMA — 参考结果（待 RTX 5090 / sm_90+ 填表）

> 正文：`article/02_memory_optim/B-08*.md`  
> 示例：`examples/02_memory_optim/08_tma_intro.cu`  
> 一键：`examples/02_memory_optim/08_profile_tma.sh`

## 平台（填写）

- GPU：_（建议 RTX 5090 / sm_120 或 H100 / sm_90）_
- 载荷：`n=4194304`，`tiles/block=64`，`block=256`，`runs=7`，`warmup=2`
- 可执行文件：`02_memory_optim_08_tma_intro`
- 口径：裸跑 CUDA event **median**（`ncu` 附着时程序打印不作证据）

## 采集命令

```bash
# 重新 cmake 后 build
./examples/02_memory_optim/08_profile_tma.sh all

# 单独导出 sweep CSV
./bin/02_memory_optim_08_tma_intro --mode sweep --n 4194304 --tiles 64 --block 256 \
  --runs 7 --warmup 2 | tee /tmp/b08_sweep.txt
# 将表头以下行保存为 docs/results/B-08_sweep.csv

# 固定 fma=8 各 mode
for m in sync bulk1d tensor2d pipe2; do
  ./bin/02_memory_optim_08_tma_intro --mode "$m" --fma-iters 8 --n 4194304 --tiles 64
done
# 整理为 docs/results/B-08_modes.csv（列：mode,median_ms,speedup_vs_sync）

python scripts/plot_b08_tma.py
```

## Intensity sweep（待填）

```text
fma_iters,sync_ms,bulk1d_ms,tensor2d_ms,pipe2_ms,speedup_bulk1d,speedup_tensor2d,speedup_pipe2
```

## 固定 mode（`fma_iters=8`，待填）

| mode | median (ms) | 相对 sync | 一句话 |
|------|-------------|-----------|--------|
| `sync` | | 1.00× | 协作 sync load 基线 |
| `bulk1d` | | | 1D TMA，立刻 wait |
| `tensor2d` | | | 2D tensor-map TMA |
| `pipe2` | | | 2-stage TMA prefetch |

## 可选旁证

- NCU：`DO_NCU=1 ./examples/02_memory_optim/08_profile_tma.sh sync`（再对 bulk1d）
- SASS：`./examples/02_memory_optim/08_dump_sass.sh` → `docs/sass/hopper|blackwell/08_tma_intro.sass`
