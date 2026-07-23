# B-07 Async Copy / Pipeline — 本机结果（待填）

> 配套正文：`article/02_memory_optim/B-07*.md` §5。  
> 跑完 `bash examples/02_memory_optim/07_profile_cp_async_pipeline.sh` 后回填本表。

## 平台

- GPU：
- SM：
- `n` / `tiles` / `block` / `runs` / `warmup`：
- 可执行文件：`02_memory_optim_07_cp_async_pipeline`

## 固定 fma-iters 对照

| mode | fma_iters | median (ms) | 相对 sync | 备注 |
|------|-----------|-------------|-----------|------|
| sync | | | 1.00× | |
| async1 | | | | |
| pipe2 | | | | |
| pipe4 | | | | |
| pipe2_blk | | | | |

## Intensity sweep（`--mode sweep`）

| fma_iters | sync_ms | pipe2_ms | pipe4_ms | speedup_pipe2 | speedup_pipe4 |
|----------|---------|----------|----------|---------------|---------------|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 32 | | | | | |
| 64 | | | | | |
| 128 | | | | | |
| 256 | | | | | |

## 一句话结论

- 低 AI 段最大加速比：
- 高 AI 段是否掉到 ≤1：
- `pipe4` vs `pipe2`：
- `pipe2_blk` vs `pipe2`：

## 复现

```bash
bash examples/02_memory_optim/07_profile_cp_async_pipeline.sh
DO_NCU=1 bash examples/02_memory_optim/07_profile_cp_async_pipeline.sh pipe2
```
