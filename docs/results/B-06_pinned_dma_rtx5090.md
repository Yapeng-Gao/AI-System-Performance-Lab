# B-06 Pinned / DMA — RTX 5090 参考结果

> 摘自正文 `article/02_memory_optim/B-06*.md` §5；便于仓库内对照，不以本文件替代文章解读。

## 平台

- GPU：RTX 5090，`asyncEngineCount=2`，`canMapHostMemory=1`
- 载荷：256 MiB；chunk=16 MiB；`warmup=1`，`runs=5`
- 口径：median（除非另注）
- 可执行文件：`02_memory_optim_06_pinned_dma`

## 结果摘要

| mode | median (ms) | GB/s | 备注 |
|------|-------------|------|------|
| `pinned` | 4.769 | 52.42 | 单向上限 ≈ Gen5 理论 82% |
| `serial` (iters=8) | 5.123 | 48.80 | 串行基线 |
| `overlap` (iters=8) | 4.808 | 51.99 | vs serial +6.1%，贴 pinned |
| `serial` (iters=256) | 5.331 | 46.90 | |
| `overlap` (iters=256) | 4.819 | 51.87 | vs serial +10.6%，仍贴 pinned |
| `bidir` | 5.336 | 93.70 | ≈ 1.79× 单向 |
| `mapped` | 5.234 | 47.76 | 单遍 host-read 有效带宽 |

## NSYS CLI 旁证（overlap，iters=256）

- 96 × H2D @ 16.777 MB；memcpy med ≈ 298 µs
- 96 × `light_touch_kernel`；med ≈ 29 µs
- 结论：copy ≫ kernel（约 10×）→ 相对加速只有几个点仍算 overlap 成功

## 复现

```bash
bash examples/02_memory_optim/06_profile_pinned_dma.sh
```
