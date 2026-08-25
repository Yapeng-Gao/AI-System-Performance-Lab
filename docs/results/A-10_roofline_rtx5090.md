# A-10 Roofline probes — RTX 5090

> 口径：CUDA event median（warmup=2, runs=7）。ridge 用**实测**屋顶。NCU 未跑。

- GPU：RTX 5090，`sm_120`，170 SM，512-bit
- Binary：`01_cuda_basics_10_roofline_demo`
- CSV：`A-10_roofline_rtx5090.csv`
- 图：`article/01_cuda_basic/assets/A-10-measured-roofline.png`（`python scripts/plot_a10_roofline.py`）

| probe | AI (FLOP/byte) | perf | note |
|---|---|---|---|
| copy | ≈0（图上画 0.01） | 1953.56 GB/s → 斜坡侧 | 短核/L2 可使绝对 GB/s 偏高 |
| FMA | 1000 | 49.01 TFLOPS | compute 侧 |
| ridge | 25.09 | — | TFLOPS×1000/GB/s |
