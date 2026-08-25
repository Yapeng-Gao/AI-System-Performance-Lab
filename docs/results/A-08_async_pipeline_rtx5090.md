# A-08 Stream / Event pipeline — RTX 5090

> 口径：CUDA event median（warmup=2, runs=7）；`clock64` 人造负载。不报 PCIe GB/s。

- GPU：RTX 5090，`sm_120`，`asyncEngineCount=2`
- Binary：`01_cuda_basics_08_async_pipeline`
- CSV：`A-08_async_pipeline_rtx5090.csv`
- 图：`article/01_cuda_basic/assets/A-08-mode-median-bars.png`（`python scripts/plot_a08_async_pipeline.py`）

| mode | median_ms | vs B |
|---|---|---|
| A serial pageable+default | 11.825 | 8.40× |
| B depth-first pinned | 1.408 | 1× |
| C breadth-first pinned | 1.835 | 1.30× |
