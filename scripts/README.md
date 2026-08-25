# 脚本工具

本目录包含辅助脚本。章节专用 profile 脚本在 `examples/**` 同目录；这里放**跨章复用**与**正文数据图重画**。

## 已提供的脚本

### 正文实测图（Module A 补充；B-05～）

| 脚本 | 输入 CSV | 输出 PNG |
|------|----------|----------|
| `plot_a08_async_pipeline.py` | `docs/results/A-08_async_pipeline_rtx5090.csv` | `article/01_cuda_basic/assets/A-08-mode-median-bars.png` |
| `plot_a10_roofline.py` | `docs/results/A-10_roofline_rtx5090.csv` | `article/01_cuda_basic/assets/A-10-measured-roofline.png` |
| `plot_b05_unified_memory.py` | `docs/results/B-05_modes_warm.csv`、`B-05_cold_fault.csv` | `B-05-mode-latency-bars.png`、`B-05-cold-vs-warm.png` |
| `plot_b06_pinned_dma.py` | `docs/results/B-06_modes.csv`、`B-06_overlap.csv` | `B-06-mode-gbs-bars.png`、`B-06-overlap-median-bars.png` |
| `plot_b07_cp_async.py` | `docs/results/B-07_sweep.csv`、`B-07_modes.csv` | `B-07-speedup-vs-fma.png`、`B-07-mode-speedup-bars.png` |

```bash
# 仓库根目录；需 matplotlib
python scripts/plot_a08_async_pipeline.py
python scripts/plot_a10_roofline.py
python scripts/plot_b05_unified_memory.py
python scripts/plot_b06_pinned_dma.py
python scripts/plot_b07_cp_async.py
```

### 通用 profiling / Roofline

- `dump_sass.sh` — 对部分 `examples/02_memory_optim/*.cu` 导出 SASS 到 `docs/sass/`（章节也可用同目录 `0N_dump_sass.sh`）
- `profile_ncu.sh` — NCU 采集，输出 CSV 到 `docs/results/ncu`（目录按需创建）
- `profile_ncu_l2_residency.sh` — L2 residency 证据三件套
- `parse_roofline.py` — 解析 NCU CSV → 带宽 / TFLOPs / OI
- `plot_roofline.py` — 根据 JSON + 硬件屋顶线画图

> 专栏主线实验优先用 `examples/**/0N_profile_*.sh`。

## 使用说明（Roofline 链路）

```bash
cd build && cmake --build . --parallel && cd ..
chmod +x scripts/*.sh   # Linux
./scripts/dump_sass.sh
./scripts/profile_ncu.sh
python scripts/parse_roofline.py docs/results/ncu/<file>.csv
python scripts/plot_roofline.py data.json "GPU-Name" <BW_GBps> <TFLOPs_peak> output.png
```

目录用途总览见 [`docs/仓库架构与现状.md`](../docs/仓库架构与现状.md)。
