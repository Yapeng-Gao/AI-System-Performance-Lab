# examples/03_compute_primitives

Module C 配套实验。正文在 `article/03_compute_primitives/`。

| 章节 | 文件 | 主命令 | 结果 |
|---|---|---|---|
| **C-01** | `01_warp_primitives.cu`（+ `01_profile_warp.sh`） | `--mode sweep` | `docs/results/C-01_*`；`python scripts/plot_c01_warp_primitives.py` |
| **C-02** | `02_cooperative_groups.cu` | `--mode sweep` / `--mode modes` | `docs/results/C-02_*`；`python scripts/plot_c02_cooperative_groups.py` |
| **C-03** | `03_atomics_contention.cu` | `--mode sweep` / `--mode modes` | `docs/results/C-03_*`；`python scripts/plot_c03_atomics_contention.py` |
| **C-04** | `04_sync_layers.cu` | `--mode sweep` / `--mode sweep_grid` / `--mode modes` | `docs/results/C-04_*`；`python scripts/plot_c04_sync_layers.py` |
| **C-05** | `05_kernel_fusion.cu` | `--mode sweep` / `--mode modes` | `docs/results/C-05_*`；`python scripts/plot_c05_kernel_fusion.py` |

增删 `.cu` 后先重跑 `cmake`，再 build。
