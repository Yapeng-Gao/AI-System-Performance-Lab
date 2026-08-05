# examples/03_compute_primitives

Module C 配套实验。正文在 `article/03_compute_primitives/`。

| 章节 | 文件 | 主命令 | 结果 |
|---|---|---|---|
| **C-01** | `01_warp_primitives.cu`（+ `01_profile_warp.sh`） | `--mode sweep` | `docs/results/C-01_*`；`python scripts/plot_c01_warp_primitives.py` |
| **C-02** | `02_cooperative_groups.cu` | `--mode sweep` / `--mode modes` | `docs/results/C-02_*`（待测）；`python scripts/plot_c02_cooperative_groups.py` |

增删 `.cu` 后先重跑 `cmake`，再 build。
