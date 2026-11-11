# C-10 Module C Checklist — 证据索引

> 本章**无新测**。  
> **先读正文 §2（如何认出症状）**，再用来下表打开对应 binary。  
> 数字口径见各章摘要（裸跑 CUDA event **median**，RTX 5090 / `sm_120`）。  
> **禁止**把 `ncu` 附着时程序自打印的 ms 当结论。

正文：`article/03_compute_primitives/C-10*.md`

## 本机形状锚点（RTX 5090 / sm_120，已落盘章）

| 章 | 锚点（摘要） | 明细 |
|---|---|---|
| C-01 | 1 warp ≈ **1.02×**；16 warp **1.15×**；32 warp shfl/smem **1.27×**。int redux ≈ **1.06×**。NCU：inst 约 −42%，shared wavefront 约 −78%（勿用附着 ms） | [C-01_warp_primitives.md](C-01_warp_primitives.md) |
| C-02 | tile≤32 ≈ **0.93～1.00×**；tile 64/128 = **1.60× / 1.66×** | [C-02_cooperative_groups.md](C-02_cooperative_groups.md) |
| C-03 | hit=1 时 smem/naive **6.31×**；手写 warp-agg ≈ **1.00×** | [C-03_atomics_contention.md](C-03_atomics_contention.md) |
| C-04 | grid/block **17.2×**；block/warp **1.10×～1.23×**；`phases` ≈ **1.0** | [C-04_sync_layers.md](C-04_sync_layers.md) |
| C-05 | k=2 **3.87×** → k=8 **9.81×**；fat/fused **8.01×**（更慢） | [C-05_kernel_fusion.md](C-05_kernel_fusion.md) |
| C-06 | 节点≥2 **~3.3～4.1×**（峰值 **4.08×**）；work=0 **5.48×**；work=4096 **1.01×**；定点 **3.68×** | [C-06_cuda_graph.md](C-06_cuda_graph.md) |
| C-07 | n_tasks 64→**32×**，16384→**3740×**；work=4096 仍 **314×**；oversub/persistent **1.48×** | [C-07_persistent.md](C-07_persistent.md) |
| C-08 | 曲线贴 **1×**；sweep 峰值 **1.10×**；work=0 **0.95×**；定点 **1.13×**；`pdl_full`/`pdl` **0.999×** | [C-08_pdl.md](C-08_pdl.md) |
| C-09 | 两边相等 **1.97×**；一侧为 0 **1.04×～1.05×**；`named_pre` **1.007×**；`indep` **0.97×** | [C-09_named_barrier.md](C-09_named_barrier.md) |
| 访存 | 不在本章重列。touch=1 SoA/AoS ≈ **13.6×** 等见 B-10 | [B-10_checklist.md](B-10_checklist.md) |

## Binary 与主命令

路径前缀：`examples/03_compute_primitives/`。可执行文件在 `build/bin/` 或 CLion `cmake-build-*/bin/`。

| 章 | 源文件 | Binary | 建议主命令 | Results / Plot |
|---|---|---|---|---|
| C-01 | `01_warp_primitives.cu` | `03_compute_primitives_01_warp_primitives` | `--mode sweep` | `C-01_*`；`plot_c01_warp_primitives.py` |
| C-02 | `02_cooperative_groups.cu` | `03_compute_primitives_02_cooperative_groups` | `--mode sweep` | `C-02_*`；`plot_c02_cooperative_groups.py` |
| C-03 | `03_atomics_contention.cu` | `03_compute_primitives_03_atomics_contention` | `--mode sweep` | `C-03_*`；`plot_c03_atomics_contention.py` |
| C-04 | `04_sync_layers.cu` | `03_compute_primitives_04_sync_layers` | `--mode modes` | `C-04_*`；`plot_c04_sync_layers.py` |
| C-05 | `05_kernel_fusion.cu` | `03_compute_primitives_05_kernel_fusion` | `--mode sweep` | `C-05_*`；`plot_c05_kernel_fusion.py` |
| C-06 | `06_cuda_graph.cu` | `03_compute_primitives_06_cuda_graph` | `--mode sweep`（加重看 `sweep_work`） | `C-06_*`；`plot_c06_cuda_graph.py` |
| C-07 | `07_persistent.cu` | `03_compute_primitives_07_persistent` | `--mode sweep`（加重看 `sweep_work`） | `C-07_*`；`plot_c07_persistent.py` |
| C-08 | `08_pdl.cu` | `03_compute_primitives_08_pdl` | `--mode sweep`（尾巴看 `sweep_tail`） | `C-08_*`；`plot_c08_pdl.py`。门槛 **sm_90+** |
| C-09 | `09_named_barrier.cu` | `03_compute_primitives_09_named_barrier` | `--mode sweep`（反例看 `modes`） | `C-09_*`；`plot_c09_named_barrier.py`。unaligned 门槛 **sm_70+** |

## 可选旁证脚本（各章已有；本章不新增）

| 章 | 脚本 |
|---|---|
| C-01 | `01_profile_warp.sh`（可选；忽略附着墙钟） |

NCU 过滤约定：`--kernel-name-base function|demangled|mangled` + `--kernel-name regex:…`。
