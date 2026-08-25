# B-10 Module B Checklist — 证据索引

> 本章**无新测**。  
> **先读正文 §2（如何认出症状）**，再用来下表打开对应 binary。  
> 数字口径见各章摘要（裸跑 CUDA event **median**）。带宽章绝对 GB/s 可能含 L2——**主看加速比/相对形状**。  
> **禁止**把 `ncu` 附着时程序自打印的 ms/GB/s 当结论。

正文：同目录文章 `article/02_memory_optim/B-10*.md`

## 本机形状锚点（RTX 5090 / sm_120，已落盘章）

| 章 | 锚点（摘要） | 明细 |
|---|---|---|
| B-01 | 5090：`misaligned` 0.988× / `float4` 1.038×（相对 aligned；offset=1 近贴齐） | [B-01_global_mem.md](B-01_global_mem.md) |
| B-02 | 5090 列扫：padding 14.12× / swizzle 12.41×（相对 naive） | [B-02_shared_mem.md](B-02_shared_mem.md) |
| B-03 | 5090：highreg 0.308×；regs/occ 三档相同，localB 128→1024 | [B-03_register.md](B-03_register.md) |
| B-04 | 5090：persist 0.999× / thrash 1.001× vs mixed；streaming 19.58× 是合并 | [B-04_l2.md](B-04_l2.md) |
| B-06 | pinned 单向 ~52 GB/s；overlap 贴 pinned | [B-06_pinned_dma_rtx5090.md](B-06_pinned_dma_rtx5090.md) |
| B-07 | 极低 AI `pipe2` ~1.15–1.31×；高 AI → ≤1；`async1` 立刻 wait 可更慢 | [B-07_cp_async_pipeline.md](B-07_cp_async_pipeline.md) |
| B-08 | 立刻 wait 的 TMA 常 ~0.86–1.05×；`pipe2` 低 AI 才明显赚（~1.69× @fma=1） | [B-08_tma.md](B-08_tma.md) |
| B-09 | touch=1 SoA/AoS ~13.6×；touch→8 ~1.80×；tiled/pad ≈ copy 91%/92%，naive ≈ 39%；NCU AoS 32 vs SoA 4 sec/req | [B-09_layout.md](B-09_layout.md) |
| B-05 | fault / prefetch / advise；看 first / median / p95 + 时间线 | [B-05_unified_memory.md](B-05_unified_memory.md) |

## Binary 与主命令

路径前缀：`examples/02_memory_optim/` → 可执行文件名见下表（`build/bin/` 或 CLion `cmake-build-*/bin/`）。

| 章 | 源文件 | Binary | 建议主命令 | Results / Plot |
|---|---|---|---|---|
| B-01 | `01_global_mem_bandwidth.cu` | `02_memory_optim_01_global_mem_bandwidth` | `--mode modes` | `B-01_*`；`plot_b01_global_mem.py` |
| B-02 | `02_shared_mem_bank_conflict.cu` | `02_memory_optim_02_shared_mem_bank_conflict` | `--mode modes` | `B-02_*`；`plot_b02_shared_mem.py` |
| B-03 | `03_register_spill.cu` | `02_memory_optim_03_register_spill` | `--mode modes` | `B-03_*`；`plot_b03_register.py` |
| B-04 | `04_l2_residency.cu` | `02_memory_optim_04_l2_residency` | `--mode modes` | `B-04_*`；`plot_b04_l2.py` |
| B-05 | `05_unified_memory_pf.cu` | `02_memory_optim_05_unified_memory_pf` | profile：`05_profile_unified_memory.sh` | `B-05_*`；`plot_b05_unified_memory.py` |
| B-06 | `06_pinned_dma.cu` | `02_memory_optim_06_pinned_dma` | `modes` / overlap | `B-06_*`；`plot_b06_pinned_dma.py` |
| B-07 | `07_cp_async_pipeline.cu` | `02_memory_optim_07_cp_async_pipeline` | `--mode sweep` | `B-07_*`；`plot_b07_cp_async.py` |
| B-08 | `08_tma_intro.cu` | `02_memory_optim_08_tma_intro` | `--mode sweep`（**sm_90+**） | `B-08_*`；`plot_b08_tma.py` |
| B-09 | `09_layout_transform.cu` | `02_memory_optim_09_layout_transform` | `--mode sweep` | `B-09_*`；`plot_b09_layout.py` |

## 可选旁证脚本（各章已有；本章不新增）

| 章 | 脚本 |
|---|---|
| B-02 | `02_profile_banks.sh`（可选；忽略附着墙钟） |
| B-05 | `05_profile_unified_memory.sh` |
| B-06 | `06_profile_pinned_dma.sh` |
| B-07 | `07_profile_cp_async_pipeline.sh` / `07_dump_sass.sh` |
| B-08 | `08_profile_tma.sh` / `08_dump_sass.sh` |
| B-09 | `09_profile_layout.sh` |

NCU 过滤约定：`--kernel-name-base function|demangled|mangled` + `--kernel-name regex:…`。