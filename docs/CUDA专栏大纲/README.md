# CUDA 专栏大纲（分册）

本目录存放 **L1 模块地图** 与 **按章写作大纲**。L0 全景与整体布局见 [`../CUDA专栏规划.md`](../CUDA专栏规划.md) §2–§4。

## 模块地图（L1）

| 文件 | 状态 | 教法弧 |
|---|---|---|
| [Module-A.md](Module-A.md) | ✅ | 模型 → … → Roofline |
| [Module-B.md](Module-B.md) | ✅ | 形状 → on-chip → 跨空间 → 引擎 → 布局 → 索引 |
| [Module-C.md](Module-C.md) | 🟡 | 通信 → 争用/同步 → 摊销 launch；（后半可裁） |
| [Module-D.md](Module-D.md) | ⏳ | 规约族 → 矩阵 → epilogue |
| [Module-E.md](Module-E.md) | ⏳ | 可调用 → 可测量 → 可交付 → 多卡钩子 |

## 按章大纲

| 文件 | 状态 | 说明 |
|---|---|---|
| [A-07_memory_spaces.md](A-07_memory_spaces.md) | ✅ | Module A：空间 / UVA / mapped |
| [A-08_async_stream.md](A-08_async_stream.md) | ✅ | Module A：Stream / Event / 流水线 |
| [A-09_sanitizer.md](A-09_sanitizer.md) | ✅ | Module A：Sanitizer |
| [A-10_roofline.md](A-10_roofline.md) | ✅ | Module A：Roofline |
| [B-01_global_mem.md](B-01_global_mem.md) | ✅ | 形状弧首章；5090 `0.988×` / `1.038×` |
| [B-02_shared_mem.md](B-02_shared_mem.md) | ✅ | on-chip：Bank / Padding / Swizzle；5090 padding 14.12× / swizzle 12.41× |
| [B-03_register.md](B-03_register.md) | ✅ | on-chip：local 足迹 0.308×；regs/occ 三档相同 |
| [B-04_l2.md](B-04_l2.md) | ✅ | on-chip：persist 0.999× / thrash 1.001×；streaming 19.58× 是合并 |
| [B-07_cp_async.md](B-07_cp_async.md) | ✅ | 已落地；**新章样板** |
| [B-08_tma.md](B-08_tma.md) | ✅ | 已落地；**新章样板** |
| [B-09_layout.md](B-09_layout.md) | ✅ | 已落地；**新章样板** |
| [C-01_warp_primitives.md](C-01_warp_primitives.md) | ✅ | 正文+示例+5090 实测+NCU |
| [C-02_cooperative_groups.md](C-02_cooperative_groups.md) | ✅ | 正文+示例+5090 实测 |
| [C-03_atomics_contention.md](C-03_atomics_contention.md) | ✅ | 正文+示例+5090 实测 |
| [C-04_sync_layers.md](C-04_sync_layers.md) | ✅ | 正文+示例+5090 实测 |
| [C-05_kernel_fusion.md](C-05_kernel_fusion.md) | ✅ | 正文+示例+5090 实测 |
| [C-06_cuda_graph.md](C-06_cuda_graph.md) | ✅ | 正文+示例+5090 实测 |
| [C-07_persistent.md](C-07_persistent.md) | ✅ | Persistent；5090 sweep 32×→3740×，work=4096→314× |
| [archive/](archive/README.md) | 📦 | 已发布 Module B 大纲归档（B-06 / B-10） |

**约定**：新章大纲新建 `A-0N_*` / `B-0N_*` / `C-0N_*.md`，在导航总表加一行链接；不要把长大纲塞回 `CUDA专栏规划.md`。  
**已落地章**：权威以正文 + `docs/results/` 为准；样板优先看 B-07～B-09，其余可进 `archive/`。
