# CUDA 专栏大纲（分册）

本目录存放**按章/按模块的写作大纲**。进度总表与文档地图见 [`../CUDA专栏规划.md`](../CUDA专栏规划.md)。

| 文件 | 状态 | 说明 |
|---|---|---|
| [B-07_cp_async.md](B-07_cp_async.md) | ✅ | 已落地；**新章样板**（对照审稿） |
| [B-08_tma.md](B-08_tma.md) | ✅ | 已落地；**新章样板**（对照审稿） |
| [B-09_layout.md](B-09_layout.md) | ✅ | 已落地；**新章样板**（含 RTX 5090 实测） |
| [Module-C.md](Module-C.md) | 🟡 | 进行中：C-01～C-04 ✅；下一章 C-05 |
| [C-01_warp_primitives.md](C-01_warp_primitives.md) | ✅ | 正文+示例+5090 实测+NCU |
| [C-02_cooperative_groups.md](C-02_cooperative_groups.md) | ✅ | 正文+示例+5090 实测（抽象税≈0；>32 悬崖） |
| [C-03_atomics_contention.md](C-03_atomics_contention.md) | ✅ | 正文+示例+5090 实测（smem~6.3×；agg≈naive） |
| [C-04_sync_layers.md](C-04_sync_layers.md) | ✅ | 正文+示例+5090 实测（grid/block~17×） |
| [Module-D.md](Module-D.md) | ⏳ | 算子实现（远期） |
| [Module-E.md](Module-E.md) | ⏳ | DL 工程集成（远期） |
| [archive/](archive/README.md) | 📦 | 已发布 Module B 大纲归档（B-06 / B-10） |

**约定**：新章大纲新建 `B-0N_<topic>.md`（或 `C-0N_*.md`），在导航总表加一行链接；不要把长大纲塞回 `CUDA专栏规划.md`。  
**已落地章**：权威以正文 + `docs/results/` 为准；样板优先看 B-07～B-09，其余可进 `archive/`。