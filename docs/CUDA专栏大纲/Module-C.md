# Module C 大纲：核心编程技巧与并发原语（21–30）

> 状态：⏳ 规划中。仓库暂无 `examples/03_*` 代码目录。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

## 模块目标

让「正确性、可维护性、可调优性」成为 CUDA 工程默认配置（而不是靠经验主义）。

## 落地约定

每篇对应一个 `examples/03_compute_primitives/<NN>_*.cu`（目录待建），并至少给一个可被 NCU/NSYS 验证的结论。文章落 `article/03_*`（路径落地时再建）。

## 建议章节主题

| 建议序号 | 主题 | MVP 方向（草稿） |
|---|---|---|
| C-01 | Warp primitives（ballot / shfl / warp reduce-scan） | 对照 naive shared reduce vs warp intrinsic |
| C-02 | Cooperative Groups / Cluster（Hopper+ 可选支线） | 集群协作最小例；无硬件则标不做 |
| C-03 | Atomics 与 contention | global vs shared；分桶 / 分层规约 |
| C-04 | Kernel fusion 代价边界 | fusion vs 多 kernel：寄存器 / occupancy |
| C-05 | CUDA Graph 与 launch overhead | 与 Module E 联动；先测 launch 墙 |

其余 26–30 在开写前按「工程索引型」补边界表与文献池；不提前占坑空目录。

## 与 Module B 的边界

- Module B 收束访存 / 内存体系（含 B-10 Checklist）。
- Module C 从 **并发原语与控制流/launch** 起笔，不重开 coalescing / TMA 教程。

## 开写前交接（从 Module B）

- **入口钩子**：`article/02_memory_optim/B-10*.md` —— 访存已收束；C 不重开 coalescing / TMA / pinned 教程。
- **证据导航**：[`../results/B-10_checklist.md`](../results/B-10_checklist.md)（若性能问题其实是访存层，先回 B）。
- **写稿流程**：`.cursor/skills/aspl-cuda-article/SKILL.md`（与 B-06～B-10 同构：大纲确认 → `.cu` → event median → 正文）。
- **首章建议**：先写分册 `C-01_warp_primitives.md`（边界 / TL;DR / MVP / 文献池），用户确认后再建 `article/03_compute_primitives/` 与 `examples/03_compute_primitives/`——**禁止先建空目录**。
- **编号**：总序号 21–30 ↔ C-01～C-10（规划总表落地时逐行补齐）。
