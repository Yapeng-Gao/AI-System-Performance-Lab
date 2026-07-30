# Module E 大纲：深度学习工程实战与系统集成（41–50）

> 状态：⏳ 规划中。目录/代码尚未落地（曾删除占位 `python/` 等）。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

## 模块目标

把「kernel 优化」放回真实工程链路：Python → C++ 扩展 → profiler → benchmark → 部署形态。

## 与仓库现状对齐

总规划里曾出现 `examples/05_dl_engineering/...` / Python 绑定等完整树；**当前仓库已删除占位**。对 41–50 明确：

- **规划中：目录/代码尚未落地**
- 每落地一篇，再新建对应路径（文章 / 示例 / 脚本）
- 不提前重建空 `python/` / `include/` 壳

## 建议主题骨架

| 主题 | 说明 |
|---|---|
| pybind / 扩展打包 | 最小可调用自定义 kernel |
| Benchmark 规范 | 与章节 micro-bench 口径统一（median / 防 DCE） |
| Profiler 工作流 | NSYS/NCU 在端到端里的位置 |
| 部署形态 | 静态库 / 插件 / 与框架集成钩子 |

开写第一章时再拆独立 `E-0N_*.md` 大纲文件。
