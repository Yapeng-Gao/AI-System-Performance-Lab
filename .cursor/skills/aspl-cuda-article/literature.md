# 文献池写法（ASPL 专栏）

大纲阶段在 `docs/CUDA专栏大纲/B-0N_<topic>.md` 末尾放「参考文献池」。正文 §10 与之对齐。

## 推荐表格式

```markdown
**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA PG — …](url) | … | §3 API / 对齐 |
| B 工程 | Colfax / NVIDIA blog … | … | §3～4 写法与坑 |
| C 实证 | Author et al., Venue’YY / arXiv:…. | 低 AI 约 X×，高 AI →1 | §4 决策表；§5 预期曲线 |
| D 前沿 | …（近 1～2 年） | … | §7 扩展；不写进必做 MVP |
```

## 查询示例（按章替换关键词）

```text
# 官方
<topic> CUDA programming guide asynchronous
<topic> site:docs.nvidia.com

# 工程
<topic> Colfax OR "NVIDIA blog" tutorial performance

# 实证 / 前沿
<topic> GPU microbenchmark arxiv 2023 OR 2024 OR 2025
<topic> site:arxiv.org <architecture feature>
```

## B-08 示例（形态参考，勿照搬条目）

| 层 | 例子 | 用途 |
|---|---|---|
| A | CUDA Async Copies、Hopper In-Depth、CCCL TMA | API / mbarrier |
| B | Colfax Mastering TMA、PyTorch TMA FP8 | 写法；descriptor 开销反例 |
| C | Luo et al. arXiv:2501.12084（~+170 cycle） | 「立刻 wait 可能不赚」 |
| D | FA3、ACTA、Cypress | 扩展阅读，不抢 Module D |

## 写入正文时

- TL;DR / 决策表：优先 C 的曲线形状 + 本机验证句。
- 机制节：A 定语义；B 补操作细节。
- **§10 参考文献**：完整分层编号；凡支撑 TL;DR/决策表的条目必须在此。
- **§7 扩展阅读**：只放 D +「不抢后续 Module」的出口；已在 §10 的条目写「见 §10-x」，**禁止整表重复**。
- 凡写「约 N× / +N cycle」：标明 **文献** 还是 **本机**。
