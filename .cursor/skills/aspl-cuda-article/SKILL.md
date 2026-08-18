---
name: aspl-cuda-article
description: >-
  Writes AI-System-Performance-Lab CUDA column articles end-to-end (outline with
  required literature research, micro-bench, measured results, markdown, figures)
  or partial edits (cover-only, results-only, NCU/SASS paste, TL;DR, 收口评估).
  Use when starting a chapter (C-01, Module C/D…, B-xx polish), 大纲优化, 写文章, 专栏稿, 调研文献,
  补实测, 补NCU, 评估章节, or following the B-06～B-09 workflow — not for unrelated
  product brainstorming.
---

# ASPL CUDA 专栏写稿

面向本仓库的**工程索引型** CUDA 文章。默认对齐 **B-06～B-10** 已落地范式（B-09 起含 useful-payload / sectors；B-10 为 Checklist 无新 `.cu` 特例）。Module C 起沿用同一证据链，主题换并发原语。

配套： [article-template.md](article-template.md) · [literature.md](literature.md) · 文风 [../aspl-article-voice/SKILL.md](../aspl-article-voice/SKILL.md) · 项目规则 `.cursor/rules/aspl-cuda-column.mdc`  
作者外发备忘（本地、不入库）：`.cursor/publish-csdn-zhihu.md`

## 何时启用

- 开始 `B-0N` / `A-0N` / Module 章、大纲优化、写文章、补实测、封面/原理图
- 用户贴了 sweep / NCU / SASS 输出要落盘
- 收口评估 / 「能不能停」

## 与其他 skill 的关系

本 skill **已含**「大纲 → 用户确认」门禁。写专栏章时：

- **不要**再走完整 product brainstorming（多轮愿景 / Visual Companion / spec 落 `docs/superpowers/specs/`）。
- 澄清时：一次只问一个与本章边界/MVP 相关的问题，然后给大纲。
- 调试 / TDD 等通用 skill 可按需叠加；**流程骨架以本 skill 为准**。

## 先分流：全流程 vs 局部任务

| 用户意图 | 走哪条 |
|---|---|
| 开始新章 / 大纲优化 / 从 0 落地 | **全流程**（下方 0→6） |
| 只贴了实测 / NCU / SASS | **立刻**落 `docs/results/` + 改正文旁证/TL;DR + 必要时 plot；不重写机制章 |
| 只改 TL;DR / SOP / 误区 / §7§10 去重 | 局部改正文；必要时回写规划 |
| 用户说优化 / 收口 / 能发吗 | **改正文**（事实、CTA、外链、文风）；**默认不压 PNG** |
| 用户明确说重画 / 修图 / 封面 | 按本章结论新画 `assets/` + 图注；不必还原旧构图；不把体积当任务 |
| 只修 `.cu` / profile 脚本 bug | 改代码/脚本；若影响口径再改结果节 |
| 只要评估能否发布 | 走 §6 收口清单；能停则停，列出 P0/P1 |

局部任务也先读该章正文与规划对应小节，避免和已写边界冲突。

## 模块路径

| 章节前缀 | article | examples |
|---|---|---|
| A-xx | `article/01_cuda_basic/` | `examples/01_cuda_basics/` |
| B-xx | `article/02_memory_optim/` | `examples/02_memory_optim/` |
| C-xx | `article/03_compute_primitives/`（落地时建） | `examples/03_compute_primitives/`（落地时建；**勿先建空 README**） |
| 更后 Module | 以 `docs/CUDA专栏规划.md` + `docs/CUDA专栏大纲/` 为准 | 同左 |

封面：`article/<module>/assets/<章号>-<topic>-cover.png`；原理图：同目录 `assets/`。

## 硬约束（来自本系列实战）

1. **新章先大纲后实现**：`docs/CUDA专栏大纲/B-0N_<topic>.md` 或 `C-0N_<topic>.md` 写满边界 / TL;DR×5 / 结构 / MVP / 文献池；`docs/CUDA专栏规划.md` 总表加链接；**用户确认后再写代码与正文**。局部任务跳过。
2. **能做就做，做不了就标不做**：MVP 按硬件/API 可行性裁决；不做的写进大纲并说明原因。
3. **默认不写 shell 包装**：主证据一条 binary（如 `--mode sweep`）。**仅当**用户明确要批量 NCU/NSYS/SASS，或多步不可避免时，才加 `0N_profile_*.sh` / `0N_dump_sass.sh`（对齐 B-07/B-08/B-09）。
4. **主证据 = 裸跑 CUDA event median**；NCU/NSYS/SASS 为旁证。**禁止**把 `ncu` 附着时程序自打印的 ms/GB/s 当结论（会被 replay 放大）。
5. **硬件门槛以该章大纲为准**（sm_80+ / sm_90+ / 不限）。写作机可能无目标 GPU；给**可粘贴命令**，用户一贴输出就落盘——不要干等或只口头解读。
6. **不擅自 commit / push**；用户明确要求再提交。Commit 时跳过「仅换行符」噪声文件（常见于 `docs/CUDA专栏大纲/`）。
7. **Commit 前必须同步相关文档**（与代码/正文同批暂存；禁止导航仍写「待测」而产物已齐）。清单见下方 §5.1。
8. **§7 扩展阅读 ≠ §10 参考文献抄两遍**：§7 只留 2～4 条「不抢后续 Module」+ 可选下一章钩子；完整分层目录在 §10；§7 条目用「见 §10-x」去重（见 [article-template.md](article-template.md)）。
9. **文首和文末都要 GitHub `https` 绝对链**（仓库 + Star；有 `.cu` 则 blob；下一篇用 `tree/main/article/<module>/` 目录链）。禁止相对路径 `./A-xx.md`、禁止只写 Star 不给链接。转载 CSDN/知乎这两段不许删。句式见 template「文首 / 文末出链」。
10. **「优化文章」不是压图。** 默认不 pngquant / 不把 PNG 体积当 P1。图只在事实画错、`csdnimg` 裂图、或用户明确要求重画时才动像素。重画按正文结论新画，不必还原旧图。

## 工作流（新章全流程 0→6）

```text
- [ ] 0. 读上下文
- [ ] 1. 大纲 + 文献池 → 用户确认；导航总表加行
- [ ] 2. MVP 代码 + CMake 可见
- [ ] 3. 跑通实测 / 收 CSV（写作机无卡则给命令等用户贴）
- [ ] 4. 正文 + 封面/原理图 + plot
- [ ] 5. 回填规划状态 / README / examples README
- [ ] 6. 收口评估（能停则停）
```

### 0. 读上下文

必读：

- `docs/CUDA专栏规划.md`（该章行 + §5）+ 相邻章大纲
- 上一章正文「下一章钩子」与边界表
- 同 module 最近一章 `0N_*.cu`（CLI、计时、`--mode`）
- `docs/仓库架构与现状.md`

弄清：**本章深化什么 / 严禁重复什么 / 钩子给谁 / 硬件门槛**。

### 1. 大纲 + 文献

新建大纲（对齐 B-07/B-08/B-09 分册）并在导航总表加链接。必含：标题、边界表、TL;DR×5、正文结构、MVP 可行性、最小实验表、证据最低要求、参考文献池。

默认写作路线：**Microbench-first**（与已落地章同构）。等用户确认后再实现。

文献调研规则见 [literature.md](literature.md)（大纲阶段**不可跳过**）。

### 2. MVP 代码

| 产物 | 路径 |
|---|---|
| 示例 | `examples/<module>/0N_<topic>.cu`（CMake GLOB，**增删后重跑 cmake**） |
| 结果 | `docs/results/B-0N_*.md` + `.csv` |
| 绘图 | `scripts/plot_b0N_*.py` → `article/.../assets/` |
| 正文 | `article/<module>/B-0N.*.md` |

代码习惯：

- `--mode` 多配置；`sweep` CSV 作主结论载体
- 启动打印 GPU 名 / `sm_XX`；CC 不够则清晰退出
- 防 DCE：结果写回 device 可见存储
- 尾块 / 越界：tile 循环 clamp
- 计时：warmup + 多次 run → **median**
- 布局类：带宽用 **useful payload**（触达字段字节），禁止整 struct 赋值冒充「单字段跨步」

### 3. 实测（含旁证）

- 主命令一条；编译门槛写清（5090 常用 `CMAKE_CUDA_ARCHITECTURES=120`，非每章必需）
- **用户一贴** sweep/表/NCU/SASS：立刻写入 `docs/results/` + 更新正文对应节与 TL;DR；有 CSV 则跑 plot
- 绝对 GB/s 可能含 **L2**：正文必须写「主看加速比/相对形状，勿当总线利用率」（见 B-09）

#### 3.1 NCU / SASS 脚本约定（用户明确要求时）

对齐 `07_profile_*` / `08_profile_tma` / `09_profile_layout`：

- 支持 `DO_NCU=1` 与 `ncu-only`（只跑旁证、不重跑全 sweep）
- **`--kernel-name-base` 只能是** `function` | `demangled` | `mangled`；过滤名用 `--kernel-name regex:kernel_foo`
- SASS dump：按 **`.text._Z…kernel_…` 段**截取；证明路径用 `UTMALDG` / `LDGSTS` / `ELECT` / `EIATTR_MBARRIER` 等，**不要**用「文件里出现 LDG+STS」误判「没走 TMA」（sync 对照与 1D 降级都可能有 LDG+STS）
- SASS/NCU **不单独报加速比**；与裸跑 event 互证
- `*.ncu-rep` / `*.nsys-rep` 进 `.gitignore`

### 4. 正文

骨架见 [article-template.md](article-template.md)。要点：

- **标题 `#` 占正文第一行**，然后封面 + 承接上章引用块
- **文首 + 文末 CTA** 都是 GitHub 绝对链（硬约束 9；template 固定句）。改旧稿时缺哪补哪
- **TL;DR 带本机数字**
- 原理：短 ASCII 或 1～2 张原理图；图注中文；图用本地 `assets/`，禁止 `csdnimg`
- 决策表 / SOP / 误区 / 钩子
- §7 与 §10 **去重**（硬约束 8）
- 实测：表 + 图；口径写明 median；有 NCU 则 §5.x 旁证小节
- **文风 pass**：散文段走 `.cursor/skills/aspl-article-voice/SKILL.md`（去套话；**不许**拆表/TL;DR/命令/文首文末 CTA）

封面风格：深色底、青/琥珀；忌紫光堆徽章。设备内拷贝图例标 **GMEM→SMEM**（勿写成 Host CE / H2D）。

**重画按本章结论新画，不必还原旧图。** 旧 CSDN / 旧模型图常密、字小、结构乱；新图以正文 TL;DR 为准：一张图一个结论、少字、大号标签、大留白。禁止复刻旧密图（全 32 格编号、双语脚注、周期表、多列白皮书对照、徽章堆）。封面 ≤4 个标签；原理图像教材插图，不像 infographic 海报。用户说「优化」默认不压 PNG；只有明确说重画/修图才动像素。

### 5. 仓库状态回填

章落地或用户要求 commit 前，按 §5.1 核完再 `git add`。

### 5.1 Commit 前文档同步清单

| 文件 | 更新什么 |
|---|---|
| `docs/CUDA专栏规划.md` | 该章状态 ✅/🟡 + 大纲链接；§4.2「当前焦点」指下一章 |
| `docs/CUDA专栏大纲/<章>.md` | 顶部状态、本机要点、交付 checklist 勾选 |
| `docs/CUDA专栏大纲/Module-*.md` | 章节总表该行状态 |
| `docs/CUDA专栏大纲/README.md` | 分册表状态行 |
| 根 `README.md` | 主线一句、专栏映射、运行/plot 示例（若有新 binary） |
| `examples/<module>/README.md` | 新 `.cu` / 主命令 / results 路径 |
| `docs/仓库架构与现状.md` | 目录树、进度表、焦点句（过时则改） |
| `docs/results/<章>_*` | 与正文 TL;DR/实测表数字一致（有测才交） |

只改图注/错字等局部任务：可只更新直接相关文件，但**不得**让规划总表与真实进度矛盾。

### 6. 收口

| 检查项 | 通过标准 |
|---|---|
| 结论有数 | TL;DR / 实测表含本机数字或明确「待测」 |
| 边界清晰 | 不重讲上章；不做的在大纲/边界写明 |
| 主命令一条 | binary `--mode sweep`（或章内等价） |
| 脚本 | 无用户未要的 profile 壳；要了则 `ncu-only` 可用 |
| 口径 | 未把 ncu 附着 ms 当结论；带宽章有 L2/useful 提醒 |
| 文首文末 CTA | 仓库 `https` + Star；有示例则 `.cu` blob；下一篇为 module **目录**绝对链 |
| 图 | 本地 `assets/`；无 `csdnimg`。**不要**把 PNG 体积/压缩列为 P0/P1 |
| §7§10 | 无大段重复链接 |
| 旁证（若做了） | 已落 `docs/results/` + 正文小节 |
| 导航同步 | §5.1 清单与产物进度一致（用户要 commit 时一并纳入） |

**能停则停**。P1（NCU/SASS/图注错字）不阻塞发布，除非用户要求补。PNG 体积不是 P1。

评估时对比相邻已发布章；区分「发布门槛」与「锦上添花」。

## 反模式

- 未确认大纲就写长文；大纲跳过文献调研
- 文献堆链接却无一条进 TL;DR/决策表；凭记忆编 DOI/arXiv
- 局部任务却重开全流程
- 默认加 profile shell；或 `--kernel-name-base kernel_foo`（非法）
- Host CE / 设备 `memcpy_async` / TMA bulk 混成一层
- 重讲上章；边界表形同虚设
- 无实测写死绝对加速比；用户已贴数却不落盘
- 默认每章都要 sm_90+ / TMA
- 用 ncu 附着时墙钟 ms 写进 TL;DR
- 宣称「换 TMA/SoA 就少指令」却无 `inst_executed` / sectors 证据
- §7 与 §10 整页重复
- **只 commit 正文/`.cu`，规划/README/Module 表仍写「待测」**
- 过度美化拖延下一章；被 brainstorming 带去写无关 product spec
- 为去 AI 味拆掉决策表 / TL;DR；或灌「说真的」「太离谱了」公众号腔
- 文末只有「请 Star」没有仓库/`cu`/下一篇绝对链；下一篇写成 `./A-xx.md`
- 把「优化文章」做成 pngquant / 压缩体积；未要求就重画图

## 快速启动话术

新章：

> 按 aspl-cuda-article skill 开始 C-01：先读规划、Module-C.md 与 B-10 钩子，出 C-01 大纲+文献，确认后再实现（目录随首章落地再建，勿空壳）。

局部（贴数）：

> 按 aspl-cuda-article：根据下面 NCU/sweep 输出更新 B-0N 实测与旁证节。

收口：

> 按 aspl-cuda-article §6 评估 B-08/B-09 能否停。
