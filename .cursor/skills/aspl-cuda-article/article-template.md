# 专栏正文模板（ASPL）

与 `docs/CUDA专栏规划.md` §5 固定结构一致；细节以 B-07～B-09 成稿与 `docs/CUDA专栏大纲/` 为准。全景 / 教法弧 / 展开权以规划 §2–§4 为准，未说优化全景不得改线。

## 文件名

- 正文：`article/<module>/B-0N. <工程索引型标题>.md`
- 封面：`assets/B-0N-<topic>-cover.png`
- 原理图：`assets/B-0N-<topic>.png`（如 `B-08-cpasync-vs-tma.png`）

## 推荐目录

```markdown
# B-0N. <工程索引型标题>

![封面](assets/B-0N-...-cover.png)

> 上章结论一句话 → 本章卡点一句话 → 本章交付一句话。

**配套可复现**：[Yapeng-Gao/AI-System-Performance-Lab](https://github.com/Yapeng-Gao/AI-System-Performance-Lab)（文章 + `.cu` + 实测表）。有用请 Star。本章示例：[`examples/<module>/0N_<topic>.cu`](https://github.com/Yapeng-Gao/AI-System-Performance-Lab/blob/main/examples/<module>/0N_<topic>.cu)。

## TL;DR（工程结论）
1～5 条；有实测后写入本机数字与判停条件。

## 1. 问题
边界对照表（与上章/Host/Device 分层）。必要时插原理对照图。

## 2. 物理 / 模型
把本章概念讲完（是什么、为什么、怎么发生、错读），再用短 ASCII 或原理图钉住。不要只留表和图注。

## 3. API / 机制分层
由浅到深；只写本章必用路径。（布局类章可改为「决策表」前置，见 B-09）

## 4. 决策表
信号 → 建议（该上 / 别上 / 回退哪章）。

## 5. 实验怎么设计
- 代码路径
- **一条主命令**
- mode 表
- 证据优先级（event 主；NCU/SASS 旁证）
- ### 5.1 本机实测：平台口径 + 表 + 图 +「怎么读」
- ### 5.x NCU/SASS 旁证（若已跑）：只写 metric，禁止 ncu 附着墙钟

## 6. 工程边界
对齐、UB、与相邻章关系、硬件门槛。

## 7. 扩展阅读（不抢后续 Module）
2～4 条「还想往哪逛」+ 可选下一章钩子；**禁止与 §10 大段重复**。
条目若已在 §10：写「见 §10-x」+ 一句边界说明即可。

## 8. SOP + 误区
可执行步骤 + 判停 + 高频坑。

## 9. 小结与下一章
三句话收敛 + 钩子（下一篇用 GitHub **目录**绝对链，见下方出链规则）。

---

> 本文配套代码与实测：[AI-System-Performance-Lab](https://github.com/Yapeng-Gao/AI-System-Performance-Lab)。觉得有用请 Star，后续章更新更好找。  
> 本章示例：[`examples/<module>/0N_<topic>.cu`](https://github.com/Yapeng-Gao/AI-System-Performance-Lab/blob/main/examples/<module>/0N_<topic>.cu)。下一篇：[X-0N 短标题](https://github.com/Yapeng-Gao/AI-System-Performance-Lab/tree/main/article/<module>/)。

## 10. 参考文献
官方 / 工程 / 实证 / 前沿 分层编号；与大纲文献池对齐。
支撑 TL;DR/决策表的条目必须出现在此节。
```

## 文首 / 文末出链（CSDN 同步硬规则）

站外相对路径必裂。**文首和文末都要有 GitHub `https` 绝对链**，不要只写 Star 口号。

| 位置 | 必有 | 有 `.cu` 时 | 下一篇 |
|---|---|---|---|
| 文首「配套可复现」 | 仓库根 + Star | `blob/main/examples/.../*.cu` | 导读/无示例章：改写「下一篇」目录链 |
| 文末（§9 后、§10 前） | 仓库根 + Star | 同上 `.cu` | `tree/main/article/<module>/`（**目录**，不要 `./A-02.md`，不要中文文件名 blob） |

无 `.cu` 的章（导读、Checklist）：两处都用「下一篇」目录链代替「本章示例」。转载时这两段绝对链不许删。

## 优化什么、不优化什么

用户说「优化 / 收口 / 能发吗」= **改正文**：事实、边界、CTA、外链、文风。  
**默认不压 PNG、不把体积当 P1。** 图只在事实画错、裂图（`csdnimg`）、或用户明确说重画/修图时才动 `assets/` 像素。重画时按**本章正文结论**新画（简单、少字、大留白），不必还原旧 CSDN / 旧模型图的构图。  
**机制节写够本章概念。** 砍套话和抢邻章的课，不砍「是什么 / 为什么 / 怎么发生」。TL;DR 和边界表不能替代讲解。

## §7 vs §10（去重硬规则）

| | §7 扩展阅读 | §10 参考文献 |
|---|---|---|
| 目的 | 读完本章后的出口（不抢下章/Module） | 引用目录与可核查来源 |
| 篇幅 | 短（≤4 条正文链接） | 完整分层 |
| 禁止 | 把 §10 整表再抄一遍 | 只有链接、无一进入 TL;DR/决策表 |

## 结果落盘

- `docs/results/B-0N_<topic>.md`：平台、完整表、怎么读、复现命令、**旁证表（若有）**
- `docs/results/B-0N_sweep.csv` / `B-0N_modes.csv`（若有）
- `scripts/plot_b0N_*.py` 读 CSV → `article/.../assets/`

### 旁证落盘最小字段

**NCU**：mode、关键 metric（如 sectors/request 或主导 stall + Mem Throughput）、一句「怎么读」；写明忽略附着时程序 ms。  
**SASS**：命中指令名（如 `UTMALDG.2D` / `LDGSTS` / `ELECT`）+ 含义；不报加速比。

## 示例代码头注释

说明：modes、硬件门槛（如 sm_80+ / sm_90+）、对齐约束、主证据是什么、useful-payload 口径（若适用）。
