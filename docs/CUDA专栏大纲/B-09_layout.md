# B-09 写作大纲：数据布局（AoS/SoA/Transpose）

> 状态：✅ 文章+示例+RTX 5090 实测已落地。对照审稿用；以正文与 `docs/results/B-09_*` 为准。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)

> **已交付**：正文 `article/02_memory_optim/B-09*.md`；示例 `09_layout_transform.cu`；封面/原理图/实测图；`scripts/plot_b09_layout.py`；`docs/results/B-09_layout.md` + CSV。  
> **本机要点**：touch=1 SoA/AoS ≈ **13.6×**；touch→8 收窄至 **~1.8×**；tiled/pad ≈ copy 的 **91%/92%**，naive ≈ **39%**。
>
> **路线**：Microbench-first——`--mode sweep`（触达字段扫）+ 固定 transpose modes；NCU 可选旁证。
>
> **硬件门槛**：**不限 sm_90+**（布局/合并是全架构问题；系列常用 RTX 5090 实测，不依赖 TMA）。
>
> **相对初稿的收紧**：
> 1. `copy`、`transpose_pad` 从可选升为 **必做**（transpose 需要上限参照；pad 一句挂钩 B-02，成本极低）。
> 2. `sweep` **只扫 touch_fields∈{1,2,4,8}**（AoS vs SoA），不扫 N——文献形状②是「触达字段比例」，不是问题规模。
> 3. 有效带宽口径写死为 **useful payload**（读写的字段字节），避免 AoS「搬了很多废字节却报高 GB/s」误导。

**标题**：`B-09. 数据布局（AoS/SoA/Transpose）：一次布局调整带来的事务变化`

**与前后章的边界**

| 已有章节 | 已覆盖 | B-09 应深化 / 避免重复 |
|---|---|---|
| B-01 | Sector / coalescing 物理、向量化、TMA 叙事钩子 | **不重讲**「什么是合并」教程；做成 **AoS↔SoA / transpose 可复现决策表 + 有效带宽数字** |
| B-02 | SMEM bank / padding / swizzle | transpose 的 `tile[T][T+1]` 只 **一句话挂钩** B-02；不重做 bank 全家桶 |
| B-07 / B-08 | async copy / TMA pipeline | 承接 B-08 钩子：「引擎再强也救不了选错布局」；**不**再扫 AI/intensity、不写 TMA descriptor |
| B-10（规划） | Module B Checklist | 本章只交付「布局症状 → 处方」条目原料；汇总表留给 B-10 |
| Module D | GEMM / Tensor Core 布局 | CuTe/CUTLASS layout / WGMMA swizzle **扩展阅读 only** |

**TL;DR 目标结论（写作时先写死；有实测后改成带本机数字）**

1. **布局决定事务形状**：同一逻辑读 1 个 `float` 字段，AoS 常把 warp 打散到多 sector；SoA 让相邻线程读相邻地址 → 少事务、高有效带宽（Best Practices：CC≥6.0 按 32B sector 计数）。
2. **该换 SoA**：kernel **按字段**扫大批量记录、且每次只用 struct 中少量字段（粒子/顶点属性/instance 状态）；文献与工程 microbench 常见 **数倍～数十倍** 量级差（文献数，非本机绝对数）。
3. **别盲目全 SoA**：若热路径几乎总是整条记录一起读写、或主机侧/API 强绑 AoS，先估转换成本；可选 AoSoA/blocked 作折中，**以 sweep 为准**。
4. **Transpose 是「布局变换算子」**：naive GMEM 跨步写 ≈ 带宽自杀；标准处方是 **SMEM tile 重排**（读合写合），padding 消 bank 冲突后可逼近 copy 带宽（Harris NVIDIA blog）。
5. **判停**：先看 CUDA event **有效 GB/s（或 SoA/AoS、tiled/naive 加速比）**；有 NCU 再看 `sectors/request`（理想 float 合并约 **4**；明显偏高先修布局，见 NCU Triage）。

**建议正文结构（8～9 节）**

1. **问题定义**：B-08 之后回到「第一天布局」——对照表：合并物理（B-01）已会，本章回答 **换哪种布局、何时付 transpose 成本**。
2. **物理模型（短）**：AoS stride = `sizeof(Struct)` vs SoA stride = `sizeof(field)`；ASCII/1 张原理图：warp 读 `x` 字段时 sector 覆盖差。
3. **决策表**：何时 SoA / 何时保留 AoS / 何时一次性 transpose（或算子内 on-the-fly tile transpose）。
4. **Transpose 处方**：naive vs tiled+SMEM（+ pad→B-02）；与「永久改 SoA」的成本对比。
5. **MVP 实验矩阵** + 主命令 `--mode sweep`。
6. **实测**：表 + 曲线；口径 CUDA event median。
7. **旁证（可选）**：一组 NCU `l1tex__average_t_sectors_per_request_*` / DRAM 吞吐。
8. **扩展阅读**：AoSoA、编译器 AoS→SoA、CuTe transpose；钩子 → B-10 Checklist。
9. **误区 + SOP + 下一章钩子**。

**写作路线（2～3 条；默认推荐 #1）**

| # | 路线 | 取舍 |
|---|---|---|
| **1（推荐）** | **Microbench-first**：AoS/SoA + naive/tiled transpose，`--mode sweep` CSV 主结论 | 与已落地章同构；实现成本可控 |
| 2 | NCU-first：先固定 sectors/request 再讲故事 | 旁证强，但写作机/环境依赖大；作补强勿作唯一门禁 |
| 3 | 教程重写 Harris transpose + 概念 AoS | 易与 B-01/公开博客重复；**不推荐**作主线 |

**MVP 可行性评估**

| 编号 | 配置 | 可行性 | 本章裁决 |
|---|---|---|---|
| A | AoS：宽 struct（8×float），按字段读写前 `touch_fields` 个 | ✅ | **必做** |
| B | SoA：同逻辑、同写回契约 | ✅ | **必做（对照）** |
| C | `touch_fields∈{1,2,4,8}` 扫（useful GB/s + SoA/AoS 加速比） | ✅ | **必做**（主结论载体） |
| D | Transpose：naive vs tiled SMEM | ✅ | **必做** |
| E | Tiled + `TILE+1` padding + **copy 上限** | ✅ | **必做**（pad→B-02；copy=天花板） |
| F | NCU sectors/request（A vs B，naive vs tiled） | ⚪ 依赖 `ncu` | **可选旁证** |
| — | 生产粒子系统 / CUTLASS layout algebra / 自动 AoS→SoA | ❌ | **不做** |

**最小可复现实验（`09_layout_transform.cu`）**

| mode | 要回答的问题 |
|---|---|
| `aos` / `soa` | 固定 `touch_fields`：useful GB/s 与时延？ |
| `copy` | 矩阵 copy 带宽上限（transpose 参照） |
| `transpose_naive` | 纯跨步写的带宽地板？ |
| `transpose_tiled` / `transpose_pad` | SMEM 重排（±pad）能否接近 copy？ |
| `sweep` | 扫 `touch_fields`：SoA/AoS 加速比何时收窄？ |
| `modes` | 一次打印 layout + transpose 全表（写结果用） |

**证据最低要求**

- 主证据：CUDA event **median** 时延 → 有效带宽（GB/s）与加速比；`sweep` CSV → `docs/results/B-09_*.csv` + 摘要 md。
- 旁证（可选）：至少一组 A vs B 的 NCU sectors/request（或 Memory Workload 表）；**禁止**把 ncu 附着时程序自打印 ms 当结论。
- 启动打印 GPU 名 / `sm_XX`；本章 **无** sm_90+ 硬门槛。
- 默认 **不加** profile shell；用户明确要批量 NCU 再补。

**本机要验证的「文献形状」假设（1～3）**

1. **少数字段热读**：SoA 有效带宽 / 时延显著优于 AoS（预期数倍级；不追求博客里的 ~28× 绝对数）。
2. **触达字段比例↑**：AoS vs SoA 差距**通常收窄**（同 sector / L2 复用；不要求加速比→1）。
3. **Transpose**：`transpose_tiled` ≫ `transpose_naive`；pad 接近或略优于 tiled；计时前正确性检查必过。

**参考文献池（与正文参考文献节对齐）**

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA C++ Best Practices Guide — Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | CC≥6.0：warp 访问合并为覆盖所需地址的 **32B sector 事务数**；跨步↑ → 有效带宽↓直至每线程一 sector | §2 物理；TL;DR① |
| A 官方 | [CUDA Programming Guide — Writing CUDA SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)（Coalesced Global Memory） | 相邻线程读相邻元素是典型合并；目标最大化 **bytes used / bytes transferred** | §2；决策表措辞 |
| A 工具 | [Nsight Compute — Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)（Memory） | Global load **sectors/request** 明显高于理想（常见 float 合并约 4）→ 先修 coalescing/layout | §5 旁证；TL;DR⑤ |
| B 工程 | Mark Harris, [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) | naive 跨步写远低于 copy；SMEM tile 重排 + pad 消 bank 后可逼近 copy 带宽 | §4 Transpose 处方；MVP D/E |
| B 工程 | Colfax, [Tutorial: Matrix Transpose in CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/) | 把跨步留在 SMEM、GMEM 两侧保持合并；CuTe 抽象对应同一物理处方 | §7 扩展；不抢 Module D |
| B 工程 | Semih Güreşçi, [GPU Memory: AoS vs SoA](https://semihguresci.com/blog/gpu-memory-access-benchmark-experiment-06-aos-vs-soa/) | 同逻辑粒子更新、只换布局：SoA 可大幅提高有效带宽（文称可达数十倍级，**文献数**） | §3 决策表「形状」；本机用 sweep 校准 |
| C 实证 | Springer / Wußing et al. 相关工作链：AoS→SoA + views（[arXiv:2405.12507](https://arxiv.org/abs/2405.12507)；扩展 [arXiv:2512.05516](https://arxiv.org/abs/2512.05516)） | 转换有成本；**视图/局部转换**可避免「全量永久 SoA」惩罚；收益随 kernel AI / 粒子规模变化 | TL;DR③；§3「别盲目」 |
| C 实证 | Kjolstad / 生态常见结论亦可借 **DynaSOAr**（[arXiv:1810.11765](https://arxiv.org/abs/1810.11765)）机制段 | SoA 同时改善 coalescing 与「未触达字段不占 cache line」 | §2 机制一句 |
| C 实证 | Eberhardt et al., **GPUDrano**（CAV’17） | 真实应用中 AoS、按行错误映射等是高频 uncoalesced 根因 | §8 误区清单 |
| D 前沿 | Annotation-guided AoS→SoA + GPU offload（[CPE 2025 / DOI:10.1002/cpe.70199](https://doi.org/10.1002/cpe.70199)）及 [arXiv:2512.05516](https://arxiv.org/abs/2512.05516) | 编译器/注解驱动布局变换是趋势；**不写进必做 MVP** | §7 扩展阅读 |

**进 TL;DR / 决策表 vs 仅扩展阅读**

| 结论 | 去向 |
|---|---|
| 32B sector 合并计数；少数字段 → SoA | TL;DR / 决策表 |
| tiled transpose ≈ copy 量级（文献形状） | TL;DR / MVP |
| sectors/request 偏高先修布局 | TL;DR⑤ / 可选 NCU |
| 编译器自动 AoS→SoA、CuTe/CUTLASS transpose | **仅** §扩展阅读 |
| 博客极端 28× | 只作「形状」提示，正文写「本机 sweep」 |
