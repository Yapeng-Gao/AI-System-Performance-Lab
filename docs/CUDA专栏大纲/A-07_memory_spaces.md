# A-07 写作大纲：内存空间与 UVA

> 状态：✅ 正文+示例已落地。本机 RTX 5090：`sm_120`、mapped PASS、`localSizeBytes=1024`（无 `docs/results/` CSV）。
>
> 导航：[`../CUDA专栏规划.md`](../CUDA专栏规划.md)
>
> **已交付**：
> - 正文：`article/01_cuda_basic/A-07. 内存模型全景：UVA、物理拓扑与编译器视角.md`（H1：内存空间：谁在控制这次访问）
> - 示例：`examples/01_cuda_basics/07_memory_spaces.cu`（A/B 必做；C 未做）
> - 图：`assets/A-07-memory-model-cover.png` + `A-07-fig1`…`fig4`
> - **本机要点**：UnifiedAddressing yes；mapped `999` PASS；`force_local_memory_spill` regs=255 / local=1024 B；`ptxas` 报 1024 bytes stack frame、0 spill stores（大数组进 Local ≠ register spill 计数）
>
> **路线**：Module A 概念章（对齐 A-04～A-06）。主证据是「空间对上 + UVA 能解引用 + Local 真的在 spill」；**不是**合并访问 / bank / UM prefetch / pinned DMA 账单。mapped vs device 的 event **可选且不进 TL;DR**。
>
> **硬件门槛**：**不限 sm_90+**。UVA / mapped host 是 64-bit + CC 2.0 起的默认能力；本仓库主测 RTX 5090 / `sm_120`。Ada 仍是 `sm_89`。

**标题（拟定 H1）**：`A-07. 内存空间：谁在控制这次访问`

正文文件名保持现名（CSDN 覆盖）。封面：`assets/A-07-memory-model-cover.png`（重画，按新 TL;DR，不必还原旧密图 / 禁止 `csdnimg`）。

**承接**：A-06 停在工具链。本章回答：这次 load/store **走哪块空间、谁决定、UVA 统一的是什么**。

---

## 与前后章的边界

| 已有章节 | 已覆盖 | A-07 应深化 / 禁止重复 |
|---|---|---|
| A-01 | 寄存器 / SMEM / Global 三层名字 | 把 **逻辑空间 vs 物理位置** 讲完；不再讲异构/异步 launch |
| A-02 | L2 芯片级；SMEM 与 L1 常同一 SRAM；carveout 点名 | 空间课用这张图；**不扫** carveout、不报 L2 GB/s |
| A-04 | Replay / bank 演示 | SMEM 只讲「Block 内、你能布置」；消冲突去 **B-02** |
| A-05 | parameter buffer / `c[0]`；UVA 一句 | 参数空间 **回 A-05**；本章不重讲 ABI |
| A-06 | Fatbin / NVRTC | 不重讲工具链 |
| **B-01** | 合并访问 / sector | **不讲** coalescing 处方；Local 与 Global 一样受合并约束只点名 |
| **B-02** | bank / padding / swizzle | 一句挂钩 |
| **B-03** | spill 分级与 occupancy 处方 | 本章只证明 Local = 逻辑私有、物理在 HBM（`localSizeBytes` / `LDL`）；怎么减 spill 去 B-03 |
| **B-04** | L2 驻留 | 点名 PoC；不测 residency |
| **B-05** | UM fault / prefetch / advise | **UVA ≠ UM**。`cudaMallocManaged` / 缺页迁移 **整章交给 B-05** |
| **B-06** | pinned DMA、overlap、mapped 吞吐 | 本章只证明 **mapped 指针能被 kernel 解引用**；PCIe 账单 / overlap / `--mode mapped` 去 B-06 |
| A-08 | Stream / Event | 若做可选 C：warmup + event median，只比形状；**数字不进 TL;DR**；禁止 Host `chrono` 当 kernel |

---

## TL;DR 目标结论（写作时先写死；有本机打印后再填数）

1. **逻辑空间 ≠ 物理芯片。** Register 在 RF；`__shared__` 在 SM SRAM；Global / Local / Constant 的**数据本体**都可以在板载显存。Local 的「local」是线程私有语义，不是「靠近 ALU 的一块小 SRAM」。
2. **UVA 统一的是 64 位虚拟地址，不是一块物理内存。** Host 与每张 GPU 各有一段 VA；驱动靠指针值判断落点。`cudaMemcpy` 可以 `cudaMemcpyDefault`。**不是** UM：没有自动迁页。
3. **Mapped / Zero-Copy：能解引用 ≠ 该循环访问。** UVA 下 `cudaMallocHost` / `cudaHostAlloc` 常常同一指针就能进 kernel；`cudaHostRegister` / WriteCombined 才要 `cudaHostGetDevicePointer`。示例读一个 `int` 证明通；反复扫大缓冲去 **B-06**，不要写成 HBM。
4. **这次访问谁说了算。** 你能布置的是 Global 指针、`__shared__`、映射哪块 Host。Register vs Local 是编译器；L1/L2 命中是硬件。对自动变量取地址、过大的线程私有数组，会进 Local——打印出的「栈指针」不是片上 SRAM。
5. **口径。** 启动打印 GPU / `sm_XX` / `cudaDevAttrUnifiedAddressing`。主证据只有：mapped 读 PASS；`cudaFuncGetAttributes.localSizeBytes > 0`。**不把 mapped vs device 的 event 比写进 TL;DR**（那是 B-06 的账单）。`__restrict__` 是别名契约，不报「一定出现 `LDG.128` / `LDG.NC`」。禁止 `ncu` 附着 ms。

---

## 建议正文结构

机制节写够（是什么 / 为什么 / 怎么发生）。图按 TL;DR 新画：一张图一个结论。

1. **本章钉什么**：边界表。
2. **逻辑空间课**：Global / Shared / Local / Constant / Register（+ 一行 Texture 名字、一行 Param 回 A-05）。作用域、寿命、物理落点、SASS 线索（`LDG`/`LDS`/`LDL`/`LDC`）。Constant / Texture **只上课、不测**（不加 32-way constant 串行实验）。
3. **谁在控制**：程序员 / 编译器 / 硬件 三列。SMEM+L1 carveout 点名 API，不扫；Hopper 以后更解耦，**不写死 5090 分区比**。
4. **UVA**：同一进程一套 VA、两套（多套）页表。ASCII：Host 指针 vs `cudaMalloc` 指针。打印 `cudaDevAttrUnifiedAddressing`。P2P 只点名，**不测** NVLink。
5. **Mapped Zero-Copy**：与「仅 pinned、给 CE 用」分开（后者 B-06）。工程禁令：kernel 里热循环扫 Host 指针。完整吞吐去 B-06，本章不立 event 账单。
6. **怎么跑**：一条命令；地址图、PASS、`localSizeBytes`。SASS：Linux `grep`、Windows `findstr`（现有 `.sh` 只覆盖 Linux，正文补 Windows，默认不改脚本）。
7. **§7 短出口**（不抢 B-01/B-05/B-06）+ SOP/误区 + 钩子 A-08。
8. **§10 文献**与文献池对齐。

**砍掉旧稿、不搬回来的**：H100 64K 寄存器当本章实测、假 PCIe 64 vs HBM 3000 账单、`csdnimg`、UM 迁页叙事、coalescing/bank 处方、`__restrict__` 保证 `LDG.NC`、P2P 当本地 HBM。

---

## MVP（改现有 `07_memory_spaces.cu`，不新建 binary）

对齐 A-05/A-06：打印 GPU；防 DCE。主证据是正确性打印，不是加速比。

| 编号 | 配置 | 要回答 | 必做？ |
|---|---|---|---|
| A | 地址探测：`cudaMalloc` / `__device__` / `__shared__` / 取地址的自动变量 / mapped Host | 空间不是同一个 VA 段；mapped 能读一个 `int`。取地址 → Local，不要读成片上栈 | **必做** |
| B | 强迫 spill：动态索引的大线程私有数组 | `localSizeBytes > 0`；可选 `cuobjdump` 见 `LDL`/`STL` | **必做** API；SASS 可选 |
| C | 同一 saxpy：device 缓冲 vs mapped Host | 若做：warmup + event median，只证明「mapped 更慢」的形状 | **可选**。数字**不进 TL;DR**；禁止 GB/s / 总线利用率；完整账单去 B-06 |
| D | `__restrict__` 两个 kernel | 契约课；可选 `cuobjdump` | **概念必做、加速比不做**。现有 `.sh` 里「理想出现 `LDG.NC`」不当正文结论 |

**不做**：UM/`cudaMallocManaged` 对照（B-05）；pageable vs pinned memcpy 扫、mapped 吞吐扫（B-06）；合并访问 stride 扫（B-01）；bank padding（B-02）；carveout sweep；多卡 P2P；32-way constant 串行；新的 `*_profile_*.sh`（已有 `07_inspect_sass.sh` 可留，默认不扩、不当 `LDG.NC` 证据）。

**主命令**（一条）：

```bash
cmake --build . --parallel --target 01_cuda_basics_07_memory_spaces
./bin/01_cuda_basics_07_memory_spaces
```

Windows 输出目录见 `examples/01_cuda_basics/README.md`。

**证据最低要求**

- 主：程序自打印（地址图、mapped PASS、`localSizeBytes`）。Module A 概念章 **不强制** `docs/results/` CSV。TL;DR **只填** PASS / `localSizeBytes` / `sm_XX`（用户贴输出后）；C 的比值即使做了也只放「怎么跑」表。
- 旁证：`cuobjdump -sass` 搜 `LDL`/`STL`（Linux `grep` / Windows `findstr`）。不要把 ncu 附着墙钟当结论。
- 写作机无卡：给可粘贴命令，用户一贴就落正文。

---

## 可行性

| 项 | 裁决 | 原因 |
|---|---|---|
| 空间 + UVA + mapped 读 | 必做 | 官方语义；打印 UnifiedAddressing |
| `localSizeBytes` | 必做 | 不靠猜 spill |
| mapped vs device event 形状 | **可选** | 禁令用散文 + 出口 B-06；做了也不进 TL;DR |
| UM / prefetch | 不做 | B-05 |
| 合并 / bank / spill 处方 | 不做 | B-01 / B-02 / B-03 |
| TMA / Cluster DSMEM | 不做 | A-02 点名 + B-08 |
| 默认 NCU 脚本 | 不做 | 用户未要求 |

---

## 参考文献池（与正文 §10 对齐）

| 层 | 条目 | 可引用结论 | 正文用途 |
|---|---|---|---|
| A 官方 | [CUDA Programming Guide — Writing CUDA Kernels / memory types](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | Global/Shared/Local/Constant/Register 的 scope、寿命、Local **物理在 device memory** | §2 空间课 |
| A 官方 | [CUDA Programming Guide — Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html) | 单一 VA；CPU 与每 GPU 各一段；mapped 曾称 zero-copy；UM 是另一套 | §4 UVA；与 B-05 分界 |
| A 官方 | [C/C++ Language Extensions — memory space specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) | `__device__` / `__shared__` / `__constant__` / `__managed__` | 说明符表；`__managed__` 只出口 B-05 |
| A 官方 | [CUDA Runtime — unified addressing / `cudaHostGetDevicePointer`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html) | UVA 下 `cudaHostAlloc`/`cudaMallocHost` 常可直接用同一指针；`cudaHostRegister` / WriteCombined 例外 | §5 mapped |
| A 官方 | CUDA C++ Best Practices Guide — Memory Optimizations | coalescing / bank / L1 配置是**处方索引** | §7 出口，不展开 |
| B 工程 | 本仓库 B-06 TL;DR：mapped 省 launch 不省 PCIe | 热循环扫 Host 退回 memcpy+HBM | §5 禁令 |
| C 实证 | Jia et al., *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*, [arXiv:1804.06826](https://arxiv.org/abs/1804.06826) | 缓存/层次延迟对照；**本仓库数字以本机为准** | 层次数量级，不进假峰值表 |
| C 实证 | Abdelkhalik et al., *Low Overhead Instruction Latency Characterization…*, [arXiv:1905.08778](https://arxiv.org/abs/1905.08778) | Local 与 Global 同类缓存路径 | Local=HBM 旁证 |
| D 扩展 | Mei & Chu, *Dissecting GPU Memory Hierarchy through Microbenchmarking*, IEEE TPDS 2017 / [arXiv:1503.03832](https://arxiv.org/abs/1503.03832) | 历史微基准；架构已变 | §7，不抢 B-04 |
| D 扩展 | RegDem, [arXiv:1907.02894](https://arxiv.org/abs/1907.02894) | 有人把 spill 改倒 SMEM；**不是** nvcc 默认 | 不进 MVP |

---

## 交付 checklist（落地时勾）

- [x] 用户确认本大纲
- [x] 重写 `07_memory_spaces.cu`（GPU / UVA 属性、A/B 必做；C 未做；restrict 不作加速比）
- [x] 改 `examples/01_cuda_basics/README.md` 第 7 章口径
- [x] 正文工程索引 + 本地图（无 `csdnimg`）+ 文首文末 GitHub 绝对链
- [x] 规划表 A-07 → ✅
- [x] 用户贴本机输出后写入 TL;DR（5090：`sm_120` / mapped PASS / `localSizeBytes=1024`）
