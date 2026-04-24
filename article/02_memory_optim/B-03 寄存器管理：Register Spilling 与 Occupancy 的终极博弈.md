# 13. 寄存器管理：Register Spilling 与 Occupancy 的终极博弈

GPU 编程中常出现一个反直觉的现象：你为了让逻辑更清晰，展开了一个循环或提取了几个中间变量，逻辑没变，性能却突然暴跌 50%。打开 Nsight Compute，显存带宽未满，Shared Memory 也未见冲突。此时，真正的瓶颈往往隐藏在一行不起眼的编译日志里——**Register Spilling（寄存器溢出）**。

寄存器是 GPU 上唯一能够跟上 Tensor Core 吞吐速度的存储单元（片上、低延迟），但它也是最稀缺的资源。本章将深入剖析如何在有限的寄存器资源下，平衡**单线程性能（ILP）**与**整体并发度（TLP/Occupancy）**，并建立一套从诊断到决策的完整工程方法论。

---

## TL;DR（工程结论清单）

**微型决策流：**  
`编译时发现 spill？` → 分级（S0–S3）→ `S2/S3 必须处理` → 判断 Kernel Bound 类型 → `Compute Bound?` 提高 ILP，允许更低 Occupancy / `Memory Bound?` 提高 TLP，小心避免溢出。

1. **先看编译证据**：打开 `-Xptxas=-v`，关注 `reg`、`spill stores`、`spill loads`。没有证据不要凭感觉调参。
2. **Spill 不等于死刑，但要分级**：若 spill 发生在 hot loop / 累加器 / tile 数据路径上，往往是性能“雪崩”的根因。
3. **Occupancy 不是 KPI**：对于 compute‑bound（尤其 Tensor Core）算子，“更低 Occupancy + 更高 ILP”常常更快；memory‑bound 算子则相反。
4. **`__launch_bounds__` 是契约，不是魔法**：它能“换 Occupancy”，也能“制造 Spill”。使用后必须回看 ptxas 日志与（最好有）NCU 证据。
5. **优化流程要闭环**：改代码/改参数 → 跑最小复现实验 → 记录日志/指标 → 对比吞吐与瓶颈归因 → 再迭代。

---

## 1. 寄存器墙：GPU 上最先撞到的物理极限

### 1.1 看起来很大，其实很小
查阅架构资料会发现，从 Ampere (A100) 到 Hopper (H100)，每个 SM 的寄存器文件（Register File, RF）规模通常被概括为：**64K（65,536）个 32‑bit 寄存器**。这看起来十分富裕，但别忘了 GPU 的立身之本是**海量线程并发**。假设我们要在一个 SM 上跑满 2048 个线程（物理上限），平均分摊下来：

\[
\text{平均寄存器数} = \frac{65536}{2048} = 32 \text{ Registers/Thread}
\]

**32 个寄存器能干什么？**  
一个双精度浮点数（FP64）就占 2 个寄存器。如果你在写 GEMM，你需要寄存器来保存 A/B 的分块、C 的累加器、各种指针和索引。32 个瞬间捉襟见肘。

> **工程注脚**：这里的“32”只是理论平均值。实际硬件分配存在**粒度（Granularity）**与**对齐**约束，寄存器占用会呈现“台阶状”变化：你多用 1 个寄存器，可能触发更大的配额跳变。最终以 `ptxas` 日志与实际 Occupancy 为准。

```
一个 SM（示意）

┌──────────────────────────────────────────┐
│                  SM                      │
│                                          │
│  ┌────────── Register File (RF) ───────┐ │
│  │  R0 R1 R2 ... （固定总量，片上）      │ │
│  └─────────────────────────────────────┘ │
│                 ↑ 以 thread 为粒度分配     │
│  ┌────────── Warp Slots / Residency ───┐ │
│  │  Warp0  Warp1  Warp2 ...             │ │  ← warp 越多，越能轮转隐藏延迟
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘

结论：RF 很快，但它既决定“单线程能拿多少状态”，也决定“同一时刻能驻留多少 warp”。
```

### 1.2 溢出（Spilling）：工程分级法则
当 Kernel 过于复杂，编译器无法将所有活跃变量（Live Variables）塞进寄存器时，它会把一部分线程私有变量“外溢”到 **Local Memory**。这就是 **Register Spilling**。

#### 1.2.1 Register Spilling 是什么？
> **一句话定义**：寄存器预算不够时，部分线程私有变量被放进 **Local Memory**；逻辑上线程私有，但物理上走缓存体系，最坏会落到 HBM。

```
（1）寄存器预算不够
      │
      ▼
（2）把一部分变量放到 local memory（STL）
      │
      ▼
（3）每次用到该变量：LDL/STL ↔ L1/L2 ↔ HBM（最坏）

本质：spill 不是“多几条指令”，而是把热路径接入了更慢的存储层级。
```

#### 1.2.2 寄存器 vs Local Memory（工程视角）

| 特性     | 寄存器（Registers）                  | 本地内存（Local Memory）                                            |
|:---------|:-------------------------------------|:--------------------------------------------------------------------|
| **位置** | SM 片上                              | 逻辑上线程私有，**物理上属于全局内存地址空间**，经由 L1/L2           |
| **速度** | 最快，零延迟                         | 命中 L1/L2 时较快，未命中则落入 HBM（非常慢）                       |
| **容量** | 总量固定且稀缺（与 Occupancy 强耦合） | 取决于 HBM 大小，但慢得多                                           |
| **触发** | 编译器自动分配                       | 寄存器不够 / live range 太长 / 局部数组动态索引等导致无法寄存器化   |
| **风险** | ——                                   | hot loop 中出现 `LDL/STL` 往往是性能断崖的开始                       |

#### 1.2.3 为什么会雪崩：Spill 的“落地点”与两个副作用

```
变量太多 / live range 太长
          │
          ▼
   编译器寄存器不够放
          │
          ├──────────────► 生成 spill：STL/LDL（local）
          │                               │
          │                               ▼
          │                        L1/L2（可能命中）
          │                               │
          │                               ▼
          └────────────────────────────► HBM（最坏路径）

两个副作用：
1) hot loop 里出现 LDL/STL → 直接插入高延迟链路（吞吐下滑）
2) local 数据进入缓存体系 → 可能挤占本该给正常 global/shared 的空间（污染）
```

#### 1.2.4 Spilling 分级法则（S0–S3）

并非所有 spill 都是死刑，但必须把它当作**高优先级信号**。以下分级以颜色标识严重程度：

| 等级 | 标识 | 描述                                      | 性能影响           |
|:-----|:-----|:------------------------------------------|:-------------------|
| S0   | 绿   | `spill stores = 0`, `spill loads = 0`     | 无                 |
| S1   | 黄   | 初始化/冷分支/循环外 spill               | 很小，可忽略       |
| S2   | 橙   | hot loop 内 spill，打断流水线              | 显著下降           |
| S3   | 红   | 累加器/tile 等核心热变量 spill            | 断崖式下降，必须重构 |

#### 1.2.5 如何判定自己属于 S0–S3（可操作三步法）

**第一步：看 ptxas 日志（必要条件）**  
编译时加上 `-Xptxas=-v`，关注输出。下面是一个真实示例：

```
$ nvcc -Xptxas=-v -arch=sm_80 kernel.cu
ptxas info    : Compiling entry function '_Z7my_kernel...' for 'sm_80'
ptxas info    : Function properties for _Z7my_kernel...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, 32 bytes smem
```

- `spill stores/loads = 0` → 倾向 S0，悬着的心可以放下一半。
- 有非零值（例如 `spill stores = 128, spill loads = 128`）→ 进入第二步。

**第二步：判断 spill 是否落在热路径**  
此时需要结合源码逻辑经验判断，或使用 profiler 定位。  
- S1：仅在初始化、冷分支或循环外出现。  
- S2：主循环每次迭代都会触发。  
- S3：累加器、tile 数据、频繁更新的中间变量被溢出。

**第三步：用 profiler 验证（建议）**  
在 Nsight Compute 中关注下列指标，若显著上升则基本确认 S2/S3：
- local load/store（与 local memory 相关的统计项）
- `achieved_occupancy` 异常变化
- 指令/访存延迟中 local memory 占比上升等信号

#### 1.2.6 减少/避免 Spilling 的手段（按推荐顺序）

1. **减少寄存器峰值**  
   - 缩短 live range：把跨循环的变量移到循环内，或用重计算代替保存。  
   - 避免大局部数组的随机访问（编译器可能无法寄存器化），改用 Shared Memory 或分阶段处理。
2. **改变编译器决策（务必看证据）**  
   - `__launch_bounds__` / `--maxrregcount`：可能提高 Occupancy，也可能诱发新的 spill。必须回看 ptxas。
3. **用结构换性能（把压力从 hot loop 挪走）**  
   - 调整 tile 尺寸以减少累加器数量。  
   - Warp Specialization（分 Producer/Consumer 角色）从根本拆解单个 warp 的寄存器负担。

> **前沿补充（请以官方文档为准）**：部分新版本工具链/编译器优化可能尝试把部分溢出压力转移到更靠近 SM 的存储层级，以缓解 local memory 的惩罚。具体是否生效、边界条件与收益大小，强依赖架构与编译器版本。

> **权衡提醒**：过度压低寄存器用量可能会让 Occupancy 表面的数字变高，但由于 spill/指令增多，运行时间反而增加。最终只信两样：`ptxas` + 实际运行耗时。

---

## 2. Occupancy 的决策边界：ILP vs TLP

既然寄存器不够分，为什么不减少并发线程数，让每个线程多拿点寄存器呢？这引入了 GPU 优化的核心矛盾。

### 2.0 ILP vs TLP：两种“隐藏等待”的方法
在 GPU 上，性能的敌人通常不是“算得慢”，而是“在等”：
- 等内存（HBM/L2/L1/Shared）返回  
- 等流水线（依赖链）解锁  
- 等发射/调度资源

隐藏等待主要有两条路：
- **ILP（指令级并行）**：让**同一个线程**内部拥有更多彼此独立的指令，即使某条指令在等，也能继续发射。
- **TLP（线程级并行）**：让 **SM 上驻留更多 warp**，当 Warp A 在等，调度器立即切换执行 Warp B。

```
ILP（一个线程内“并行度”更高）
Time →
Thread0:  [ADD] [MUL] [ADD] [MUL]
           ↑     ↑
      独立指令多 → 更容易把等待藏在别的指令后面（但需要更多寄存器保存中间状态）

TLP（多个 warp“轮转”更快）
Time →
Warp0:    [RUN] [WAIT_MEM] [RUN]
Warp1:    [RUN] [RUN]      [WAIT_MEM]
Warp2:    [WAIT_MEM] [RUN] [RUN]
           ↑  Warp0 等内存时，直接换 Warp1/2 跑（但需要更高 Occupancy 才有得换）
```

**收益与代价简表：**

|        | ILP 偏重                             | TLP 偏重                           |
|:-------|:-------------------------------------|:-----------------------------------|
| 收益   | 算力满、单线程效率高、适合 compute-bound | 延迟隐藏强、适合 memory/latency-bound |
| 代价   | 寄存器需求大、Occupancy 低、易溢出     | 易因寄存器不足而引发溢出，单线程弱   |
| 典型手段 | 循环展开、register blocking         | 减少 reg/smem 占用、提高 warp 驻留  |

### 2.1 什么是 Occupancy？
**Occupancy（占用率）** 指的是 SM 上实际驻留的 Warp 数量与最大可能驻留数量的比值。它代表了**TLP（线程级并行）**的能力。

### 2.2 寄存器是 Occupancy 的调节阀
SM 的寄存器总量是固定的。
- **策略 A：省着用**。每个线程只用 32 个寄存器 → 100% Occupancy。
- **策略 B：敞开用**。每个线程用 255 个寄存器（最大值） → 12.5% Occupancy。

#### 2.2.1 核心公式
寄存器直接限制可驻留 warp 数。工程近似公式：

\[
\text{Warps}_{\text{reg}} = \left\lfloor \frac{\text{TotalRegsPerSM}}{\text{RegsPerThread} \times 32} \right\rfloor \tag{1}
\]

实际可驻留 warp 数是多种资源的交集：

\[
\text{ActiveWarpsPerSM} = \min\left( \text{Warps}_{\text{reg}},\ \left\lfloor\frac{\text{MaxThreadsPerSM}}{32}\right\rfloor,\ \text{Shared Memory 等其他约束} \right) \tag{2}
\]

- **TotalRegsPerSM**：该架构每个 SM 的 32‑bit 寄存器总量（工程口径常写作 64K regs）
- **RegsPerThread**：编译器为每个线程分配的寄存器数（看 `-Xptxas=-v`）

> 真实驻留还会被**分配粒度、max blocks/SM**等因素裁剪，最终以 Occupancy Calculator 或实际运行值为准。

#### 2.2.2 阶梯效应示例
假设某架构 SM 拥有 **64K 寄存器**，最大可驻留 **64 warps**。仅考虑寄存器约束（公式 1）：

| Regs/Thread | 寄存器可驻留 Warps | Occupancy（相对 64 warps） |
|:---:|:---:|:---:|
| 32  | 64 | 100% |
| 64  | 32 | 50% |
| 96  | 21 | 33% |
| 128 | 16 | 25% |
| 160 | 12 | 19% |
| 192 | 10 | 16% |
| 256 | 8  | 12.5% |

- **台阶效应**：Regs/Thread 不是线性影响，而是按整数向下取整出现台阶。
- **双刃剑**：Regs 少 → Occupancy 高（TLP 强）；Regs 多 → ILP 空间大但 Occupancy 低。

```
Max Warps/SM
^
|  ┌───────────────┐
|  │               │
|  │   台阶式下降   │   （寄存器/共享内存/其它资源都会造成这种台阶）
|  │               │
|  └───────┐       │
|          └───────┘
+------------------------------> Registers / Thread

工程含义：你“多用一点点寄存器”，可能直接跨过一个台阶，驻留 warp 数骤减。
```

### 2.3 决策边界表
“高 Occupancy 一定好”是初学者最大的误区。在 Tensor Core 时代，我们往往通过**ILP**而非 TLP 来掩盖延迟。以下是 2025 年主流架构的经验法则：

| 算子类型 | 特征 | 目标 Occupancy | 策略逻辑 |
|:---|:---|:---|:---|
| **Memory Bound** <br>(如 VectorAdd, Softmax) | 依赖 HBM 带宽 | **≥ 75%** | 需要大量 Warp 轮转掩盖 HBM 数百周期延迟 |
| **Shared‑Heavy** <br>(如经典 Tiled GEMM) | 依赖 Shared Memory | **≥ 50%** | Shared 延迟较低，中等 Occupancy 即可掩盖 |
| **Compute Bound** <br>(如 Tensor Core GEMM) | 依赖 ALU/TC 吞吐 | **25% – 50%** | **单线程性能优先**。大量寄存器做 Double Buffering，靠 ILP 掩盖延迟 |

> **关键结论**：超过“完全掩盖延迟的阈值”后，继续压低寄存器追求更高 Occupancy，不仅不会提升性能，反而会因寄存器不足导致 Spill，得不偿失。
>
> 这个“低 Occupancy 反而更快”的反直觉结论，最经典的论证来自 Volkov 的 GTC 报告 [1]。

#### 2.4 自查清单：决定该追 ILP 还是 TLP

- **你的 Kernel 是 Memory Bound 吗？**  
  - 现象：dram 吞吐接近上限，sm 吞吐很低。  
  - 行动：减少寄存器/共享内存占用，提高驻留；优先避免 S2/S3 spill。
- **你的 Kernel 是 Compute Bound 吗？**  
  - 现象：sm 吞吐接近上限，dram 吞吐不高。  
  - 行动：允许更高寄存器用量做 buffering/accumulation，但确保没有滑入 S2/S3 spill。

---

## 3. 驯服编译器：`__launch_bounds__` 的双刃剑

编译器（NVCC）通常比较保守。`__launch_bounds__` 是开发者与编译器签订的**契约**。

### 3.1 语法与语义
```cpp
__global__ void 
__launch_bounds__(max_threads_per_block, min_blocks_per_multiprocessor)
my_kernel(...) { ... }
```
- `max_threads_per_block`：限制 Block 大小，辅助编译器优化。
- `min_blocks_per_multiprocessor`：**强制承诺**。要求编译器保证每个 SM 至少能跑 `min_blocks` 个 Block。

> 这部分属于 CUDA 官方“Execution Configuration / Occupancy”范畴，建议对照 CUDA Best Practices Guide 与 CUDA Programming Guide 阅读 [2][3]。

### 3.2 踩坑警告：强制的代价
如果你设置了 `min_blocks=2`，编译器会被迫限制每个线程的寄存器用量。如果代码逻辑复杂，物理上无法压缩寄存器，编译器为了履行契约，**会主动生成 Spill 代码**。

**后果**：Occupancy 指标上去了（好看），但主循环里全是 Local Memory 读写（慢），**性能反而雪崩**。

**修正示例**：
```cpp
// 危险：强制 min_blocks=2，但代码复杂 → 编译器被迫溢出
__global__ __launch_bounds__(256, 2) void dangerous_kernel(...) { /* 大量变量 */ }

// 安全：先不强制 min_blocks，观察 ptxas 后再决定
__global__ __launch_bounds__(256) void safer_kernel(...) { /* 同上逻辑 */ }
```

### 3.3 安全使用流程（严格 SOP）
1. **先不填 `min_blocks`**：使用 `__launch_bounds__(max_threads)`，避免直接施压。
2. **逐步加压**：如需更高并发，尝试 `min_blocks = 2`，每次只改一个变量。
3. **每一步都做两件事**：
   - 记录 `ptxas`（reg + spill loads/stores）  
   - 用统一输入规模做 event 计时或 NCU 采样
4. **出现 S2/S3 倾向则回退**：回退参数，或重构结构（缩短 live range、拆分 hot loop 等）。

> **提醒**：不要用“某个寄存器数必然导致 1 block/SM”这种绝对叙述。真实驻留由多重因素共同决定。工程上只信 **看 ptxas** + **跑基准**。

---

## 4. 工业级案例：CUTLASS 的寄存器魔法

为什么 NVIDIA 官方 CUTLASS 库能写出媲美 cuBLAS 的性能？看看它们是如何榨干寄存器的。

### 4.1 Register Blocking（寄存器分块）
将 Shared Memory 中的小块数据加载到寄存器，然后在寄存器级别做外积。**计算完全锁定在 RF 内部，最大化复用。**
（CUTLASS 对“分层分块/寄存器分块”的解释很系统，见官方文档 [4]。）

### 4.2 Warp Specialization（Warp 特化）
这是 Hopper 架构及 CUTLASS 3.x 的核心思想。将 Warp 分为：
- **Producer Warps**：只负责发 TMA 指令搬运数据，逻辑简单，寄存器用量极少。
- **Consumer Warps**：只负责执行 WGMMA 计算，可独占大量寄存器用于累加。

**收益**：将搬运和计算的活跃变量生命周期（live range）**拆散**，从源头减少 spill 风险。
（想看 Hopper/WGMMA 语境下的实现剖析，可参考 Colfax 教程 [5]。）

### 4.3 Register Double Buffering
当计算单元消费寄存器块 A 时，LD/ST 单元把下一块数据搬到寄存器块 B。  
- **收益**：埋藏搬运延迟，提高 ILP。  
- **代价**：寄存器需求显著上升，Occupancy 下降，若控制不好 live range 则易引发 spill。

### 4.4 将复杂 Epilogue 隔离
复杂的写回/融合（bias、activation、layout transform 等）如果内联到主循环，会拉长 live range、推高寄存器峰值，诱发溢出。工程上常见做法是将其**隔离为独立阶段**，把寄存器压力从 hot loop 挪出去。
（CUTLASS 的 Epilogue Visitor Trees 设计也很适合用来理解“为什么要把复杂性从主循环挪走” [6]。）

下表总结 CUTLASS 技术与其解决的寄存器问题：

| CUTLASS 技术            | 主要解决的寄存器问题              | 关键手段                     |
|:------------------------|:----------------------------------|:-----------------------------|
| Register Blocking       | 减少对 Shared/L1 依赖，提升 ILP   | 块数据常驻寄存器             |
| Warp Specialization     | 降低单个 warp 的 live range 峰值  | 拆分搬运/计算角色            |
| Register Double Buffering | 掩藏延迟，提升流水线饱和度        | 双缓冲寄存器块               |
| Epilogue 隔离           | 避免热路径被写回逻辑撑大寄存器    | 拆分为独立阶段或函数         |

---

## 5. 实战：Register Spilling 复现实验（配套代码）

理论说得再多，不如代码跑一跑。本章的配套代码将通过一个**可控的寄存器压力 micro-bench**，演示三种实现/策略在性能与 spilling 行为上的差异。

### 实验目标

1. **观察寄存器压力如何触发 spilling**
2. **观察 spilling 如何影响 kernel time**
3. **观察 `__launch_bounds__` 的“契约效应”**：它可能改善驻留，也可能诱发新的 spill

### 实验设计（同一输入规模，对比三种变体）

我们实现一个 kernel：每个线程维护一组线程私有状态，并在循环中反复读写，放大寄存器压力与 live range。

- **Variant A（baseline）**：寄存器压力较低（更不容易 spill）
- **Variant B（high-reg / spill-prone）**：寄存器压力很高（更可能触发 local spill）
- **Variant C（launch_bounds）**：对 Variant B 增加 `__launch_bounds__` 提示，观察编译器是否改变 reg/spill 行为

### 你需要关注的证据链（强烈建议至少做到前两项）

1. **编译证据**：`-Xptxas=-v` 的输出（`Used XX registers` / `spill stores` / `spill loads` / `stack frame`）
2. **运行证据**：同一输入下的 kernel 平均耗时（CUDA Event）
3. **可选（更强证据）**：Nsight Compute 中 local memory 相关统计项是否显著上升（用于确认 S2/S3 风险）

### 预期现象（不强行承诺数值，但承诺“对比方法”）

- 如果 Variant B 明显变慢，同时 `spill loads/stores` 非 0，通常意味着你击中了 spilling（且大概率落在热路径）。
- **如果 `spill loads/stores = 0` 但 `stack frame` 很大**，同样要警惕：这往往意味着线程私有状态被放到了 local/stack 路径上（仍然可能显著变慢）。所以这里的“判定”应以**时间断崖 + ptxas 证据组合**为准，而不是只盯着 spill 字段。
- 如果 Variant C 改变了 `reg` 或 `spill`，说明 `__launch_bounds__` 确实在影响编译器的资源分配决策（它是契约，不是免费午餐）。

#### 示例结果（不同设备会不同，重点看“对比方法”）

以下示例以 `sm_120`（例如 RTX 5090 一类架构）为编译目标，参数为 `N=1048576`、`inner_iters=256`：

```
[A] baseline (REGS=32)    : 4.6272 ms
[B] high-reg (REGS=256)   : 16.4192 ms
[C] launch_bounds (2 blks): 16.4200 ms
```

解读要点：

- B 相比 A 约 **3.55×** 变慢，说明“线程私有状态膨胀”确实可能造成断崖式回退。
- 如果你看到 `ptxas` 输出中 `spill loads/stores` 为 0，但 `stack frame` 很大（例如 1024 bytes），这依然是一个强烈信号：热路径可能走到了 local/stack 相关访问上。

> [💻 代码占位符：参见项目 `examples/02_memory_optim/03_register_spill.cu`]

---

## 6. 总结：实战闭环（SOP）

第 5 章就是本章 SOP 的**最小示例（MVP）**。把它抽象成一屏能执行的流程如下：

1. **先体检**：编译加 `-Xptxas=-v`，记录 `reg / spill loads / spill stores`（作为回归基线）。
2. **先分级**：有 spill 就按 S1/S2/S3 判断；**S2/S3 不要硬调参数，先处理结构/热路径**。
3. **再归因**：用 profiler 判断更偏 **memory-bound** 还是 **compute-bound**（决定追 TLP 还是 ILP）。
4. **再动手**：
   - 想追 **TLP**：减少 reg/smem 占用，提高驻留；但避免把自己挤进 spill。
   - 想追 **ILP**：允许更多寄存器做 blocking/buffering，但盯住 live range 与 spill 风险。
5. **最后验收**：同一输入规模对比运行时间 + `ptxas` 变化；任何“指标更好但更慢”的优化都应回退。

👉 **下一章：[Module B] 14. L2 Cache 策略：CUDA 12 的 L2 驻留控制**  
解决了 SM 内部的资源博弈，下一章我们将跳出 SM，来到 GPU 的 **L2 Cache**。我们将学习如何利用 CUDA 12 的新 API，像管理 Shared Memory 一样手动管理 L2，让关键数据“钉”在缓存里。

---

## 📚 参考文献

### 正文已引用（含可点击链接）

1. **Volkov, V. (2010).** *Better Performance at Lower Occupancy.*（GTC 2010 Slides）  
   <https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf>

2. **NVIDIA CUDA C++ Best Practices Guide（最新）**  
   <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>

3. **NVIDIA CUDA Programming Guide（最新）**  
   <https://docs.nvidia.com/cuda/cuda-programming-guide/>

4. **NVIDIA CUTLASS Documentation — Efficient GEMM in CUDA**  
   <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html>

5. **Colfax Research (2024).** *CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs*  
   <https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/>

6. **Colfax Research (2024).** *Epilogue Fusion in CUTLASS with Epilogue Visitor Trees*  
   <https://research.colfax-intl.com/epilogue_visitor_tree/>

### 延伸阅读（本文未逐条引用，按需深入）

- CUTLASS 源码仓库：<https://github.com/NVIDIA/cutlass>
- NVIDIA Developer Blog 与学术论文（寄存器缓存、SMRS 等）：建议你在后续章节（例如 L2/系统工程部分）真的用到时，再把对应条目提升为“正文已引用”，避免参考文献列表过长且难以维护。

---

