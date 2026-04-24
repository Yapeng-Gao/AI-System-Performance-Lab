# 昇腾 CANN 专栏规划（单独文档）

本文件从 `docs/大模型算法系列规划.md` 中抽离整理《国产 AI 计算生态（昇腾 CANN）》专栏，作为本仓库后续文章与代码落地的工作底稿。

---

## 0. 专栏定位与落地方式

**专栏主题**：面向华为昇腾生态，系统解析 **达芬奇架构** 与 **CANN 软件栈**，并以 **ASCEND C / TBE** 为核心完成算子级实战。

**本仓库的落地约定（建议）**：

- **文章**：`article/03_ascend_cann/`
- **示例代码**：`examples/05_ascend_cann/`
- **基准与 profiling**：优先复用 `benchmarks/` / `scripts/` 的方法论，新增 Ascend 对应的采集脚本与结果目录（例如 `docs/results/ascend/`）

**状态**：⏳ 规划中（仓库当前主要落地在 CUDA 专栏；本专栏以规划为主，后续逐篇补齐“文章 + 可运行示例 + 可复现指标”。）

---

## 1. 模块一：昇腾生态入门 —— 硬件、软件与开发环境

### 第 1 章：国产 AI 芯片与昇腾计算平台

- 国产 AI 芯片的崛起：机遇、挑战与主流玩家
- \[图解\] 华为昇腾（Ascend）AI 处理器全景：从 Atlas 系列到 SoC
- \[图解\] 达芬奇架构深度剖析：AI Core 与 AI CPU
- \[图解\] AI Core：标量/向量/矩阵计算单元（Cube Unit）
- \[对比\] 昇腾 AI Core vs NVIDIA GPU SM：架构异同

### 第 2 章：CANN 软件栈详解

- \[图解\] CANN 全景：从上层框架到底层驱动
- AscendCL：应用层 API
- Graph Engine：图编译与执行
- TBE：算子开发与优化引擎
- Kernel Driver / Runtime：硬件交互桥梁

### 第 3 章：开发环境搭建与工具链

- 昇腾开发环境搭建（物理机/虚拟机/云）
- MindStudio IDE 入门
- Ascend-Toolkit 核心组件
- Profiling 工具入门
- \[实践\] 编译并运行官方示例

---

## 2. 模块二：ASCEND C 编程与算子开发

### 第 4 章：ASCEND C 编程模型入门

- ASCEND C 核心概念：Kernel / Tiling / Queue / Pipe
- \[图解\] AI Core 内存体系：L0 / L1 / UB
- 数据搬运原语：DataCopy / dma 指令
- 同步：PipeBarrier 与事件同步
- \[实践\] 第一个 ASCEND C Kernel：Vector Add

### 第 5 章：ASCEND C 编程进阶

- 向量计算（Vector Core）接口
- 矩阵计算（Cube Core）接口
- Tiling 策略：性能优化核心
- \[图解\] Pipelining：搬运与计算流水线化
- \[优化\] Bank Conflict 与数据布局

### 第 6 章：TBE 算子开发流程

- TBE DSL 与 Schedule（Pythonic 开发方式）
- ASCEND C 自定义高性能算子（Custom OP）
- 算子工程组织与编译
- \[实践\] Element-wise / Reduction 算子并集成到网络

### 第 7 章：对标实践——在昇腾上实现核心算子

- \[挑战\] GEMM：用 ASCEND C 实现简化矩阵乘
- \[挑战\] 性能调优：Tiling / Pipelining / 访存优化
- \[分析\] 类 FlashAttention 的可行性、难点与替代方案
- \[对比\] CUDA 优化技巧如何迁移到 ASCEND C

---

## 3. 模块三：上层框架适配与生态展望

### 第 8 章：主流框架在昇腾上的适配与使用

- MindSpore 与昇腾协同
- PyTorch-Ascend（`torch_npu`）工作机制与使用
- TensorFlow-Ascend 插件
- \[实践\] 将 PyTorch 模型迁移到昇腾训练/推理

### 第 9 章：其他国产 AI 平台概览

- 寒武纪 MLU / BANG
- 燧原 Enflame / GCU
- 壁仞 BR100 / BIRENSUPA
- 国产生态的共性与差异

### 第 10 章：总结与展望

- 昇腾生态优势、挑战与未来
- 开发者如何参与生态共建
- \[联动\] 与《异构计算与生态迁移》专栏的知识衔接

---

## 4. 与本仓库代码架构的衔接建议（先定“怎么落地”，再写代码）

- **统一“算子库抽象层”**：在 `include/aspl/ops/` 定义与硬件无关的 Host API；在 `src/kernels/<backend>/` 放各后端 kernel（CUDA / CANN / …）。
- **Backend 分层**：
  - `src/kernels/cuda/`：CUDA kernels（已有雏形）
  - `src/kernels/cann/`：CANN kernels（建议补真实目录与最小示例）
- **示例工程按专栏分目录**：`examples/05_ascend_cann/` 与 `examples/01_cuda_basics/` 类似的“章节—文件”映射方式，保证读者可跑。

