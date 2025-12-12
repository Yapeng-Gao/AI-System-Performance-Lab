
# AI System Performance Lab (ASPL)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

**AI System Performance Lab** 是一个工业级的高性能计算与大模型推理优化代码库。它是《AI 系统性能工程》五大专栏的官方配套项目，涵盖了从底层 CUDA 优化到上层大模型系统实现的完整技术栈。

## 📂 项目结构

```text
AI-System-Performance-Lab/
├── cmake/               # CMake 构建脚本与架构探测
├── include/             # 对外头文件 (API 接口)
├── src/                 # 核心库实现 (libaspl_core)
│   ├── kernels/cuda/    # CUDA Kernels (Reduction, Attention, TMA...)
│   ├── ops/             # C++ Host 算子调度
│   └── utils/           # 工具函数
├── python/              # Python 绑定与 Triton 算子
├── examples/            # 专栏配套实战代码 (独立可运行)
│   ├── 01_cuda_basics/  # 基础架构
│   ├── 02_memory_optim/ # 内存优化 (Coalescing, Bank Conflict)
│   └── ...
├── tests/               # 单元测试 (GTest)
└── benchmarks/          # 性能基准测试 (NVBench)
```

## 🚀 快速开始 (Windows/Linux)

### 前置要求
*   CMake >= 4
*   CUDA Toolkit >= 12.0 (推荐 12.0+)
*   C++17 Compiler (MSVC / GCC)

### 编译构建

```bash
# 1. 创建构建目录
mkdir build && cd build

# 2. 生成构建文件 (Windows 推荐使用 Ninja 或 Visual Studio)
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. 编译
cmake --build . --parallel 8
```

### 运行示例

编译成功后，可执行文件位于 `build/bin` 目录下：

```bash
./bin/01_basics_main
```

## 📚 专栏内容映射

| 模块 | 路径 | 核心内容 |
| :--- | :--- | :--- |
| **Module A** | `examples/01_cuda_basics` | 架构映射, 线程调度, SASS 分析 |
| **Module B** | `src/kernels/cuda/memory` | TMA, Pinned Memory, Coalescing |
| **Module C** | `examples/03_compute_primitives` | Warp Primitives, CUDA Graphs |
| **Module D** | `src/kernels/cuda/math` | Tensor Core (WGMMA), FP8 |
| **Module E** | `python/aspl/triton_kernels` | vLLM PagedAttention, Triton |

## 🤝 贡献指南

1.  Fork 本仓库
2.  新建特性分支 (`git checkout -b feature/AmazingFeature`)
3.  提交更改 (`git commit -m 'Add some AmazingFeature'`)
4.  推送到分支 (`git push origin feature/AmazingFeature`)
5.  提交 Pull Request

## 📄 许可证

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

