
# AI System Performance Lab (ASPL)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)
![License](https://img.shields.io/badge/license-GPL%20v3-blue)

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
*   **CMake**: `>= 3.25` (推荐 3.28+, 4.0 为预览版支持)
*   **CUDA Toolkit**: `>= 12.0` (推荐 12.3+ 以支持 Hopper 完整特性)
*   **Compiler**: C++17 Support Required
    *   Linux: GCC >= 9.0
    *   Windows: MSVC v143+ (Visual Studio 2022,vision 17)
*   **Driver**: `>= 535.xx` (与 CUDA 12 匹配)

### 编译构建

#### Linux 环境
```bash
# 1. 创建构建目录
mkdir build && cd build

# 2. 生成构建文件
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. 编译
cmake --build . --parallel 8
```

> 说明：`examples` 目录新增/删除 `.cu` 示例后，需要先重新执行一次 `cmake ..`（重新配置）再 `cmake --build .`，否则新 target 不会出现在 `build/bin`。

#### Windows/CLion 环境
- **CLion**: 直接使用 IDE 构建（输出在 `cmake-build-debug` 目录）
- **手动构建**:
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8
```

### 运行示例

编译成功后，可执行文件位置：
- **Linux**: `build/bin/` 目录
- **Windows/CLion**: `cmake-build-debug/bin/` 或 `cmake-build-debug-visual-studio/bin/` 目录

```bash
# Linux
./build/bin/01_cuda_basics_01_hello_modern

# Windows (PowerShell)
.\cmake-build-debug\bin\01_cuda_basics_01_hello_modern.exe
```

## 📚 专栏内容映射

| 模块 | 路径 | 核心内容 |
| :--- | :--- | :--- |
| **Module A** | `examples/01_cuda_basics` | 架构映射, 线程调度, SASS 分析 |
| **Module B** | `src/kernels/cuda/memory` | TMA, Pinned Memory, Coalescing |
| **Module C** | `examples/03_compute_primitives` | Warp Primitives, CUDA Graphs |
| **Module D** | `src/kernels/cuda/math` | Tensor Core (WGMMA), FP8 |
| **Module E** | `python/aspl/triton_kernels` | vLLM PagedAttention, Triton |

## 📌 专栏规划文档（本仓库主线）

- `docs/CUDA专栏规划.md`
- `docs/昇腾CANN专栏规划.md`
- `docs/异构计算与生态迁移专栏规划.md`
- `docs/硬核进阶专栏规划.md`
- `docs/专家进阶专栏规划.md`

## 🤝 贡献指南

1.  Fork 本仓库
2.  新建特性分支 (`git checkout -b feature/AmazingFeature`)
3.  提交更改 (`git commit -m 'Add some AmazingFeature'`)
4.  推送到分支 (`git push origin feature/AmazingFeature`)
5.  提交 Pull Request

## 📄 许可证

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

