
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
*   **CMake**: `>= 3.25`（Ubuntu 24.04 自带 3.28 即可；与 CUDA 13.2 无关）
*   **CUDA Toolkit**: `>= 12.0`（推荐 12.6+ / 13.x）
*   **Compiler**: C++17（Linux GCC >= 9；Windows MSVC VS2022）
*   **Driver**: 能跑所选 Toolkit（`nvidia-smi` 右上角是驱动支持上限，不是 `nvcc` 版本）

> 请装 NVIDIA 官方 Toolkit。不要用 Ubuntu 源的 `apt install nvidia-cuda-toolkit`。

### 编译构建

#### Linux 环境

```bash
# 一次性：让 nvcc 进 PATH（装完 Toolkit 后通常只需做一次，可写进 ~/.bashrc）
export CUDA_HOME=/usr/local/cuda-13.2   # 或 /usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --parallel "$(nproc)"
```

若 `native` 探测失败，把架构写死（`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`，如 `8.9` → `89`）：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --parallel "$(nproc)"
```

只编某个示例：

```bash
cmake --build . --parallel "$(nproc)" --target 02_memory_optim_06_pinned_dma
```

> `examples` 里增删 `.cu` 后，先再跑一次 `cmake ..`，再 `cmake --build .`。

常见问题（可选阅读）：

| 报错 | 处理 |
|---|---|
| `nvcc: command not found` | 检查 `CUDA_HOME`/`PATH`，`which nvcc` |
| `CMAKE_CUDA_ARCHITECTURES must be non-empty` | `rm -rf build` 后重配，并显式传架构 |
| `nvbench ... CMake 3.30.4 or higher` | 与 Toolkit 无关；默认已不拉 nvbench。勿开 `-DASPL_ENABLE_NVBENCH=ON`，除非本机 CMake ≥ 3.30.4 |

#### Windows/CLion 环境
- **CLion**: 直接构建（输出在 `cmake-build-debug` 等目录）
- **手动**:
```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --config Release --parallel 8
```

### 运行示例

- **Linux**: `build/bin/`
- **Windows/CLion**: `cmake-build-debug/bin/` 或 `build/bin/Release/`

```bash
./build/bin/01_cuda_basics_01_hello_modern
./build/bin/02_memory_optim_06_pinned_dma --mode pinned --mb 256
```

## 📚 专栏内容映射

| 模块 | 路径 | 核心内容 |
| :--- | :--- | :--- |
| **Module A** | `examples/01_cuda_basics` + `article/01_cuda_basic` | 架构映射, 线程调度, SASS 分析 |
| **Module B** | `examples/02_memory_optim` + `article/02_memory_optim` | Coalescing, Shared, L2, UM, Pinned/DMA, Async Copy/Pipeline |
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

