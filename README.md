
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
*   **CMake**: `>= 3.25`（推荐 3.28+）
*   **CUDA Toolkit**: `>= 12.0`（推荐 12.6/12.8 或 13.x；本仓库 `cmake/NVCCFlags.cmake` 硬性要求 ≥12.0）
*   **Compiler**: C++17
    *   Linux: GCC >= 9.0
    *   Windows: MSVC v143+（Visual Studio 2022）
*   **Driver**: 需支持所选 Toolkit（`nvidia-smi` 右上角 `CUDA Version` 是驱动上限，不是已装的 `nvcc` 版本）

> Linux 请装 NVIDIA 官方 Toolkit（`cuda-toolkit-12-x` / `cuda-toolkit-13-x`）。  
> **不要**用 Ubuntu 源的 `apt install nvidia-cuda-toolkit`，版本旧且易与官方包冲突。

### 编译构建

#### Linux 环境

```bash
# 0) 确认驱动与 nvcc（官方安装后通常在 /usr/local/cuda-* ）
nvidia-smi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# 若只有带版本号的目录，例如：
#   export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version   # 应打印 Cuda compilation tools, release 12.x / 13.x

# 1) 查本机 GPU 架构（例：8.9 → 传 89；9.0 → 90；7.5 → 75）
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 2) 干净配置（推荐；避免旧缓存里 CMAKE_CUDA_ARCHITECTURES="" 导致失败）
cd /path/to/AI-System-Performance-Lab
rm -rf build && mkdir build && cd build
unset CMAKE_CUDA_ARCHITECTURES

# 3) 配置：显式指定 nvcc + 架构（也可改成 native）
ARCH=89   # <-- 改成上一步的 compute_cap（去掉小数点）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"

# 4) 全量编译，或只编某个示例
cmake --build . --parallel "$(nproc)"
# cmake --build . --parallel "$(nproc)" --target 02_memory_optim_06_pinned_dma
```

常用说明：

- 新增/删除 `examples/**/*.cu` 后，必须先重新 `cmake ..`，再 `cmake --build .`，否则新 target 不会进 `build/bin`。
- 若报 `CMAKE_CUDA_ARCHITECTURES must be non-empty if set`：删掉 `build/` 后重配，并显式传 `-DCMAKE_CUDA_ARCHITECTURES=...`。
- 若报 `CUDA compiler identification is unknown`：多半是 `nvcc` 不在 `PATH`，先 `which nvcc`。
- 不确定架构时可试：`-DCMAKE_CUDA_ARCHITECTURES=native`（需本机有可见 GPU）。

#### Windows/CLion 环境
- **CLion**: 直接使用 IDE 构建（输出在 `cmake-build-debug` 等目录）
- **手动构建**:
```powershell
# 先确认 nvcc 在 PATH；架构按本机 GPU 修改（例 sm_89 → 89）
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CUDA_COMPILER="nvcc" `
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --config Release --parallel 8
```

### 运行示例

编译成功后，可执行文件位置：
- **Linux**: `build/bin/`
- **Windows/CLion**: `cmake-build-debug/bin/` 或 `build/bin/Release/`

```bash
# Linux
./build/bin/01_cuda_basics_01_hello_modern
./build/bin/02_memory_optim_06_pinned_dma --mode pinned --mb 256

# Windows (PowerShell)
.\cmake-build-debug\bin\01_cuda_basics_01_hello_modern.exe
```

## 📚 专栏内容映射

| 模块 | 路径 | 核心内容 |
| :--- | :--- | :--- |
| **Module A** | `examples/01_cuda_basics` + `article/01_cuda_basic` | 架构映射, 线程调度, SASS 分析 |
| **Module B** | `examples/02_memory_optim` + `article/02_memory_optim` | Coalescing, Shared, L2, UM, Pinned/DMA |
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

