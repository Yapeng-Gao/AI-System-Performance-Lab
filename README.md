
# AI System Performance Lab (ASPL)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)
![License](https://img.shields.io/badge/license-GPL%20v3-blue)

**AI System Performance Lab** 是《AI 系统性能工程》专栏的配套仓库。  
**当前主线**：CUDA Module A + Module B（B-01～B-09 已落地；B-10 Checklist 规划中）。  
结构说明见 [`docs/仓库架构与现状.md`](docs/仓库架构与现状.md)。

## 怎么读

| 你想做的事 | 去哪 |
|---|---|
| 读文章 | `article/01_cuda_basic/`、`article/02_memory_optim/` |
| 跑章节实验 | `examples/01_cuda_basics/`、`examples/02_memory_optim/` |
| 看实测与 CSV | `docs/results/` |
| 重画 B-05～B-09 数据图 | `python scripts/plot_b05_unified_memory.py` 等 |
| 专栏进度（导航） | [`docs/CUDA专栏规划.md`](docs/CUDA专栏规划.md) |
| 按章写作大纲 | [`docs/CUDA专栏大纲/`](docs/CUDA专栏大纲/README.md) |

```text
AI-System-Performance-Lab/
├── article/           # 专栏正文 + assets
├── examples/          # 章节 .cu（CMake 唯一构建目标）
│   ├── 01_cuda_basics/
│   └── 02_memory_optim/
├── docs/              # 规划 + results/
├── scripts/           # 绘图 / profile 辅助
├── cmake/
└── CMakeLists.txt
```

## 快速开始 (Windows/Linux)

### 前置要求
*   **CMake**: `>= 3.25`
*   **CUDA Toolkit**: `>= 12.0`（推荐 12.6+ / 13.x）
*   **Compiler**: C++17（Linux GCC >= 9；Windows MSVC VS2022）

> 请装 NVIDIA 官方 Toolkit。不要用 Ubuntu 源的 `apt install nvidia-cuda-toolkit`。

### 编译

#### Linux

```bash
export CUDA_HOME=/usr/local/cuda-13.2   # 或 /usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --parallel "$(nproc)"
```

架构探测失败时写死（如 `8.9` → `89`，Blackwell 消费卡常见 `120`）：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build . --parallel "$(nproc)"
```

只编某个示例：

```bash
cmake --build . --parallel "$(nproc)" --target 02_memory_optim_07_cp_async_pipeline
```

> `examples` 增删 `.cu` 后，先再跑一次 `cmake ..`。

| 报错 | 处理 |
|---|---|
| `nvcc: command not found` | 检查 `CUDA_HOME`/`PATH` |
| `CMAKE_CUDA_ARCHITECTURES must be non-empty` | `rm -rf build` 后显式传架构 |

#### Windows/CLion
- **CLion**: 直接构建（输出在 `cmake-build-debug` 等）
- **手动**:
```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --config Release --parallel 8
```

### 运行

```bash
./build/bin/01_cuda_basics_01_hello_modern
./build/bin/02_memory_optim_06_pinned_dma --mode pinned --mb 256
./build/bin/02_memory_optim_07_cp_async_pipeline --mode sweep
```

重画实测图：

```bash
python scripts/plot_b05_unified_memory.py
python scripts/plot_b06_pinned_dma.py
python scripts/plot_b07_cp_async.py
```

## 专栏映射

| 模块 | 路径 | 状态 |
| :--- | :--- | :--- |
| **Module A** | `article/01_cuda_basic` + `examples/01_cuda_basics` | ✅ |
| **Module B** | `article/02_memory_optim` + `examples/02_memory_optim` | 🟡 B-01～B-09 ✅；B-10 ⏳ |
| **Module C–E** | 仅规划文档 | ⏳ |

正文插图：原理短 ASCII，实测 matplotlib（见架构文档 §4）。

## 规划文档

- [`docs/CUDA专栏规划.md`](docs/CUDA专栏规划.md)（导航总表）
- [`docs/CUDA专栏大纲/`](docs/CUDA专栏大纲/README.md)（按章大纲）
- [`docs/仓库架构与现状.md`](docs/仓库架构与现状.md)
- 远期：`docs/昇腾CANN专栏规划.md` 等

## 许可证

GNU GPL v3.0 — 见 `LICENSE`。
