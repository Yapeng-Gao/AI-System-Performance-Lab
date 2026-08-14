# AI System Performance Lab (ASPL)

[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Yapeng-Gao/AI-System-Performance-Lab?style=social)](https://github.com/Yapeng-Gao/AI-System-Performance-Lab)

**CUDA 性能工程：专栏正文 + 可跑微基准 + RTX 5090 实测表。**  
不是教程摘抄——每章给决策表、一条主命令、本机 median 数字。  
仓库现在没有 Triton / FlashAttention / vLLM；那些在 Module D/E 规划里。

先读 [专栏导读](article/01_cuda_basic/00.%20专栏导读：怎么读、怎么跑、本机数字从哪来.md)。有用请 **[Star](https://github.com/Yapeng-Gao/AI-System-Performance-Lab)**，方便后续章更新检索。

### 本机结论速览（RTX 5090 / `sm_120`）

| 章 | 一句话 |
|---|---|
| B-09 布局 | `touch_fields=1` 时 SoA / AoS ≈ **13.6×** |
| C-06 Graph | 短核链 `stream/graph` ≈ **3.7～4.1×**；`work=4096` → **1.01×** |
| C-05 Fusion | 瘦融合随链长 **3.9×→9.8×**；`fat` occupancy 6→1，相对瘦融合慢 **8×** |
| C-04 Sync | `grid/block` ≈ **17×**；`phases`≈1（少 sync ≠ 自动更快） |

全文在 `article/`；复现命令在下方。知乎 / CSDN / 掘金转载请保留本仓库链接。

---

## 怎么读

| 你想做的事 | 去哪 |
|---|---|
| 从哪读起 | [专栏导读](article/01_cuda_basic/00.%20专栏导读：怎么读、怎么跑、本机数字从哪来.md) |
| 读文章 | `article/01_cuda_basic/`、`article/02_memory_optim/`、`article/03_compute_primitives/` |
| 跑章节实验 | `examples/01_cuda_basics/`、`examples/02_memory_optim/`、`examples/03_compute_primitives/` |
| 看实测与 CSV | `docs/results/` |
| 重画实测图 | `python scripts/plot_b05_unified_memory.py` 等；C-01～C-06：`plot_c0N_*.py` |
| Module B 证据索引（B-10） | [`docs/results/B-10_checklist.md`](docs/results/B-10_checklist.md) |
| 专栏进度（导航） | [`docs/CUDA专栏规划.md`](docs/CUDA专栏规划.md) |
| 按章写作大纲 | [`docs/CUDA专栏大纲/`](docs/CUDA专栏大纲/README.md) |

```text
AI-System-Performance-Lab/
├── article/           # 专栏正文 + assets
├── examples/          # 章节 .cu（CMake 唯一构建目标）
│   ├── 01_cuda_basics/
│   ├── 02_memory_optim/
│   └── 03_compute_primitives/
├── docs/              # 规划 + results/
├── scripts/           # 绘图 / profile 辅助
├── cmake/
└── CMakeLists.txt
```

**当前主线**：导读已收进仓库；Module A + B 已收束；Module C：C-01～C-06 ✅。结构说明见 [`docs/仓库架构与现状.md`](docs/仓库架构与现状.md)。

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
./build/bin/03_compute_primitives_01_warp_primitives --mode sweep
./build/bin/03_compute_primitives_02_cooperative_groups --mode sweep
./build/bin/03_compute_primitives_03_atomics_contention --mode sweep
./build/bin/03_compute_primitives_04_sync_layers --mode sweep
./build/bin/03_compute_primitives_04_sync_layers --mode sweep_grid
./build/bin/03_compute_primitives_04_sync_layers --mode modes
./build/bin/03_compute_primitives_05_kernel_fusion --mode sweep
./build/bin/03_compute_primitives_05_kernel_fusion --mode modes
./build/bin/03_compute_primitives_06_cuda_graph --mode sweep
./build/bin/03_compute_primitives_06_cuda_graph --mode sweep_work
./build/bin/03_compute_primitives_06_cuda_graph --mode modes
```

重画实测图：

```bash
python scripts/plot_b05_unified_memory.py
python scripts/plot_b06_pinned_dma.py
python scripts/plot_b07_cp_async.py
python scripts/plot_c01_warp_primitives.py
python scripts/plot_c02_cooperative_groups.py
python scripts/plot_c03_atomics_contention.py
python scripts/plot_c06_cuda_graph.py
```

## 专栏映射

| 模块 | 路径 | 状态 |
| :--- | :--- | :--- |
| **导读** | `article/01_cuda_basic/00. 专栏导读*` | ✅ |
| **Module A** | `article/01_cuda_basic` + `examples/01_cuda_basics` | ✅ |
| **Module B** | `article/02_memory_optim` + `examples/02_memory_optim` | ✅ B-01～B-10 |
| **Module C** | `article/03_compute_primitives` + `examples/03_compute_primitives` | 🟡 C-01～C-06 ✅；C-07～C-10 规划 |
| **Module D–E** | 仅规划文档 | ⏳ |

正文插图：原理短 ASCII，实测 matplotlib（见架构文档 §4）。

## 规划文档

- [`docs/CUDA专栏规划.md`](docs/CUDA专栏规划.md)（导航总表）
- [`docs/CUDA专栏大纲/`](docs/CUDA专栏大纲/README.md)（按章大纲）
- [`docs/仓库架构与现状.md`](docs/仓库架构与现状.md)
- 远期：`docs/昇腾CANN专栏规划.md` 等

## 许可证

GNU GPL v3.0 — 见 `LICENSE`。
