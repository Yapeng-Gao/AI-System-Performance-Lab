# Module A: CUDA 基础架构

本目录包含 CUDA 核心概念和架构相关的示例代码，是《AI 系统性能工程》专栏 Module A 的配套实战代码。

## 📚 目录

| 章节 | 文件 | 核心内容 | 知识点 |
|------|------|----------|--------|
| **第 1 章** | `01_hello_modern.cu` | CUDA 核心概念总览 | Grid-Block-Thread 模型、错误检查、异步执行、Unified Memory |
| **第 2 章** | `02_hardware_query.cu` | GPU 硬件架构深度解析 | SM 架构、内存层次、L2 Cache、Tensor Core 能力、带宽分析 |
| **第 3 章** | `03_*.cu` | *待添加* | *待补充* |

## 🚀 快速开始

### 编译构建

```bash
# 在项目根目录
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8
```

### 运行示例

编译成功后，可执行文件位于 `build/bin` 目录下：

```bash
# 运行第 1 章示例
./bin/01_cuda_basics_01_hello_modern

# 运行第 2 章示例
./bin/01_cuda_basics_02_hardware_query
```

---

## 📖 各章详细说明

### 第 1 章：CUDA 核心概念总览 (`01_hello_modern.cu`)

**现代版 Hello World**：展示 CUDA 12+ 的工程规范和生产级代码特征。

#### 核心知识点

1. **宏定义封装**：使用 `CUDA_CHECK` 宏包裹所有 CUDA API 调用，确保在出错时能打印具体的文件名和行号，并安全退出。这种 Fail Fast 策略在生产环境中至关重要。

2. **硬件感知**：在 Kernel 内部通过 `__CUDA_ARCH__` 宏判断当前硬件架构（编译期常量），同时在 Host 端使用 `cudaGetDeviceProperties` 进行运行时硬件查询。

3. **异步执行与同步**：演示 Kernel 启动的异步特性，以及 `cudaGetLastError()` 和 `cudaDeviceSynchronize()` 的正确使用。

4. **统一内存管理**：使用 `cudaMallocManaged` 简化内存分配，数据会在 CPU/GPU 间自动按需迁移（Page Migration）。

5. **线程索引计算**：展示 CUDA 的 Grid-Block-Thread 三层执行模型，通过 `global_id = blockIdx.x * blockDim.x + threadIdx.x` 计算全局线程索引。

#### 预期输出

```
[Host] Starting Modern CUDA Hello World...
[Host] GPU Name: NVIDIA GeForce RTX 4090
[Host] SM Count: 128
[Host] Compute Capability: 8.9
[Host] Launching Kernel...
[Device] Kernel running on SM arch sm_890
[Device] GridDim=1, BlockDim=32
[Host] Verifying results...
[Host] Verification PASSED! [OK]
```

#### 二进制分析工具

项目提供了 `01_fatbin_inspect.sh` 脚本，利用 `cuobjdump` 工具分析编译后的二进制文件：

```bash
cd examples/01_cuda_basics
bash 01_fatbin_inspect.sh
```

该脚本可以展示：
- **PTX（虚拟架构）**：中间表示代码，由驱动程序在运行时 JIT 编译到目标架构
- **SASS（真实架构）**：实际运行在 GPU 上的机器码

这可以验证 CMake 配置中的 `CMAKE_CUDA_ARCHITECTURES` 是否生效。

---

### 第 2 章：GPU 硬件架构深度解析 (`02_hardware_query.cu`)

**硬件拓扑侦探**：挖掘 SM 架构、L2 Cache、Tensor Core 能力与带宽极限。

#### 核心知识点

1. **计算能力与架构识别**：根据 Compute Capability 识别 GPU 架构（Hopper、Ampere、Volta 等），并推算每个 SM 的 CUDA Core 数量。

2. **SM 宏观拓扑**：展示多处理器数量、CUDA Core 总数等宏观计算资源。

3. **内存体系分析**：
   - 全局内存容量（HBM/DDR）
   - 内存总线宽度
   - 理论峰值带宽计算
   - **L2 Cache 大小**（关键性能指标，影响数据驻留控制）

4. **SM 微架构资源**（Occupancy 分析关键）：
   - Shared Memory 限制（每 Block 和动态分配）
   - 寄存器数量限制
   - 线程数限制（每 Block 和每 SM）
   - Warp 大小

5. **现代特性支持侦测**：
   - Unified Addressing (UVA)
   - Managed Memory (Page Migration)
   - TMA (Tensor Memory Accelerator) - Hopper 架构
   - Thread Block Clusters - Hopper 架构

#### 预期输出

```
=================================================================
   AI System Performance Lab - Hardware Topology Detective   
=================================================================
Detected 1 CUDA Capable Device(s)

[Device 0]: NVIDIA GeForce RTX 4090
-----------------------------------------------------------------
  [Architecture]
    Compute Capability      : 8.9 (Ampere / Ada class)
  [Compute Topology]
    Multiprocessors (SMs)   : 128
    CUDA Cores / SM         : 128
    Total CUDA Cores        : 16384
    GPU Clock Rate          : N/A (removed in CUDA 12+)
  [Memory Hierarchy]
    Global Memory (HBM/DDR) : 24.00 GB
    Memory Bus Width        : 384-bit
    Memory Clock Rate       : N/A (removed in CUDA 12+)
    Theoretical Bandwidth   : N/A (use nvml API for accurate value)
    L2 Cache Size           : 72.00 MB (Key for residency control)
  [SM Micro-Architecture]
    Max Shared Mem / Block  : 48.00 KB
    Max Shared Mem (Opt-in) : 164.00 KB (Dynamic)
    Max Registers / Block   : 65536
    Max Threads / Block     : 1024
    Max Threads / SM        : 1536
    Warp Size               : 32
  [Modern Features Support]
    Unified Addressing      : Yes
    Managed Memory          : Yes
    TMA (Tensor Mem Accel)  : No
    Thread Block Clusters   : No
```

#### 注意事项

- 在 CUDA 12+ 版本中，`clockRate` 和 `memoryClockRate` 字段已被移除，代码使用条件编译兼容新旧版本。
- 如需获取准确的时钟频率和带宽信息，建议使用 NVML (NVIDIA Management Library) API。

---

### 第 3 章：*待添加*

*本章内容待补充...*

---

## 🔧 工具脚本

- `01_fatbin_inspect.sh`：二进制文件分析工具，用于查看 PTX 和 SASS 代码

## 📝 注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台
- 兼容 CUDA 12.0+ 版本（部分字段在 CUDA 12+ 中已移除，使用条件编译处理）
