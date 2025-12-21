# Module B: 内存优化

本目录包含 CUDA 内存优化相关的示例代码，是《AI 系统性能工程》专栏 Module B 的配套实战代码。

## 📚 目录

| 章节 | 文件 | 核心内容 | 知识点 |
|------|------|----------|--------|
| **第 11 章** | `01_global_mem_bandwidth.cu` | Global Memory 极致优化 | HBM 对齐、向量化访问 (float4)、L2 Cache 驻留控制、Coalescing 验证 |

## 🚀 快速开始

### 编译构建

#### Linux 环境
```bash
# 在项目根目录
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8
```

#### Windows/CLion 环境
- 使用 CLion 直接构建（构建输出在 `cmake-build-debug` 或 `cmake-build-debug-visual-studio` 目录）
- 或手动构建：
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --parallel 8
```

### 运行示例

编译成功后，可执行文件位置：
- **Linux**: `build/bin/` 目录
- **Windows/CLion**: `cmake-build-debug/bin` 或 `cmake-build-debug-visual-studio/bin` 目录

```bash
# Linux: 在 build 目录下运行
./bin/02_memory_optim_01_global_mem_bandwidth

# Windows/CLion: 在 cmake-build-debug/bin 目录下运行
# 或在 PowerShell 中（从项目根目录）
.\cmake-build-debug\bin\02_memory_optim_01_global_mem_bandwidth.exe
```

---

## 📖 各章详细说明

### 第 11 章：Global Memory 极致优化 (`01_global_mem_bandwidth.cu`)

**Bandwidth Micro-Benchmark**：验证 HBM 对齐、向量化访问 (float4) 与 L2 Cache 驻留控制。

#### 核心知识点

1. **Misaligned Access（错位访问）**：
   - 故意制造错位访问（Offset=1），破坏 Coalescing 机制
   - 一个 Warp 的请求会分裂成多个 Memory Transactions
   - 导致带宽利用率大幅下降
   - 演示了内存对齐对性能的严重影响

2. **Aligned Scalar Copy（对齐标量拷贝）**：
   - 标准的合并访问模式，所有线程访问连续对齐的内存
   - 使用 `float` 类型，生成 `LDG.E.32` 指令
   - 作为性能基准线

3. **Vectorized Copy（向量化拷贝）**：
   - 使用 `float4` 类型，强制生成 `LDG.E.128` 指令
   - 减少 75% 的指令发射压力
   - 提升内存访问效率，接近硬件带宽上限

4. **L2 Persistence（L2 缓存驻留控制）**：
   - 使用 CUDA 11/12 的 `cudaStreamSetAttribute` API
   - 通过 `cudaAccessPropertyPersisting` 锁定 L2 Cache
   - 模拟深度学习中的 Weight Reuse 场景
   - 对比默认 LRU 策略与显式驻留策略的性能差异

#### 预期输出

```
Target GPU: NVIDIA GeForce RTX 4090 (L2 Cache: 72.00 MB)
Total Data: 64.00 MB

[Misaligned (Off=1)    ] Time:  0.234 ms | Bandwidth:  548.72 GB/s
[Aligned (Float)       ] Time:  0.156 ms | Bandwidth:  821.33 GB/s
[Vectorized (Float4)   ] Time:  0.145 ms | Bandwidth:  883.45 GB/s

=== L2 Persistence Control (Weight Reuse Simulation) ===
[L2 Default (LRU)      ] Time:  0.089 ms | Bandwidth:  224.72 GB/s
[L2 Persisting         ] Time:  0.045 ms | Bandwidth:  444.44 GB/s
>> L2 Control Improvement: 97.78%
```

#### 性能分析工具

项目提供了 `01_profile_bandwidth.sh` 脚本，使用 Nsight Compute 进行内存性能分析：

```bash
cd examples/02_memory_optim
bash 01_profile_bandwidth.sh
```

**注意**：
- 脚本会自动检测构建目录（支持 Windows/CLion 和 Linux 两种构建方式）
- **需要安装 Nsight Compute**（随 CUDA Toolkit 一起安装）
- 脚本会生成 `.ncu-rep` 文件，需要在 Nsight Compute GUI 中打开

该脚本可以：
- **验证向量化效果**：在 "SOL Memory" 部分查看向量化访问的效率
- **分析内存事务**：在 "Memory Analysis" 中查看 "Sectors/Request" 指标
- **对比不同访问模式**：观察 Misaligned vs Aligned vs Vectorized 的性能差异

#### 技术细节

- **Coalescing 机制**：当 Warp 中的线程访问连续对齐的内存时，硬件会将多个请求合并成一个 Memory Transaction
- **向量化访问**：使用 `float4` 等向量类型可以减少指令数量，提升指令发射效率
- **L2 Cache 驻留**：通过 `cudaStreamSetAttribute` 可以控制 L2 Cache 的替换策略，适合权重重用场景
- **数据规模**：使用 64MB 数据确保足够大以触达 HBM 带宽墙，避开 L2 Cache 的影响

#### 注意事项

- Misaligned Access 是性能杀手，应尽量避免非对齐的内存访问
- Vectorized Access 可以显著提升性能，但需要确保数据对齐（128-byte 对齐）
- L2 Persistence 功能需要 CUDA 11.0+ 和计算能力 8.0+（Ampere 及以上架构）
- 实际带宽会受到硬件限制、PCIe 带宽等多种因素影响，实测值可能低于理论峰值

---

## 🔧 工具脚本

- `01_profile_bandwidth.sh`：性能分析脚本（Linux/WSL 专用），使用 Nsight Compute 分析内存访问模式和带宽利用率

## 📝 注意事项

- 所有示例代码遵循 CUDA 12+ 规范
- 代码包含完整的错误检查机制
- 支持 Windows 和 Linux 平台
- 兼容 CUDA 11.0+ 版本（L2 Persistence 功能需要 CUDA 11.0+）

