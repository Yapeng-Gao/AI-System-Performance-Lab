# Module A: CUDA 基础架构

本目录包含 CUDA 核心概念和架构相关的示例代码。

## 核心实现逻辑

不同于简单的 Hello World，该示例展示了生产级代码的几个特征：

1. **宏定义封装**：使用 `CUDA_CHECK` 宏包裹所有 CUDA API 调用，确保在出错时能打印具体的文件名和行号，并安全退出。这种 Fail Fast 策略在生产环境中至关重要，能够快速定位问题。

2. **硬件感知**：在 Kernel 内部通过 `__CUDA_ARCH__` 宏判断当前硬件架构（编译期常量），从而执行不同的逻辑。同时在 Host 端使用 `cudaGetDeviceProperties` 进行运行时硬件查询，这在兼容多代显卡和大规模集群部署时非常有用。

3. **异步执行与同步**：演示了 Kernel 启动的异步特性，以及 `cudaGetLastError()` 和 `cudaDeviceSynchronize()` 的正确使用。`cudaGetLastError()` 捕获启动配置错误，`cudaDeviceSynchronize()` 确保在验证结果前 GPU 已经真正完成了计算，避免数据竞争。

4. **统一内存管理**：使用 `cudaMallocManaged` 简化内存分配，数据会在 CPU/GPU 间自动按需迁移（Page Migration）。虽然在极致优化时我们会改用手动管理（`cudaMalloc` + `cudaMemcpy`），但在原型开发阶段这能显著降低心智负担。

5. **线程索引计算**：展示了 CUDA 的 Grid-Block-Thread 三层执行模型，通过 `global_id = blockIdx.x * blockDim.x + threadIdx.x` 计算全局线程索引，这是所有 CUDA Kernel 的基础。

## 二进制分析

项目提供了一个脚本 `02_fatbin_inspect.sh`，利用 `cuobjdump` 工具分析编译后的二进制文件。运行该脚本，你可以直观地看到：

- **PTX（虚拟架构）**：中间表示代码，由驱动程序在运行时 JIT 编译到目标架构
- **SASS（真实架构）**：实际运行在 GPU 上的机器码

这可以验证 CMake 配置中的 `CMAKE_CUDA_ARCHITECTURES` 是否生效，以及你的程序中包含了哪些架构的代码。

## 运行示例

```bash
# 编译
cd build
cmake ..
cmake --build . --parallel

# 运行程序
./bin/01_cuda_basics_01_hello_modern

# 分析二进制（Linux/Mac）
cd examples/01_cuda_basics
bash 02_fatbin_inspect.sh
```

## 预期输出

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

