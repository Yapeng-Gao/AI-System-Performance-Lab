# Module B: 内存优化

本目录包含 CUDA 内存优化相关的示例代码。

## 核心内容

- **Coalescing (合并访问)**: 优化全局内存访问模式
- **Bank Conflict**: Shared Memory 访问优化
- **TMA (Tensor Memory Accelerator)**: Hopper 架构的新特性
- **Pinned Memory**: 零拷贝内存管理

## 计划示例

- `01_coalescing.cu` - 合并访问示例
- `02_bank_conflict.cu` - Shared Memory 优化
- `03_tma_basics.cu` - TMA 基础用法 (Hopper 架构)

## 运行说明

```bash
cd build
cmake ..
cmake --build . --parallel
./bin/02_memory_optim_<example_name>
```

