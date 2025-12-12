# Module C: 计算原语

本目录包含 CUDA 高级计算原语和优化技术。

## 核心内容

- **Warp Primitives**: `__shfl`, `__ballot`, `__syncwarp` 等
- **CUDA Graphs**: 异步执行图优化
- **Cooperative Groups**: 线程组协作
- **Tensor Core**: 矩阵运算加速

## 计划示例

- `01_warp_primitives.cu` - Warp 级原语使用
- `02_cuda_graphs.cu` - CUDA Graphs 示例
- `03_cooperative_groups.cu` - 协作组示例
- `04_tensor_core.cu` - Tensor Core 矩阵运算

## 运行说明

```bash
cd build
cmake ..
cmake --build . --parallel
./bin/03_compute_primitives_<example_name>
```

