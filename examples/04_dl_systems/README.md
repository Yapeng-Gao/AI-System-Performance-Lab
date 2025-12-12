# Module E: 深度学习系统

本目录包含大模型推理系统相关的实现。

## 核心内容

- **vLLM PagedAttention**: 分页注意力机制
- **Triton Kernels**: 自定义 GPU Kernel 开发
- **KV Cache 优化**: 推理加速技术
- **量化推理**: FP8/INT8 支持

## 计划示例

- `01_paged_attention.cu` - vLLM PagedAttention 实现
- `02_triton_basics.py` - Triton 基础示例
- `03_kv_cache_optim.cu` - KV Cache 优化

## 运行说明

### CUDA 示例
```bash
cd build
cmake ..
cmake --build . --parallel
./bin/04_dl_systems_<example_name>
```

### Python/Triton 示例
```bash
cd python
python -m examples.04_dl_systems.02_triton_basics
```

