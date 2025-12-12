# Python 绑定与 Triton 算子

本目录包含 ASPL 的 Python 绑定和 Triton 自定义 Kernel。

## 目录结构

```
python/
├── aspl/              # Python 包主目录
│   ├── __init__.py
│   ├── triton_kernels/  # Triton Kernel 实现
│   └── bindings/        # C++ 扩展绑定
├── csrc/              # Python 扩展的 C++ 源码
└── setup.py           # 安装脚本
```

## 计划内容

- **Python Bindings**: 使用 pybind11 绑定 C++ 接口
- **Triton Kernels**: 自定义 GPU Kernel（如 PagedAttention）
- **PyTorch Integration**: 与 PyTorch 集成

## 安装说明

```bash
cd python
pip install -e .
```

## 使用示例

```python
import aspl

# 使用 C++ 绑定
result = aspl.ops.reduction(data)

# 使用 Triton Kernel
import aspl.triton_kernels.paged_attention as pa
output = pa.forward(query, key, value, ...)
```

