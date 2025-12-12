# Python C++ 扩展源码

本目录包含 Python 绑定的 C++ 扩展源码。

## 计划内容

- 使用 pybind11 包装 C++ 接口
- 将 `aspl_core` 库暴露给 Python

## 文件结构

```
csrc/
├── module.cpp      # pybind11 模块定义
├── bindings/       # 各模块的绑定代码
│   ├── ops.cpp
│   └── profiler.cpp
└── CMakeLists.txt  # Python 扩展构建配置
```

