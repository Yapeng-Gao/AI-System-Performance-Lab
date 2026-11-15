# examples/04_operators

Module D 配套实验。正文在 `article/04_operators/`。

| 章节 | 文件 | 主命令 | 结果 |
|---|---|---|---|
| **D-01** | `01_device_reduce.cu` | `--mode sweep` | float；5090 大数组 **1.15×**，`vec4/cub`≈1.00。单路径用 `--mode vec4` |
| **D-02** | `02_dot.cu` | `--mode sweep` | 点积；大数组 `vec4/cublas`≈1.00，`scalar/vec4`≈1.02× |
| **D-03** | `03_elementwise.cu` | `--mode sweep` | 逐元素加；~1.55 TB/s，`scalar/vec4`≈1.03×。短章激活共用口径 |

增删 `.cu` 后先重跑 `cmake`，再 build。5090 上建议 `-DCMAKE_CUDA_ARCHITECTURES=120`。
