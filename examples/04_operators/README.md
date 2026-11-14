# examples/04_operators

Module D 配套实验。正文在 `article/04_operators/`。

| 章节 | 文件 | 主命令 | 结果 |
|---|---|---|---|
| **D-01** | `01_device_reduce.cu` | `--mode sweep` | float；5090 大数组 **1.15×**，`vec4/cub`≈1.00。单路径用 `--mode vec4` |

增删 `.cu` 后先重跑 `cmake`，再 build。5090 上建议 `-DCMAKE_CUDA_ARCHITECTURES=120`。
