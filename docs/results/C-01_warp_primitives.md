# C-01 Warp Primitives — 实测摘要

> 状态：**待测**。正文骨架见 `article/03_compute_primitives/C-01*.md`。
>
> 复现后把 `--mode sweep` / `modes` 输出贴回，本文件将补全平台、表与怎么读。

## 服务器复现（建议）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target 03_compute_primitives_01_warp_primitives -j
# 可执行文件常见路径：build/bin/ 或 build/bin/Release/

./build/bin/03_compute_primitives_01_warp_primitives --mode sweep
./build/bin/03_compute_primitives_01_warp_primitives --mode sweep --csv-only
./build/bin/03_compute_primitives_01_warp_primitives --mode modes
```

把完整终端输出（含 GPU / sm_XX 启动行）贴回即可回填正文 TL;DR 与本表。

## 主证据口径

- CUDA event **median**（默认 warmup=2, runs=7）
- 主曲线：`shfl/smem` vs `nwarps∈{1,2,4,8,16,32}`
- 正确性：float/int reduce 与 host 参考和比对；ballot 奇数计数

## 表（待填）

（贴输出后回填）
