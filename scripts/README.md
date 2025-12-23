# 脚本工具

本目录包含各种辅助脚本工具。

## 已提供的脚本

- `dump_sass.sh`        ：为 benchmarks 下的核心基准生成 cubin + SASS（按 sm_80/90/100）
- `profile_ncu.sh`      ：用 Nsight Compute 采集带宽/算力/指令等指标，输出 CSV 到 `docs/results/ncu`
- `parse_roofline.py`   ：解析 NCU CSV，计算带宽 / TFLOPs / OI
- `plot_roofline.py`    ：根据 JSON + 硬件屋顶线数据绘制 Roofline 图

## 使用说明

```bash
# 1) 确保基准可执行已构建 (位于 build/bin)
cd build
cmake --build . --parallel

# 2) 导出 SASS
cd ..
chmod +x scripts/*.sh
./scripts/dump_sass.sh

# 3) 采集 NCU 指标（需要 Nsight Compute CLI）
./scripts/profile_ncu.sh

# 4) 解析单个 NCU CSV
python scripts/parse_roofline.py docs/results/ncu/bench_cp_async_pipeline.csv

# 5) 绘制 Roofline 图
python scripts/plot_roofline.py data.json "GPU-Name" <BW_GBps> <TFLOPs_peak> output.png
```

