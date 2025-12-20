#!/bin/bash

# 自动检测构建目录（支持 Windows/CLion 和 Linux 标准构建）
# Windows/CLion: cmake-build-debug 或 cmake-build-debug-visual-studio
# Linux: build
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 按优先级查找构建目录
BUILD_DIR=""
if [ -d "$PROJECT_ROOT/cmake-build-debug-visual-studio/bin" ]; then
    BUILD_DIR="$PROJECT_ROOT/cmake-build-debug-visual-studio/bin"
elif [ -d "$PROJECT_ROOT/cmake-build-debug/bin" ]; then
    BUILD_DIR="$PROJECT_ROOT/cmake-build-debug/bin"
elif [ -d "$PROJECT_ROOT/build/bin" ]; then
    BUILD_DIR="$PROJECT_ROOT/build/bin"
else
    echo "Error: Build directory not found!"
    echo "Please build the project first using one of:"
    echo "  - Linux: mkdir build && cd build && cmake .. && cmake --build ."
    echo "  - Windows/CLion: Build from IDE or cmake-build-debug directory"
    exit 1
fi

TARGET_BIN="$BUILD_DIR/01_cuda_basics_10_roofline_demo"

# Windows 下检查 .exe 扩展名
if [ -f "${TARGET_BIN}.exe" ]; then
    TARGET_BIN="${TARGET_BIN}.exe"
fi

if [ ! -f "$TARGET_BIN" ]; then
    echo "Error: Binary not found at $TARGET_BIN"
    echo "Please build the project first!"
    exit 1
fi

echo "Using binary: $TARGET_BIN"
echo ""

OUTPUT_REP="roofline_report"

echo "=========================================================="
echo "   Profiling Roofline with Nsight Compute (ncu)"
echo "=========================================================="
echo "Output: ${OUTPUT_REP}.ncu-rep"
echo ""

# ncu 指令说明:
# --set roofline : 开启 Roofline 分析集合
# --section SpeedOfLight_RooflineChart : 显式指定收集 Roofline 图表数据
# --force-overwrite : 覆盖旧报告
ncu --set roofline \
    --section SpeedOfLight_RooflineChart \
    --force-overwrite \
    --output "$OUTPUT_REP" \
    "$TARGET_BIN"

echo ""
echo "=========================================================="
echo "Done!"
echo "Please open '${OUTPUT_REP}.ncu-rep' in Nsight Compute GUI."
echo "You will see two dots on the chart:"
echo "  1. One hitting the sloped ceiling (Memory Bound)"
echo "  2. One hitting the flat ceiling (Compute Bound)"
echo "=========================================================="