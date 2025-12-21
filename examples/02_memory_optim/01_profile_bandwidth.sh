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

TARGET_BIN="$BUILD_DIR/02_memory_optim_01_global_mem_bandwidth"

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

OUTPUT_REP="global_mem_report"

echo "=========================================================="
echo "   Profiling Memory Metrics with Nsight Compute"
echo "=========================================================="
echo "Looking for metrics:"
echo "  - gpu__dram_throughput (HBM Bandwidth)"
echo "  - l1tex__t_sectors_pipe_lsu_mem_global_op_ld (Global Load Sectors)"
echo "  - l1tex__data_pipe_lsu_wavefronts_mem_shared (Shared Mem Wavefronts)"
echo ""

# --set full : 收集所有指标 (包含 Memory Workload Analysis)
# 或者只收集特定 section 以加快速度
ncu --set full \
    --force-overwrite \
    --output "$OUTPUT_REP" \
    "$TARGET_BIN"

echo ""
echo "Done. Open ${OUTPUT_REP}.ncu-rep to see:"
echo "1. Verify 'SOL Memory' for Vectorized vs Misaligned."
echo "2. Check 'Sectors/Request' in Memory Analysis."