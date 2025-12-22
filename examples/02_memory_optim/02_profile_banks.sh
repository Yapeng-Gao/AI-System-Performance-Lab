#!/bin/bash

# Shared Memory Bank Conflict Profiling Script
# 对应示例：examples/02_memory_optim/02_shared_mem_bank_conflict.cu

# 自动检测构建目录（支持 Windows/CLion 和 Linux 标准构建）
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

# 可执行文件名需与 CMake 保持一致，如有修改请同步这里
TARGET_BIN="$BUILD_DIR/02_memory_optim_02_shared_mem_bank_conflict"

# Windows 下检查 .exe 扩展名
if [ -f "${TARGET_BIN}.exe" ]; then
    TARGET_BIN="${TARGET_BIN}.exe"
fi

if [ ! -f "$TARGET_BIN" ]; then
    echo "Error: Binary not found at $TARGET_BIN"
    echo "Please build the project first!"
    exit 1
fi

OUTPUT_REP="smem_conflict_report"

echo "=========================================================="
echo "   Profiling Shared Memory Bank Conflicts (Nsight Compute)"
echo "=========================================================="
echo "Metrics to watch:"
echo "  - l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
echo "    (Ideally ~1 per instruction. 32 means 32-way conflict)"
echo "  - smsp__inst_executed.sum"
echo ""

ncu --metrics \
    l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__inst_executed.sum \
    --force-overwrite \
    --output "$OUTPUT_REP" \
    "$TARGET_BIN"

echo ""
echo "Done. Open ${OUTPUT_REP}.ncu-rep and check 'Source' / 'Details' view."
echo "Expectations:"
echo "  - Naive   Kernel : ~32 wavefronts per shared load (heavy conflict)"
echo "  - Padding Kernel : ~1 wavefront per shared load"
echo "  - Swizzle Kernel : ~1 wavefront per shared load (compact layout)"