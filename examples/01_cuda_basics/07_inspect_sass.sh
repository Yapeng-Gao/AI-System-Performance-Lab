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

TARGET_BIN="$BUILD_DIR/01_cuda_basics_07_memory_spaces"

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

echo "========================================================"
echo "   SASS Analysis: Local Memory Spilling"
echo "========================================================"
echo "Searching for Local Store (STL) and Local Load (LDL) instructions..."
echo "These indicate data is spilling to HBM (Slow!)."
echo ""

# 查找 STL/LDL 指令，并显示所属的函数名
# 预期在 force_local_memory_spill 函数中大量出现
cuobjdump -sass "$TARGET_BIN" | grep -E "STL|LDL" -B 5 | head -n 20

echo ""
echo "NOTE: If you see STL/LDL inside 'force_local_memory_spill', spilling occurred."
echo "========================================================"

echo ""
echo "========================================================"
echo "   SASS Analysis: __restrict__ Optimization"
echo "========================================================"
echo "Comparing No-Restrict vs With-Restrict kernels..."
echo "Ideally, 'With-Restrict' might use LDG.NC (Non-coherent/Texture) or fewer instructions."
echo ""

# 简单列出两个函数的指令，人工对比（自动化对比比较复杂）
echo "[Functions found in binary]"
cuobjdump -sass "$TARGET_BIN" | grep "Function :" | grep "add_"

echo ""
echo "Tip: Use 'cuobjdump -sass ... > out.txt' to manually compare the assembly."
echo "========================================================"