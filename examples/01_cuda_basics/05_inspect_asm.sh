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

TARGET_BIN="$BUILD_DIR/01_cuda_basics_05_kernel_structure"

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
echo "   SASS Analysis: Inlining Behavior"
echo "========================================================"
echo "Searching for 'CAL' (Call) instructions in Device Code..."
echo "If __noinline__ works, you should see CAL instructions below:"
echo ""

# 使用 cuobjdump 反汇编 SASS，并查找函数调用
# grep -C 2 显示匹配行的上下 2 行
cuobjdump -sass "$TARGET_BIN" | grep "CAL" -C 2

echo ""
echo "--------------------------------------------------------"
echo "NOTE:"
echo "1. 'CAL' instruction means a subroutine call (no-inline)."
echo "2. If you don't see CAL for forceinline_kernel, it was successfully inlined."
echo "========================================================"

echo ""
echo "========================================================"
echo "   SASS Analysis: Parameter Loading (Constant Memory)"
echo "========================================================"
echo "Searching for Constant Memory loads (c[0x0])..."
echo "These instructions move kernel arguments from Bank 0 to Registers:"
echo ""

# 查找从 Constant Bank 0 读取数据的指令 (MOV R1, c[0x0][...])
cuobjdump -sass "$TARGET_BIN" | grep "c\[0x0\]" | head -n 10

echo ""
echo "... (showing first 10 occurrences)"
echo "========================================================"