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

TARGET_BIN="$BUILD_DIR/01_cuda_basics_09_debug_and_sanitizer"

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

# ============================================================
# 开始 Sanitizer 测试
# ============================================================

echo ""
echo "=========================================================="
echo "   CASE 1: Detecting Out-of-Bounds Access (Memcheck)"
echo "=========================================================="
# 使用 --tool memcheck (默认工具)
# --leak-check full: 同时检查内存泄漏
compute-sanitizer --tool memcheck --leak-check full "$TARGET_BIN" 0

echo ""
echo "=========================================================="
echo "   CASE 2: Detecting Data Race (Racecheck)"
echo "=========================================================="
# 使用 --tool racecheck
compute-sanitizer --tool racecheck "$TARGET_BIN" 1

echo ""
echo "=========================================================="
echo "   CASE 3: Detecting Illegal Sync (Synccheck)"
echo "=========================================================="
# 使用 --tool synccheck
compute-sanitizer --tool synccheck "$TARGET_BIN" 2

echo ""
echo "Done. Analyze the output above to find the bugs."