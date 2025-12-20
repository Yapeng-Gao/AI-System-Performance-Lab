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

TARGET_BIN="$BUILD_DIR/01_cuda_basics_08_async_pipeline"

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

OUTPUT_FILE="pipeline_trace"

echo "========================================================"
echo "   Profiling with Nsight Systems (nsys)"
echo "========================================================"
echo "Tracing CUDA API and GPU Workload..."
echo ""

# nsys profile 指令
# --trace=cuda,osrt: 追踪 CUDA API, OS Runtime
# --output: 输出文件名 (.nsys-rep)
# --force-overwrite: 覆盖旧文件
nsys profile \
    --trace=cuda,osrt \
    --output="$OUTPUT_FILE" \
    --force-overwrite=true \
    "$TARGET_BIN"

echo ""
echo "========================================================"
echo "Done! Please open '${OUTPUT_FILE}.nsys-rep' in Nsight Systems GUI."
echo "Look for the 'CUDA HW' row to see the overlap."
echo "========================================================"