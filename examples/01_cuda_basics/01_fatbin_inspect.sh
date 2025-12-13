#!/bin/bash

# 定义生成的可执行文件路径 (根据你的 CMake 输出路径调整)
# Windows 通常是 build/bin/01_cuda_basics_01_hello_modern.exe
# Linux 通常是 build/bin/01_cuda_basics_01_hello_modern
TARGET_BIN="../../build/bin/01_cuda_basics_01_hello_modern"

if [ -f "${TARGET_BIN}.exe" ]; then
    TARGET_BIN="${TARGET_BIN}.exe"
fi

if [ ! -f "$TARGET_BIN" ]; then
    echo "Error: Binary not found at $TARGET_BIN"
    echo "Please build the project first!"
    exit 1
fi

echo "=== 1. Inspecting Virtual Architectures (PTX) ==="
echo "PTX is just-in-time compiled by the driver."
cuobjdump "$TARGET_BIN" -ptx | grep "arch =" | sort | uniq

echo ""
echo "=== 2. Inspecting Real Architectures (SASS) ==="
echo "SASS is the actual machine code running on silicon."
cuobjdump "$TARGET_BIN" -sass | grep "arch =" | sort | uniq

echo ""
echo "Done."