#!/bin/bash
# Run the three A-09 planted bugs under Compute Sanitizer.
# Requires: compute-sanitizer on PATH (CUDA Toolkit).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BUILD_DIR=""
if [ -d "$PROJECT_ROOT/cmake-build-debug-visual-studio/bin" ]; then
    BUILD_DIR="$PROJECT_ROOT/cmake-build-debug-visual-studio/bin"
elif [ -d "$PROJECT_ROOT/cmake-build-debug/bin" ]; then
    BUILD_DIR="$PROJECT_ROOT/cmake-build-debug/bin"
elif [ -d "$PROJECT_ROOT/build/bin" ]; then
    BUILD_DIR="$PROJECT_ROOT/build/bin"
else
    echo "Error: Build directory not found. Build 01_cuda_basics_09_debug_and_sanitizer first."
    exit 1
fi

TARGET_BIN="$BUILD_DIR/01_cuda_basics_09_debug_and_sanitizer"
if [ -f "${TARGET_BIN}.exe" ]; then
    TARGET_BIN="${TARGET_BIN}.exe"
fi

if [ ! -f "$TARGET_BIN" ]; then
    echo "Error: Binary not found at $TARGET_BIN"
    exit 1
fi

if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "Error: compute-sanitizer not on PATH (install CUDA Toolkit)."
    exit 1
fi

echo "Using binary: $TARGET_BIN"
echo ""

echo "=========================================================="
echo "   CASE 0: OOB → memcheck"
echo "=========================================================="
compute-sanitizer --tool memcheck "$TARGET_BIN" 0

echo ""
echo "=========================================================="
echo "   CASE 1: SMEM race → racecheck"
echo "=========================================================="
compute-sanitizer --tool racecheck "$TARGET_BIN" 1

echo ""
echo "=========================================================="
echo "   CASE 2: illegal __syncwarp mask → synccheck"
echo "=========================================================="
compute-sanitizer --tool synccheck "$TARGET_BIN" 2

echo ""
echo "Done. Look for ERROR / Hazard lines naming the planted kernels."
