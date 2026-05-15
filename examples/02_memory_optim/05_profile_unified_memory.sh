#!/bin/bash
# B-05 Unified Memory：NSYS 一键采集 fault / prefetch / advise 证据链
# 对应示例：examples/02_memory_optim/05_unified_memory_pf.cu
#
# 用法：
#   bash 05_profile_unified_memory.sh              # 采集 A/B/C 三组
#   bash 05_profile_unified_memory.sh fault        # 仅采集 fault
#   bash 05_profile_unified_memory.sh prefetch advise
#
# 输出：当前目录下 um_<mode>_trace.nsys-rep（及可选 .sqlite）

set -euo pipefail

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
    echo "Error: Build directory not found!"
    echo "Please build the project first:"
    echo "  mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --parallel"
    exit 1
fi

TARGET_BIN="$BUILD_DIR/02_memory_optim_05_unified_memory_pf"
if [ -f "${TARGET_BIN}.exe" ]; then
    TARGET_BIN="${TARGET_BIN}.exe"
fi

if [ ! -f "$TARGET_BIN" ]; then
    echo "Error: Binary not found at $TARGET_BIN"
    echo "Hint: after adding new .cu files, run 'cmake ..' then rebuild."
    exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "Error: nsys not found. Install Nsight Systems and ensure it is on PATH."
    exit 1
fi

# 与文章 SOP / README 默认参数对齐
N="${N:-16777216}"
ITERS="${ITERS:-32}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
DEVICE="${DEVICE:-0}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR}"

ALL_MODES=(fault prefetch advise)
if [ "$#" -gt 0 ]; then
    MODES=("$@")
else
    MODES=("${ALL_MODES[@]}")
fi

for m in "${MODES[@]}"; do
    case "$m" in
        fault|prefetch|advise) ;;
        *)
            echo "Error: unknown mode '$m' (expected fault|prefetch|advise)"
            exit 1
            ;;
    esac
done

mkdir -p "$OUT_DIR"

COMMON_ARGS=(--n "$N" --iters "$ITERS" --runs "$RUNS" --warmup "$WARMUP" --device "$DEVICE")

echo "Using binary: $TARGET_BIN"
echo "Output dir:   $OUT_DIR"
echo "Params:       n=$N iters=$ITERS runs=$RUNS warmup=$WARMUP device=$DEVICE"
echo ""

profile_one() {
    local mode="$1"
    local stem="um_${mode}_trace"
    local rep="${OUT_DIR}/${stem}.nsys-rep"

    echo "=========================================================="
    echo "  NSYS: mode=${mode}"
    echo "=========================================================="

    nsys profile \
        --trace=cuda,osrt,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --output="${OUT_DIR}/${stem}" \
        --force-overwrite=true \
        "$TARGET_BIN" --mode "$mode" "${COMMON_ARGS[@]}"

    echo "  -> ${rep}"
    echo ""

    if command -v nsys >/dev/null 2>&1; then
        echo "  [stats] CUDA API summary (grep UVM / Mem / Prefetch):"
        nsys stats --report cuda_api_sum "${rep}" 2>/dev/null \
            | grep -iE 'Mem|Prefetch|Advise|Managed|UVM|Migrate|Fault' \
            || echo "    (no matching API lines; open .nsys-rep in GUI)"
        echo ""
    fi
}

echo "=========================================================="
echo "  B-05 Unified Memory — Nsight Systems batch profile"
echo "=========================================================="
echo "In Nsight Systems GUI, open each .nsys-rep and check:"
echo "  - CUDA HW: UVM page fault / page migration clusters"
echo "  - cudaMemPrefetchAsync / cudaMemAdvise vs kernel overlap"
echo "  - Whether faults occur inside kernel critical path (thrash)"
echo ""

for mode in "${MODES[@]}"; do
    profile_one "$mode"
done

echo "=========================================================="
echo "Done. Reports:"
for mode in "${MODES[@]}"; do
    echo "  ${OUT_DIR}/um_${mode}_trace.nsys-rep"
done
echo ""
echo "Quick timing (no profiler):"
for mode in "${MODES[@]}"; do
    "$TARGET_BIN" --mode "$mode" "${COMMON_ARGS[@]}" --csv-only
done
echo "=========================================================="
