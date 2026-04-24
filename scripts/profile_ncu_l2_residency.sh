#!/usr/bin/env bash
set -euo pipefail

# Profile the Module B-14 example (L2 residency control).
#
# Output: CSV files under docs/results/ncu/
# Usage:
#   cd build
#   cmake --build . --target 02_memory_optim_04_l2_residency --parallel
#   cd ..
#   chmod +x scripts/profile_ncu_l2_residency.sh
#   ./scripts/profile_ncu_l2_residency.sh
#
# Optional env vars:
#   DATA_MB=64 ITERS=2048 SET_ASIDE_MB=8 WINDOW_MB=32 HIT_RATIO=0.25 ./scripts/profile_ncu_l2_residency.sh

EXE="./build/bin/02_memory_optim_04_l2_residency"
OUT_DIR="docs/results/ncu"
mkdir -p "${OUT_DIR}"

DATA_MB="${DATA_MB:-64}"
ITERS="${ITERS:-2048}"
SET_ASIDE_MB="${SET_ASIDE_MB:-8}"
WINDOW_MB="${WINDOW_MB:-32}"
HIT_RATIO="${HIT_RATIO:-0.25}"

if [ ! -x "${EXE}" ]; then
  echo "Executable not found: ${EXE}" >&2
  echo "Hint: build it first from build/: cmake --build . --target 02_memory_optim_04_l2_residency --parallel" >&2
  exit 1
fi

# Minimal evidence trio + a few helpers.
# Notes:
# - metric availability varies by GPU + Nsight Compute version; if one metric is missing,
#   remove it or select an equivalent in NCU UI sections.
METRICS=(
  gpu__time_duration.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_sectors_hit_rate.pct
)

METRIC_STR=$(IFS=, ; echo "${METRICS[*]}")

tag="l2res_data${DATA_MB}mb_it${ITERS}_sa${SET_ASIDE_MB}mb_win${WINDOW_MB}mb_hr${HIT_RATIO}"
out="${OUT_DIR}/example_02_memory_optim_04_l2_residency_${tag}.csv"

echo "==== Profiling 02_memory_optim_04_l2_residency ===="
echo "Params: data_mb=${DATA_MB}, iters=${ITERS}, set_aside_mb=${SET_ASIDE_MB}, window_mb=${WINDOW_MB}, hit_ratio=${HIT_RATIO}"
echo "Output: ${out}"

ncu --metrics "${METRIC_STR}" \
  --csv \
  --target-processes all \
  "${EXE}" "${DATA_MB}" "${ITERS}" "${SET_ASIDE_MB}" "${WINDOW_MB}" "${HIT_RATIO}" \
  > "${out}"

echo "NCU profiling done."

