#!/usr/bin/env bash
# C-01: bare sweep/modes (+ optional NCU for smem vs shfl / redux)
# Main evidence remains bare --mode sweep / --mode modes (CUDA event median).
# Usage:
#   bash examples/03_compute_primitives/01_profile_warp.sh
#   bash examples/03_compute_primitives/01_profile_warp.sh sweep
#   DO_NCU=1 bash examples/03_compute_primitives/01_profile_warp.sh ncu-only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=""
for cand in \
  "${ROOT_DIR}/build/bin/03_compute_primitives_01_warp_primitives" \
  "${ROOT_DIR}/cmake-build-debug/bin/03_compute_primitives_01_warp_primitives" \
  "${ROOT_DIR}/cmake-build-release/bin/03_compute_primitives_01_warp_primitives" \
  "${ROOT_DIR}/build/bin/Release/03_compute_primitives_01_warp_primitives.exe" \
  "${ROOT_DIR}/cmake-build-debug/bin/03_compute_primitives_01_warp_primitives.exe"
do
  if [[ -x "${cand}" ]] || [[ -f "${cand}" ]]; then
    BIN="${cand}"
    break
  fi
done

if [[ -z "${BIN}" ]]; then
  echo "ERROR: 03_compute_primitives_01_warp_primitives not found. Build first (re-run cmake)." >&2
  exit 1
fi

N="${N:-16777216}"
NWARPS="${NWARPS:-8}"
NCU_NWARPS="${NCU_NWARPS:-32}"   # NCU 默认盯加速比最大处
GRID="${GRID:-0}"               # 0 = binary auto
RUNS="${RUNS:-7}"
WARMUP="${WARMUP:-2}"
DEVICE="${DEVICE:-0}"
MODE_FILTER="${1:-all}"         # all | sweep | modes | smem | shfl | redux | ballot | ncu-only
DO_NCU="${DO_NCU:-0}"

# inst + shared traffic + barrier / scoreboard stalls（旁证；勿读附着墙钟）
NCU_METRICS="smsp__inst_executed.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__warps_issue_stalled_barrier.sum,smsp__warps_issue_stalled_long_scoreboard.sum,sm__sass_inst_executed_op_shared_ld.sum,sm__sass_inst_executed_op_shared_st.sum"

if [[ "${MODE_FILTER}" == "ncu-only" ]]; then
  DO_NCU=1
  modes=()
else
  modes=(sweep modes)
  if [[ "${MODE_FILTER}" != "all" ]]; then
    modes=("${MODE_FILTER}")
  fi
fi

run_one() {
  local mode="$1"
  echo "======== mode=${mode} ========"
  local args=(
    --mode "${mode}"
    --n "${N}"
    --nwarps "${NWARPS}"
    --runs "${RUNS}"
    --warmup "${WARMUP}"
    --device "${DEVICE}"
  )
  if [[ "${GRID}" != "0" ]]; then
    args+=(--grid "${GRID}")
  fi
  "${BIN}" "${args[@]}"
  echo
}

if ((${#modes[@]} > 0)); then
  for m in "${modes[@]}"; do
    run_one "${m}"
  done
fi

if [[ "${DO_NCU}" != "1" ]]; then
  exit 0
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "WARNING: ncu not found; skip profiling." >&2
  exit 0
fi

# verify launch + warmup×2 + 1 timed → skip 3 对准最后一次 launch
NCU_COMMON=(--n "${N}" --nwarps "${NCU_NWARPS}" --runs 1 --warmup 2 --device "${DEVICE}")
if [[ "${GRID}" != "0" ]]; then
  NCU_COMMON+=(--grid "${GRID}")
fi

echo
echo "======== NCU (nwarps=${NCU_NWARPS}): smem vs shfl ========"
echo "Ignore binary-printed median while ncu is attached."
# --kernel-name-base 只能是 function|demangled|mangled；过滤名用 --kernel-name
for m in smem shfl; do
  echo "---- mode=${m} ----"
  ncu --kernel-name-base function --kernel-name "regex:kernel_reduce_${m}" \
    --launch-skip 3 --launch-count 1 \
    --metrics "${NCU_METRICS}" \
    "${BIN}" --mode "${m}" "${NCU_COMMON[@]}" || \
    echo "WARNING: NCU failed for mode=${m}" >&2
  echo
done

echo "======== NCU (nwarps=${NCU_NWARPS}): shfl_int vs redux ========"
# --mode redux 会先跑 shfl_int 再跑 redux；各用 kernel-name 过滤
echo "---- kernel_reduce_shfl_int (via --mode redux) ----"
ncu --kernel-name-base function --kernel-name "regex:kernel_reduce_shfl_int" \
  --launch-skip 3 --launch-count 1 \
  --metrics "${NCU_METRICS}" \
  "${BIN}" --mode redux "${NCU_COMMON[@]}" || \
  echo "WARNING: NCU failed for shfl_int" >&2
echo

echo "---- kernel_reduce_redux_int (via --mode redux) ----"
# redux mode：verify(shfl)+verify(redux)+warmup… 路径更复杂；放宽 skip 并靠 regex 抓 redux
ncu --kernel-name-base function --kernel-name "regex:kernel_reduce_redux_int" \
  --launch-skip 3 --launch-count 1 \
  --metrics "${NCU_METRICS}" \
  "${BIN}" --mode redux "${NCU_COMMON[@]}" || \
  echo "WARNING: NCU failed for redux (need sm_80+)" >&2
echo

echo "NOTE: paste metric tables into chat / docs/results/C-01_warp_primitives.md §NCU."
echo "Main conclusion remains bare --mode sweep median, not ncu-attached ms."
