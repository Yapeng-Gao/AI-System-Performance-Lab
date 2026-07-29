#!/usr/bin/env bash
# B-09: batch-run AoS/SoA + transpose modes (+ optional NCU sectors/request)
# Main evidence remains bare --mode sweep / --mode modes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=""
for cand in \
  "${ROOT_DIR}/build/bin/02_memory_optim_09_layout_transform" \
  "${ROOT_DIR}/cmake-build-debug/bin/02_memory_optim_09_layout_transform" \
  "${ROOT_DIR}/cmake-build-release/bin/02_memory_optim_09_layout_transform" \
  "${ROOT_DIR}/build/bin/Release/02_memory_optim_09_layout_transform.exe" \
  "${ROOT_DIR}/cmake-build-debug/bin/02_memory_optim_09_layout_transform.exe"
do
  if [[ -x "${cand}" ]] || [[ -f "${cand}" ]]; then
    BIN="${cand}"
    break
  fi
done

if [[ -z "${BIN}" ]]; then
  echo "ERROR: 02_memory_optim_09_layout_transform not found. Build first (re-run cmake)." >&2
  exit 1
fi

N="${N:-4194304}"
DIM="${DIM:-4096}"
TOUCH="${TOUCH:-1}"
BLOCK="${BLOCK:-256}"
RUNS="${RUNS:-7}"
WARMUP="${WARMUP:-2}"
DEVICE="${DEVICE:-0}"
MODE_FILTER="${1:-all}"   # all | sweep | modes | aos | soa | copy | transpose_* | ncu-only
DO_NCU="${DO_NCU:-0}"

SECTOR_METRICS="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"

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
  "${BIN}" \
    --mode "${mode}" \
    --n "${N}" \
    --dim "${DIM}" \
    --touch-fields "${TOUCH}" \
    --block "${BLOCK}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}"
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

COMMON_ARGS=(--n "${N}" --dim "${DIM}" --block "${BLOCK}"
             --runs 1 --warmup 2 --device "${DEVICE}")

echo
echo "======== NCU sectors/request (AoS vs SoA) ========"
echo "Ideal float coalesced ≈ 4 sectors/request; AoS touch=1 often ≫ SoA."
for m in aos soa; do
  for tf in 1 8; do
    echo "---- mode=${m} touch=${tf} ----"
    ncu --kernel-name-base "kernel_${m}" --launch-skip 2 --launch-count 1 \
      --metrics "${SECTOR_METRICS}" \
      "${BIN}" --mode "${m}" --touch-fields "${tf}" "${COMMON_ARGS[@]}" || \
      echo "WARNING: sectors metrics failed for ${m} touch=${tf}" >&2
    echo
  done
done

echo "======== NCU sectors/request (transpose naive vs tiled) ========"
# verify launch + fill + warmup×2 + timed run → skip 3 keeps last timed launch
for m in transpose_naive transpose_tiled; do
  echo "---- mode=${m} ----"
  ncu --kernel-name-base "kernel_${m}" --launch-skip 3 --launch-count 1 \
    --metrics "${SECTOR_METRICS}" \
    "${BIN}" --mode "${m}" "${COMMON_ARGS[@]}" || \
    echo "WARNING: sectors metrics failed for ${m}" >&2
  echo
done

echo "NOTE: ignore the binary's printed median/GB/s while ncu is attached."
echo "Paste the metric tables into chat / docs/results/B-09_layout.md NCU section."
