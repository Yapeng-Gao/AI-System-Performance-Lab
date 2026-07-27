#!/usr/bin/env bash
# B-08: batch-run TMA modes + intensity sweep (+ optional NCU)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=""
for cand in \
  "${ROOT_DIR}/build/bin/02_memory_optim_08_tma_intro" \
  "${ROOT_DIR}/cmake-build-debug/bin/02_memory_optim_08_tma_intro" \
  "${ROOT_DIR}/cmake-build-release/bin/02_memory_optim_08_tma_intro" \
  "${ROOT_DIR}/build/bin/Release/02_memory_optim_08_tma_intro.exe" \
  "${ROOT_DIR}/cmake-build-debug/bin/02_memory_optim_08_tma_intro.exe"
do
  if [[ -x "${cand}" ]] || [[ -f "${cand}" ]]; then
    BIN="${cand}"
    break
  fi
done

if [[ -z "${BIN}" ]]; then
  echo "ERROR: 02_memory_optim_08_tma_intro not found. Build first (re-run cmake)." >&2
  exit 1
fi

N="${N:-4194304}"
TILES="${TILES:-64}"
BLOCK="${BLOCK:-256}"
FMA_ITERS="${FMA_ITERS:-8}"
RUNS="${RUNS:-7}"
WARMUP="${WARMUP:-2}"
DEVICE="${DEVICE:-0}"
MODE_FILTER="${1:-all}"   # all | sync | bulk1d | tensor2d | pipe2 | sweep
DO_NCU="${DO_NCU:-0}"

run_one() {
  local mode="$1"
  echo "======== mode=${mode} ========"
  "${BIN}" \
    --mode "${mode}" \
    --n "${N}" \
    --tiles "${TILES}" \
    --block "${BLOCK}" \
    --fma-iters "${FMA_ITERS}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}"
  echo
}

modes=(sync bulk1d tensor2d pipe2 sweep)
if [[ "${MODE_FILTER}" != "all" ]]; then
  modes=("${MODE_FILTER}")
fi

for m in "${modes[@]}"; do
  run_one "${m}"
done

if [[ "${DO_NCU}" == "1" ]]; then
  if ! command -v ncu >/dev/null 2>&1; then
    echo "WARNING: ncu not found; skip profiling." >&2
    exit 0
  fi
  OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ncu --launch-skip 2 --launch-count 1 \
    --section WarpStateStats --section MemoryWorkloadAnalysis \
    -o "${OUT_DIR}/tma_sync" --force-overwrite \
    "${BIN}" --mode sync --n "${N}" --tiles "${TILES}" --block "${BLOCK}" \
    --fma-iters 4 --runs 1 --warmup 2 --device "${DEVICE}"
  ncu --launch-skip 2 --launch-count 1 \
    --section WarpStateStats --section MemoryWorkloadAnalysis \
    -o "${OUT_DIR}/tma_bulk1d" --force-overwrite \
    "${BIN}" --mode bulk1d --n "${N}" --tiles "${TILES}" --block "${BLOCK}" \
    --fma-iters 4 --runs 1 --warmup 2 --device "${DEVICE}"
  echo "Wrote ${OUT_DIR}/tma_sync.ncu-rep and tma_bulk1d.ncu-rep"
  echo "NOTE: ignore the binary's printed median/GB/s while ncu is attached"
fi
