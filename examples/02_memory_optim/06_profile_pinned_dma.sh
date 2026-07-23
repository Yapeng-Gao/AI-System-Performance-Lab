#!/usr/bin/env bash
# B-06: batch-run pinned/DMA modes + optional NSYS on overlap
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=""
for cand in \
  "${ROOT_DIR}/build/bin/02_memory_optim_06_pinned_dma" \
  "${ROOT_DIR}/cmake-build-debug/bin/02_memory_optim_06_pinned_dma" \
  "${ROOT_DIR}/cmake-build-release/bin/02_memory_optim_06_pinned_dma" \
  "${ROOT_DIR}/build/bin/Release/02_memory_optim_06_pinned_dma.exe" \
  "${ROOT_DIR}/cmake-build-debug/bin/02_memory_optim_06_pinned_dma.exe"
do
  if [[ -x "${cand}" ]]; then
    BIN="${cand}"
    break
  fi
done

if [[ -z "${BIN}" ]]; then
  echo "ERROR: 02_memory_optim_06_pinned_dma not found. Build first." >&2
  exit 1
fi

MB="${MB:-256}"
CHUNK_MB="${CHUNK_MB:-16}"
STREAMS="${STREAMS:-4}"
KERNEL_ITERS="${KERNEL_ITERS:-8}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
DEVICE="${DEVICE:-0}"
MODE_FILTER="${1:-all}"   # all | pageable | pinned | serial | overlap | bidir | mapped
DO_NSYS="${DO_NSYS:-0}"

run_one() {
  local mode="$1"
  echo "======== mode=${mode} ========"
  "${BIN}" \
    --mode "${mode}" \
    --mb "${MB}" \
    --chunk-mb "${CHUNK_MB}" \
    --streams "${STREAMS}" \
    --kernel-iters "${KERNEL_ITERS}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}"
  echo
}

modes=(pageable pinned serial overlap bidir mapped)
if [[ "${MODE_FILTER}" != "all" ]]; then
  modes=("${MODE_FILTER}")
fi

for m in "${modes[@]}"; do
  run_one "${m}"
done

if [[ "${DO_NSYS}" == "1" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "WARNING: nsys not found; skip profiling." >&2
    exit 0
  fi
  OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  nsys profile -o "${OUT_DIR}/pinned_overlap" --force-overwrite true \
    "${BIN}" --mode overlap --mb "${MB}" --chunk-mb "${CHUNK_MB}" \
    --streams "${STREAMS}" --kernel-iters "${KERNEL_ITERS}" \
    --runs 1 --warmup 0 --device "${DEVICE}"
  echo "Wrote ${OUT_DIR}/pinned_overlap.nsys-rep"
fi
