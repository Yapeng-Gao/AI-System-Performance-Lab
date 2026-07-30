#!/usr/bin/env bash
# B-08: batch-run TMA modes + intensity sweep (+ optional NCU / inst counts)
# Needs sm_90+ (Hopper / Blackwell). Main evidence remains bare --mode sweep.
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
  echo "ERROR: 02_memory_optim_08_tma_intro not found. Build first (re-run cmake; ARCH>=90)." >&2
  exit 1
fi

N="${N:-4194304}"
TILES="${TILES:-64}"
BLOCK="${BLOCK:-256}"
FMA_ITERS="${FMA_ITERS:-8}"
RUNS="${RUNS:-7}"
WARMUP="${WARMUP:-2}"
DEVICE="${DEVICE:-0}"
MODE_FILTER="${1:-all}"   # all | sync | bulk1d | tensor2d | pipe2 | sweep | ncu-only
DO_NCU="${DO_NCU:-0}"
NCU_FMA="${NCU_FMA:-1}"   # latency-bound point for stall / inst contrast

if [[ "${MODE_FILTER}" == "ncu-only" ]]; then
  DO_NCU=1
  modes=()
else
  modes=(sync bulk1d tensor2d pipe2 sweep)
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
    --tiles "${TILES}" \
    --block "${BLOCK}" \
    --fma-iters "${FMA_ITERS}" \
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

# 不要用 --set full：会污染程序自身的 event 计时；看 section / 显式 metrics 即可
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ARGS=(--n "${N}" --tiles "${TILES}" --block "${BLOCK}"
             --fma-iters "${NCU_FMA}" --runs 1 --warmup 2 --device "${DEVICE}")

echo
echo "======== NCU WarpState + Memory (fma=${NCU_FMA}) ========"
for m in sync bulk1d pipe2; do
  echo "---- mode=${m} ----"
  ncu --launch-skip 2 --launch-count 1 \
    --section WarpStateStats --section MemoryWorkloadAnalysis \
    -o "${OUT_DIR}/tma_${m}" --force-overwrite \
    "${BIN}" --mode "${m}" "${COMMON_ARGS[@]}"
  ncu --import "${OUT_DIR}/tma_${m}.ncu-rep" --page details 2>/dev/null | head -n 60 || true
  echo
done

echo "======== NCU instruction counts (sync vs bulk1d) ========"
# --kernel-name-base 只能是 function|demangled|mangled；过滤名用 --kernel-name
for m in sync bulk1d; do
  echo "---- mode=${m} ----"
  ncu --kernel-name-base function --kernel-name "regex:kernel_${m}" \
    --launch-skip 2 --launch-count 1 \
    --metrics smsp__inst_executed.sum,sm__warps_launched.sum \
    "${BIN}" --mode "${m}" "${COMMON_ARGS[@]}" || \
    echo "WARNING: inst metrics failed for mode=${m}; try: ncu --query-metrics | rg inst_executed" >&2
  echo
done

echo "Wrote ${OUT_DIR}/tma_sync.ncu-rep tma_bulk1d.ncu-rep tma_pipe2.ncu-rep"
echo "NOTE: ignore the binary's printed median while ncu is attached."
echo "Import: ncu --import ${OUT_DIR}/tma_pipe2.ncu-rep --page details"
