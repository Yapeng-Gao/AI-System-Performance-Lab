#!/usr/bin/env bash
# B-07: dump SASS for 07_cp_async_pipeline.cu (verify LDGSTS / CP.ASYNC)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT_DIR}/examples/02_memory_optim/07_cp_async_pipeline.cu"
OUT_DIR="${ROOT_DIR}/docs/sass"
mkdir -p "${OUT_DIR}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found" >&2
  exit 1
fi
if ! command -v nvdisasm >/dev/null 2>&1 && ! command -v cuobjdump >/dev/null 2>&1; then
  echo "ERROR: need nvdisasm or cuobjdump" >&2
  exit 1
fi

# sm_80 = Ampere (cp.async 引入)；sm_120 = Blackwell (RTX 5090)
ARCHS=("80" "120")
NAMES=("ampere" "blackwell")

for i in "${!ARCHS[@]}"; do
  arch="${ARCHS[$i]}"
  name="${NAMES[$i]}"
  mkdir -p "${OUT_DIR}/${name}"
  cubin="${OUT_DIR}/${name}/07_cp_async_pipeline_sm${arch}.cubin"
  sass="${OUT_DIR}/${name}/07_cp_async_pipeline.sass"

  echo "==== sm_${arch} (${name}) ===="
  nvcc -std=c++17 -O3 -lineinfo -cubin -arch=sm_"${arch}" \
    "${SRC}" -o "${cubin}"

  if command -v nvdisasm >/dev/null 2>&1; then
    nvdisasm "${cubin}" > "${sass}"
  else
    cuobjdump -sass "${cubin}" > "${sass}"
  fi

  echo "Wrote ${sass}"
  echo "---- LDGSTS / CP.ASYNC / LDG hits (first 40) ----"
  grep -nE 'LDGSTS|CP\.ASYNC|LDG\.E|STS' "${sass}" | head -n 40 || \
    echo "(no LDGSTS/CP.ASYNC string; search LDG/STS manually)"
  echo
done

echo "Done. Prefer pipe2/async kernels showing LDGSTS or CP.ASYNC vs sync path LDG+STS."
