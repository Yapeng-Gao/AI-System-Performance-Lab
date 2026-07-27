#!/usr/bin/env bash
# B-08: dump SASS for 08_tma_intro.cu (verify TMA / UTMALDG / CP.ASYNC.BULK)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT_DIR}/examples/02_memory_optim/08_tma_intro.cu"
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

# sm_90 = Hopper TMA；sm_120 = Blackwell (RTX 5090)
ARCHS=("90" "120")
NAMES=("hopper" "blackwell")

for i in "${!ARCHS[@]}"; do
  arch="${ARCHS[$i]}"
  name="${NAMES[$i]}"
  mkdir -p "${OUT_DIR}/${name}"
  cubin="${OUT_DIR}/${name}/08_tma_intro_sm${arch}.cubin"
  sass="${OUT_DIR}/${name}/08_tma_intro.sass"

  echo "==== sm_${arch} (${name}) ===="
  if ! nvcc -std=c++17 -O3 -lineinfo -cubin -arch=sm_"${arch}" \
      "${SRC}" -o "${cubin}" -lcuda 2>/tmp/b08_nvcc_err.txt; then
    echo "WARNING: failed to compile sm_${arch}; skip"
    cat /tmp/b08_nvcc_err.txt || true
    continue
  fi

  if command -v nvdisasm >/dev/null 2>&1; then
    nvdisasm "${cubin}" > "${sass}"
  else
    cuobjdump -sass "${cubin}" > "${sass}"
  fi

  echo "Wrote ${sass}"
  echo "---- TMA / BULK / LDG hits (first 40) ----"
  grep -nE 'UTMA|TMA|CP\.ASYNC\.BULK|BULK|LDGSTS|LDG\.E' "${sass}" | head -n 40 || \
    echo "(no obvious TMA string; search manually)"
  echo
done

echo "Done. Prefer bulk1d/tensor2d/pipe2 showing TMA/bulk ops vs sync LDG path."
