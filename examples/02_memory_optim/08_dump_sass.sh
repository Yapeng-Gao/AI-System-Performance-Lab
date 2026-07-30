#!/usr/bin/env bash
# B-08: dump SASS for 08_tma_intro.cu (verify TMA / elect / mbarrier vs sync LDG+STS)
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

# sm_90 = Hopper；sm_120 = Blackwell (RTX 5090)
ARCHS=("90" "120")
NAMES=("hopper" "blackwell")

for i in "${!ARCHS[@]}"; do
  arch="${ARCHS[$i]}"
  name="${NAMES[$i]}"
  mkdir -p "${OUT_DIR}/${name}"
  cubin="${OUT_DIR}/${name}/08_tma_intro_sm${arch}.cubin"
  sass="${OUT_DIR}/${name}/08_tma_intro.sass"

  echo "==== sm_${arch} (${name}) ===="
  nvcc -std=c++17 -O3 -lineinfo -cubin -arch=sm_"${arch}" \
    "${SRC}" -o "${cubin}"

  if command -v nvdisasm >/dev/null 2>&1; then
    nvdisasm "${cubin}" > "${sass}"
  else
    cuobjdump -sass "${cubin}" > "${sass}"
  fi

  echo "Wrote ${sass}"
  echo "---- counts ----"
  printf "  UTMALDG*: "; grep -cE 'UTMALDG' "${sass}" || true
  printf "  ELECT:    "; grep -cE 'ELECT' "${sass}" || true
  printf "  MBARRIER attrs: "; grep -cE 'EIATTR_.*MBARRIER' "${sass}" || true

  echo "---- UTMALDG / ELECT / MBARRIER hits (first 30) ----"
  grep -nE 'UTMALDG|^\s+/\*.*\*/\s+ELECT |EIATTR_.*MBARRIER' "${sass}" \
    | head -n 30 || echo "(none)"

  # 只从 .text 段起截，避免 .dword 引用污染
  echo "---- .text.kernel_tensor2d: TMA-ish ----"
  awk '/\.text\._Z14kernel_tensor2d/,/\.nv\.(shared|constant0)\._Z14kernel_tensor2d/' "${sass}" \
    | grep -nE 'UTMALDG|ELECT|LDG\.E|STS' | head -n 20 || true

  echo "---- .text.kernel_bulk1d: TMA-ish / LDG+STS ----"
  awk '/\.text\._Z13kernel_bulk1d/,/\.nv\.(shared|constant0)\._Z13kernel_bulk1d/' "${sass}" \
    | grep -nE 'UTMALDG|ELECT|LDG\.E|STS|ARRIVES|MBAR' | head -n 25 || true

  echo "---- .text.kernel_sync: expect LDG+STS, little/no UTMALDG ----"
  awk '/\.text\._Z11kernel_sync/,/\.nv\.(shared|constant0)\._Z11kernel_sync/' "${sass}" \
    | grep -nE 'UTMALDG|ELECT|LDG\.E|STS' | head -n 15 || true
  echo
done

echo "Done. Expect tensor2d: UTMALDG.2D; bulk/pipe: ELECT + mbarrier; sync: LDG+STS."
