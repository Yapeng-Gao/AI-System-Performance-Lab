#!/usr/bin/env bash
set -euo pipefail

# 为 examples 下若干关键 .cu 直接 nvcc→cubin→nvdisasm
# 章节专用 dump 优先用 examples/02_memory_optim/07_dump_sass.sh 等

ARCHS=("80" "90" "100")
ARCH_NAMES=("ampere" "hopper" "blackwell")

SOURCES=(
  "examples/02_memory_optim/01_global_mem_bandwidth.cu"
  "examples/02_memory_optim/07_cp_async_pipeline.cu"
)

OUT_DIR="docs/sass"
mkdir -p "${OUT_DIR}"

for i in "${!ARCHS[@]}"; do
    arch=${ARCHS[$i]}
    name=${ARCH_NAMES[$i]}
    mkdir -p "${OUT_DIR}/${name}"

    echo "==== Dumping SASS for ${name} (sm_${arch}) ===="

    for src in "${SOURCES[@]}"; do
        bin=$(basename "${src}" .cu)
        echo "  -> ${bin}"

        tmp_cubin="${bin}_sm${arch}.cubin"
        nvcc -arch=sm_${arch} -lineinfo -cubin \
             "${src}" \
             -o "${tmp_cubin}"

        nvdisasm "${tmp_cubin}" > "${OUT_DIR}/${name}/${bin}.sass"
        rm -f "${tmp_cubin}"
    done
done

echo "All SASS dumped into ${OUT_DIR}/"
