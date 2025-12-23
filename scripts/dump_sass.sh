#!/usr/bin/env bash
set -euo pipefail

# 针对 benchmarks 目录下的单文件基准，直接用 nvcc 编出 cubin 后 nvdisasm
ARCHS=("80" "90" "100")
ARCH_NAMES=("ampere" "hopper" "blackwell")

SOURCES=(
  "benchmarks/attention_memory_bound.cu"
  "benchmarks/cp_async_pipeline.cu"
  "benchmarks/hbm_pointer_chasing.cu"
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
