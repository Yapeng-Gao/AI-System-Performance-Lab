#!/usr/bin/env bash
set -euo pipefail

# 基准可执行文件名（与 benchmarks/CMakeLists.txt 输出一致，位于 build/bin）
BINARIES=(
  "bench_attention_memory_bound"
  "bench_cp_async_pipeline"
  "bench_hbm_pointer_chasing"
)

OUT_DIR="docs/results/ncu"
mkdir -p "${OUT_DIR}"

METRICS=(
dram__bytes_read.sum
dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__inst_executed.sum
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
gpu__time_duration.sum
)

METRIC_STR=$(IFS=, ; echo "${METRICS[*]}")

for bin in "${BINARIES[@]}"; do
    exe="./bin/${bin}"
    if [ ! -x "${exe}" ]; then
        echo "Skip ${bin}: executable not found at ${exe}" >&2
        continue
    fi
    echo "==== Profiling ${bin} ===="
    ncu --metrics "${METRIC_STR}" \
        --csv \
        --target-processes all \
        "${exe}" \
        > "${OUT_DIR}/${bin}.csv"
done

echo "NCU profiling done."
