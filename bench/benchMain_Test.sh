#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHES=(
    "bench_cublas"
    "bench_cusparseLt"
    "bench_n2m4"
)

# 默认测试参数。每一项格式必须为: "M K N SPLIT_K"
DEFAULT_CASES=(
    "4096 4096 128 4"
    "4096 4096 256 4"
)

BUILD_FIRST="${BUILD_FIRST:-1}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
BREAK_NUM="${BREAK_NUM:--1}"

usage() {
    cat <<'EOF'
Usage:
  ./benchMain_Test.sh
  ./benchMain_Test.sh "M K N SPLIT_K" "M K N SPLIT_K" ...

Examples:
  ./benchMain_Test.sh
  ./benchMain_Test.sh "4096 4096 128 4" "8192 4096 256 4"

Environment variables:
  BUILD_FIRST=1          Build benchmarks before running (default: 1)
  CUDA_VISIBLE_DEVICES=0 GPU device selection (default: 0)
  BREAK_NUM=1            Stop after the given case index; -1 disables early stop
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# > 0 )); then
    CASES=("$@")
else
    CASES=("${DEFAULT_CASES[@]}")
fi

if [[ "$BUILD_FIRST" == "1" ]]; then
    echo "Building benchmarks: ${BENCHES[*]}"
    make "${BENCHES[@]}"
fi

for bench in "${BENCHES[@]}"; do
    if [[ ! -x "./$bench" ]]; then
        echo "Error: executable ./$bench not found. Run 'make ${BENCHES[*]}' first."
        exit 1
    fi
done

for ((i=0; i<${#CASES[@]}; i++)); do
    case_string="${CASES[i]}"

    read -r m k n splitk extra <<< "$case_string"
    if [[ -n "${extra:-}" || -z "${m:-}" || -z "${k:-}" || -z "${n:-}" || -z "${splitk:-}" ]]; then
        echo "Error: invalid case '$case_string'. Expected format: 'M K N SPLIT_K'"
        exit 1
    fi

    echo "============================================================"
    echo "Case[$i]: M=$m, K=$k, N=$n, SPLIT_K=$splitk"
    echo "============================================================"

    for bench in "${BENCHES[@]}"; do
        echo "[Run] $bench M=$m K=$k N=$n SPLIT_K=$splitk"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "./$bench" "$m" "$k" "$n" "$splitk"
        echo
    done

    if [[ "$BREAK_NUM" -ge 0 && "$i" -eq "$BREAK_NUM" ]]; then
        echo "Reached BREAK_NUM=$BREAK_NUM, exiting loop."
        break
    fi
done
