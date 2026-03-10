#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
    echo "Usage:"
    echo "  ./kernelTest.sh [kernel|bench]"
    echo "  ./kernelTest.sh [kernel|bench] M K N SPLIT_K"
    echo "Environment overrides:"
    echo "  PROGRAM_MODE=kernel|bench"
    echo "  M_CASES=4096,8192"
    echo "  K_CASES=4096,4096"
    echo "  N_CASES=128,64"
    echo "  SPLIT_K_CASES=4,1"
    echo "  BREAK_NUM=0"
}

PROGRAM_MODE="${PROGRAM_MODE:-kernel}"
if [[ $# -eq 1 && ( "$1" == "kernel" || "$1" == "bench" ) ]]; then
    PROGRAM_MODE="$1"
    shift
elif [[ $# -eq 5 && ( "$1" == "kernel" || "$1" == "bench" ) ]]; then
    PROGRAM_MODE="$1"
    shift
fi

declare -a M
declare -a K
declare -a N
declare -a SPLIT_K

if [[ $# -eq 4 ]]; then
    M=("$1")
    K=("$2")
    N=("$3")
    SPLIT_K=("$4")
elif [[ $# -eq 0 ]]; then
    IFS=',' read -r -a M <<< "${M_CASES:-4096}"
    IFS=',' read -r -a K <<< "${K_CASES:-4096}"
    IFS=',' read -r -a N <<< "${N_CASES:-128}"
    IFS=',' read -r -a SPLIT_K <<< "${SPLIT_K_CASES:-4}"
else
    usage
    exit 1
fi

BREAK_NUM="${BREAK_NUM:--1}"
CUDA_VISIBLE_DEVICE="${CUDA_VISIBLE_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
SPMM_PRINT_CONFIG="${SPMM_PRINT_CONFIG:-1}"
SPMM_PREVIEW_COUNT="${SPMM_PREVIEW_COUNT:-16}"

case "$PROGRAM_MODE" in
    kernel)
        PROGRAM_PATH="${PROGRAM_PATH:-$SCRIPT_DIR/kernelTest}"
        ;;
    bench)
        PROGRAM_PATH="${PROGRAM_PATH:-$PROJECT_ROOT/bench/benchMain}"
        ;;
    *)
        echo "Unsupported PROGRAM_MODE: $PROGRAM_MODE"
        exit 1
        ;;
esac

if [[ ${#M[@]} -ne ${#K[@]} || ${#M[@]} -ne ${#SPLIT_K[@]} ]]; then
    echo "Error: M_CASES, K_CASES, and SPLIT_K_CASES must have the same length."
    exit 1
fi

echo "[TrialRun] role=correctness program_mode=$PROGRAM_MODE program_path=$PROGRAM_PATH device=$CUDA_VISIBLE_DEVICE cases=${#M[@]} n_values=${#N[@]}"

for ((i=0; i<${#M[@]}; i++)); do
    m=${M[i]}
    k=${K[i]}
    splitk=${SPLIT_K[i]}
    for n in "${N[@]}"; do
        echo "[TrialRun] launching M=$m K=$k N=$n SPLIT_K=$splitk"
        env \
            CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICE" \
            SPMM_RUN_ROLE=correctness \
            SPMM_PRINT_CONFIG="$SPMM_PRINT_CONFIG" \
            SPMM_PREVIEW_COUNT="$SPMM_PREVIEW_COUNT" \
            "$PROGRAM_PATH" "$m" "$k" "$n" "$splitk"
    done

    if [[ "$BREAK_NUM" -ge 0 && "$i" -eq "$BREAK_NUM" ]]; then
        echo "Reached BREAK_NUM=$BREAK_NUM, exiting loop."
        break
    fi
done
