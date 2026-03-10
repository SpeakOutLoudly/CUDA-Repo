#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
    echo "Usage:"
    echo "  ./ncu.sh [kernel|bench]"
    echo "  ./ncu.sh [kernel|bench] M K N SPLIT_K"
    echo "Environment overrides:"
    echo "  PROGRAM_MODE=kernel|bench"
    echo "  NCU_PATH=/usr/local/cuda/bin/ncu"
    echo "  REPORT_DIR=/data1/lys/workspaces/code/cudaCode_v0/test/nsys"
    echo "  NCU_KERNEL_FILTER='regex:SpMM_N2M4_Kernel|SplitK_Reduction'"
    echo "  NCU_NO_SUDO=1"
}

PROGRAM_MODE="${PROGRAM_MODE:-bench}"
if [[ $# -ge 1 && ( "$1" == "kernel" || "$1" == "bench" ) ]]; then
    PROGRAM_MODE="$1"
    shift
fi

M="${M:-4096}"
K="${K:-4096}"
N="${N:-128}"
SPLIT_K="${SPLIT_K:-1}"

if [[ $# -eq 4 ]]; then
    M="$1"
    K="$2"
    N="$3"
    SPLIT_K="$4"
elif [[ $# -ne 0 ]]; then
    usage
    exit 1
fi

NCU_PATH="${NCU_PATH:-/usr/local/cuda/bin/ncu}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_ROOT/test/nsys}"
REPORT_PREFIX="${REPORT_PREFIX:-kernelProfile}"
CUDA_VISIBLE_DEVICE="${CUDA_VISIBLE_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
SPMM_PRINT_CONFIG="${SPMM_PRINT_CONFIG:-1}"
SPMM_PREVIEW_COUNT="${SPMM_PREVIEW_COUNT:-0}"
if [[ "${NCU_NO_SUDO:-0}" == "1" ]]; then
    SUDO_CMD=""
else
    SUDO_CMD="${SUDO_CMD:-sudo}"
fi

case "$PROGRAM_MODE" in
    kernel)
        PROG_PATH="${PROG_PATH:-$SCRIPT_DIR/kernelTest}"
        DEFAULT_KERNEL_FILTER='regex:SpMM_N2M4_Kernel|SplitK_Reduction'
        ;;
    bench)
        PROG_PATH="${PROG_PATH:-$PROJECT_ROOT/bench/benchMain}"
        DEFAULT_KERNEL_FILTER=""
        ;;
    *)
        echo "Unsupported PROGRAM_MODE: $PROGRAM_MODE"
        exit 1
        ;;
esac

NCU_KERNEL_FILTER="${NCU_KERNEL_FILTER:-$DEFAULT_KERNEL_FILTER}"
REPORT_STEM="${REPORT_PREFIX}_${PROGRAM_MODE}_M${M}_K${K}_N${N}_SK${SPLIT_K}_$(date +%Y%m%d_%H%M%S)"
REPORT_BASE="$REPORT_DIR/$REPORT_STEM"
REPORT_PATH="$REPORT_BASE.ncu-rep"
LOG_PATH="$REPORT_BASE.log"

mkdir -p "$REPORT_DIR"

COMMAND=()
if [[ -n "$SUDO_CMD" ]]; then
    COMMAND+=("$SUDO_CMD" "env")
else
    COMMAND+=("env")
fi
COMMAND+=(
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE"
    "SPMM_RUN_ROLE=profile"
    "SPMM_PRINT_CONFIG=$SPMM_PRINT_CONFIG"
    "SPMM_PREVIEW_COUNT=$SPMM_PREVIEW_COUNT"
    "$NCU_PATH"
    "--target-processes" "all"
    "--section" "LaunchStats"
    "--section" "MemoryWorkloadAnalysis"
    "--section" "MemoryWorkloadAnalysis_Tables"
    "--section" "InstructionStats"
    "--section" "ComputeWorkloadAnalysis"
    "--section" "WarpStateStats"
    "--section" "SchedulerStats"
    "--section" "SpeedOfLight"
    "--section" "SpeedOfLight_RooflineChart"
    "--section" "Occupancy"
    "--section" "SourceCounters"
)

if [[ -n "$NCU_KERNEL_FILTER" ]]; then
    COMMAND+=("-k" "$NCU_KERNEL_FILTER")
fi

COMMAND+=(
    "-o" "$REPORT_BASE"
    "$PROG_PATH" "$M" "$K" "$N" "$SPLIT_K"
)

echo "[TrialProfile] mode=$PROGRAM_MODE program_path=$PROG_PATH report_path=$REPORT_PATH log_path=$LOG_PATH M=$M K=$K N=$N SPLIT_K=$SPLIT_K kernel_filter=${NCU_KERNEL_FILTER:-none}"

set +e
"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
COMMAND_STATUS=${PIPESTATUS[0]}
set -e

echo "Nsight Compute report saved to: $REPORT_PATH"
echo "Nsight Compute log saved to: $LOG_PATH"

exit "$COMMAND_STATUS"