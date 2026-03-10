#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NCU_PATH="${NCU_PATH:-/usr/local/cuda/bin/ncu}"
PROG_PATH="${PROG_PATH:-$SCRIPT_DIR/bin/benchMain}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/ncu}"
REPORT_BASENAME="${REPORT_BASENAME:-benchMain}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
USE_SUDO="${USE_SUDO:-1}"

DEFAULT_CASE="4096 4096 16 1"

usage() {
    cat <<'EOF'
Usage:
  ./benchMain_Test.sh
  ./benchMain_Test.sh M K N SPLIT_K

Examples:
  ./benchMain_Test.sh
  ./benchMain_Test.sh 4096 4096 16 1

Environment variables:
  CUDA_VISIBLE_DEVICES=0   GPU device selection (default: 0)
  NCU_PATH=...             Path to Nsight Compute CLI
  PROG_PATH=...            Executable to profile (default: ./bin/benchMain)
  REPORT_DIR=...           Report output directory (default: ./ncu)
  REPORT_BASENAME=...      Report filename prefix (default: benchMain)
  USE_SUDO=1               Run ncu via sudo when set to 1 (default: 1)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# == 0 )); then
    read -r M K N SPLIT_K <<< "$DEFAULT_CASE"
elif (( $# == 4 )); then
    M="$1"
    K="$2"
    N="$3"
    SPLIT_K="$4"
else
    echo "Error: expected either no arguments or exactly 4 arguments: M K N SPLIT_K" >&2
    usage
    exit 1
fi

mkdir -p "$REPORT_DIR"

echo "[1/4] make clean"
make clean

echo "[2/4] make"
make

if [[ ! -x "$PROG_PATH" ]]; then
    echo "Error: executable not found: $PROG_PATH" >&2
    exit 1
fi

if [[ ! -x "$NCU_PATH" ]]; then
    echo "Error: ncu not found: $NCU_PATH" >&2
    exit 1
fi

echo "[3/4] run benchmark"
"$PROG_PATH" "$M" "$K" "$N" "$SPLIT_K"

REPORT_NAME="${REPORT_BASENAME}_$(date +%Y%m%d_%H%M%S).ncu-rep"

NCU_CMD=(
    "$NCU_PATH"
    --target-processes all
    --section LaunchStats
    --section MemoryWorkloadAnalysis
    --section MemoryWorkloadAnalysis_Tables
    --section InstructionStats
    --section ComputeWorkloadAnalysis
    --section WarpStateStats
    --section SchedulerStats
    --section SpeedOfLight
    --section SpeedOfLight_RooflineChart
    --section Occupancy
    --section SourceCounters
    -o "$REPORT_DIR/$REPORT_NAME"
    "$PROG_PATH" "$M" "$K" "$N" "$SPLIT_K"
)

echo "[4/4] ncu profile"
echo "Command target: $PROG_PATH $M $K $N $SPLIT_K"
echo "Report path: $REPORT_DIR/$REPORT_NAME"

if [[ "$USE_SUDO" == "1" ]]; then
    sudo env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${NCU_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${NCU_CMD[@]}"
fi

echo "Nsight Compute report saved to: $REPORT_DIR/$REPORT_NAME"
