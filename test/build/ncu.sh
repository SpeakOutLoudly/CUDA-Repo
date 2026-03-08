#!/bin/bash
# ===============================================
# Nsight Compute profiling script for kernelTest
# ===============================================

# 可配置项
NCU_PATH="/usr/local/cuda/bin/ncu"
# PROG_PATH="/data1/lys/workspaces/code/cudaCode_v0/test/build/kernelTest"
PROG_PATH="/data1/lys/workspaces/code/cudaCode_v0/bench/benchMain"
REPORT_DIR="/data1/lys/workspaces/code/cudaCode_v0/test/nsys"
REPORT_NAME="kernelTest_$(date +%Y%m%d_%H%M%S).ncu-rep"

# 创建报告目录（如果不存在）
# mkdir -p "$REPORT_DIR"

# 可选：启用 sudo 以访问 Performance Counters
SUDO_CMD="sudo"

# Nsight Compute CLI 命令
$SUDO_CMD CUDA_VISIBLE_DEVICES=0 $NCU_PATH \
    --target-processes all \
    --section LaunchStats \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section InstructionStats \
    --section ComputeWorkloadAnalysis \
    --section WarpStateStats \
    --section SchedulerStats \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section Occupancy \
    --section SourceCounters \
    -o "$REPORT_DIR/$REPORT_NAME" \
    "$PROG_PATH" 4096 4096 128 1

# 提示报告保存位置
echo "Nsight Compute report saved to: $REPORT_DIR/$REPORT_NAME"

# 没有对应的 Section
#    --section WorkloadDistribution \