#!/bin/bash

# 矩阵维度
# M=(8192 4096 32000 32000 28672 5120 5120 3584 4096 13824 8192 18944 14336 4096 8192 11008 32000 20480 3584 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
# K=(29568 4096 5120 8192 8192 5120 13824 20480 11008 5120 8192 3584 4096 14336 28672 4096 4096 3584 18944 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)

# Split_K
# SPLIT_K=(7 7 3 3 4 5 5 7 7 3 7 7 7 7 7 7 3 6 7 1 3 4 3 7 5 2 5 2 6 3 6)

# 输出矩阵列数
# N=(8 16 32)



# 正确性 debug 测试
M=(4096)
K=(4096)
N=(128)
SPLIT_K=(4)

# 可选参数：提前退出的索引
breakNum=-1  # 默认 -1 表示不提前退出

# 检查 M 和 K 数组长度是否一致
if [ ${#M[@]} -ne ${#K[@]} ]; then
    echo "Error: M and K arrays must have the same length."
    exit 1
fi

# 循环运行测试
for ((i=0; i<${#M[@]}; i++)); do
    m=${M[i]}
    k=${K[i]}
    splitk=${SPLIT_K[i]}
    for n in "${N[@]}"; do
        echo "Running spinfer and cublas test case: M=$m, K=$k, N=$n, SPLIT_K=$splitk"
        CUDA_VISIBLE_DEVICES=0 ./kernelTest $m $k $n $splitk
    done

    # 提前退出判断
    if [ "$breakNum" -ge 0 ] && [ "$i" -eq "$breakNum" ]; then
        echo "Reached breakNum=$breakNum, exiting loop."
        break
    fi
done
