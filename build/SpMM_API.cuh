#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>



// 用户从CPU代码中调用的主要函数，用于执行一次完整的稀疏矩阵乘法 `C = A * B`
// 参数：
// - A: M×K 激活（row-major）
// - B_values: (K/2)×N 的压缩权重值（row-major）
// - B_meta: 稀疏元数据（与 K 步长对齐，调用端负责布局）
// - C: M×N 输出（row-major，FP32）
// - M, N, K: 维度（需满足 K 为 32 的倍数，N 多数情况下为 8 的倍数）
cudaError_t SpMM_N2M4_Launch(   cudaStream_t stream,
                                const half*  Compressed_A,      // M × (K/2), 行主序（2:4压缩值）
                                const half*  B,                 // K x N, 列主序（等效于 N x K 行主序）
                                const uint16_t*  metadata,      // metadata（与 K 方向对齐）
                                half* C,                        // M×N，row-major
                                const int    M_Global,
                                const int    N_Global,
                                const int    K_Global,
                                half*        Split_Reduction,
                                const int    Split_K);









