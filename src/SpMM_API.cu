#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "TileConfig.h"        // 使用 Tile 配置与派生常量
#include "reduce_Kernel.cuh"
#include "SpMM_Kernel.cuh"


// cuda 的启动函数
template<typename N2M4TilingConfig, int stages>
static void SpMM_N2M4_Kernel_API(   cudaStream_t stream,
                                    const half*  Compressed_A,
                                    const half*  B,
                                    const uint16_t*  metadata,
                                    half*        C,
                                    const int    M_Global,
                                    const int    N_Global,
                                    const int    K_Global,
                                    const int    Split_K)
{
    // 进行 shared memory 的申请，其中包括两部分：1.计算数据的存储空间 2.计算结果的存储空间 考虑分时复用只需要取大就行
    int computeSize = (N2M4TilingConfig::TILE_N * N2M4TilingConfig::TILE_K) * sizeof(half) * stages       // 稠密激活矩阵，使用双缓冲区的设计
                    + (N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2) * sizeof(half) * stages   // 稀疏权重矩阵，使用双缓冲区设计
                    + ((N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)) / 4 * stages;               // metadata，每个数据需要2bit,即1/4字节，使用双缓冲区设计

    // PADDING_SHARED_MEM_FOR_C 在每个行的长度上额外增加了1个 float 元素的空间，用于消除 bank 冲突
    int resultSize = (N2M4TilingConfig::TILE_M * (N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C)) * sizeof(half);  // 计算结果的存储空间，注意这里是 half 类型
    int SHMEM_SZ = max(computeSize, resultSize);

    // 进行 cuda 设置，表示 SpMM_Kernel_bitmap_v3 内核需要使用 SHMEM_SZ 大小的共享内存
    // cudaFuncAttributeMaxDynamicSharedMemorySize 表示进行设置的属性：函数属性-最大动态共享内存大小
    cudaFuncSetAttribute(SpMM_N2M4_Kernel<N2M4TilingConfig, stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    
    // 分块确定grid维度
    int dimN = max(N_Global / N2M4TilingConfig::TILE_N, 1);  // max(N_Global/N2M4TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int dimM = M_Global / N2M4TilingConfig::TILE_M * Split_K;  // 使用 Split_K 策略增加线程块的数量，提升占用率

    // 设置 Grid 内线程块的排布 
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy、
    // 设置线程块内线程的排布
    dim3 BlockDim(WARP_SIZE * N2M4TilingConfig::BLOCK_WARPS, 1, 1);  

    // 异步启动 GPU 上的主计算内核
    SpMM_N2M4_Kernel<N2M4TilingConfig, stages><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(Compressed_A, metadata, B, C, M_Global, N_Global, K_Global, Split_K);
}



// 用户从CPU代码中调用的主要函数，用于执行一次完整的稀疏矩阵乘法 `C = A * B`
// 参数：
// - A: M×K 激活（row-major）
// - B_values: (K/2)×N 的压缩权重值，按照16*8进行打包，16*8块内是行主序，块之间也是行主序
// - B_meta: 稀疏元数据（与 K 步长对齐，调用端负责布局）
// - C: M×N 输出（row-major，FP32）
// - M, N, K: 维度（需满足 K 为 32 的倍数，N 多数情况下为 8 的倍数）
cudaError_t SpMM_N2M4_Launch(   cudaStream_t stream,
                                const half*  Compressed_A,      // M × (K/2), 行主序（2:4压缩值）
                                const half*  B,                 // K x N, 行主序
                                const uint16_t*  metadata,      // metadata（与 K 方向对齐）
                                half* C,                        // M×N，row-major
                                const int    M_Global,
                                const int    N_Global,
                                const int    K_Global,
                                half*        Split_Reduction,  // 用于 SplitK 规约的中间结果存储空间
                                const int    Split_K)
{
    // KernelOutputPtr 是内核的输出指针，用于指向结果地址
    // 如果使用了 Split_K 策略，则结果需要先存储在 Split_Reduction 指向的临时空间，最后再进行规约存储到 C 中
    half* KernelOutputPtr;
    if (Split_K == 1)
        KernelOutputPtr = C;  // 指向用户传入的、用于存储最终结果的矩阵 C 的内存地址
    else
        KernelOutputPtr = Split_Reduction;  // 指向一块临时的、独立的中间存储空间，用于存储 Split_k 的规约结果

    // ----------------------------debug 测试------------------------------- //
    // if(M_Global == 16){
    //     SpMM_N2M4_Kernel_API<N2M4TilingConfig<32, 1, 1, 1, 1, 2>>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
    //     cudaDeviceSynchronize(); 
    //     cudaError_t Error = cudaGetLastError();
    //     if (Error != cudaSuccess)
    //         return Error;

    //     // 如果没有使用 split_k 则返回结果
    //     if (Split_K == 1)
    //         return Error;

    //     // 重新分配 Grid 和 Block，启动 Split_k 规约计算的内核
    //     dim3 GridDim(max((M_Global * N_Global) / 256, 1), 1, 1);
    //     dim3 BlockDim(WARP_SIZE, 1, 1);
    //     SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, KernelOutputPtr, M_Global, N_Global, Split_K);
    //     return cudaGetLastError();
    // }else if(M_Global == 32){
    //     SpMM_N2M4_Kernel_API<N2M4TilingConfig<32, 2, 1, 1, 1, 2>>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
    //     cudaDeviceSynchronize(); 
    //     cudaError_t Error = cudaGetLastError();
    //     if (Error != cudaSuccess)
    //         return Error;

    //     // 如果没有使用 split_k 则返回结果
    //     if (Split_K == 1)
    //         return Error;

    //     // 重新分配 Grid 和 Block，启动 Split_k 规约计算的内核
    //     dim3 GridDim(max((M_Global * N_Global) / 256, 1), 1, 1);
    //     dim3 BlockDim(WARP_SIZE, 1, 1);
    //     SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, KernelOutputPtr, M_Global, N_Global, Split_K);
    //     return cudaGetLastError();
    // }
    // ----------------------------debug 测试------------------------------- //




    // Batched SpMM
    // 启动 SpMM 矩阵乘计算 Kernel
    // WARP_COL_TENSORS 是唯一在不同 case 中变化的关键参数。它决定了一个 Warp 沿 N 方向（列方向）执行多少次 MMA 矩阵乘法，从而直接影响了 TILE_N 的大小
    // int _K,                    // 一般使用 32
    // int _BLOCK_ROW_WARPS,
    // int _BLOCK_COL_WARPS,
    // int _WARP_COL_TENSORS,
    // int _WARP_ROW_TENSORS = 1,  // warp 在 M 方向并行 tensor 数，默认为1
    // int _BLOCK_K_STEPS = 2      // 可选：默认 2 次 mma
    switch (N_Global) {
        case 8:
            // SpMM_N2M4_Kernel_API<N2M4TilingConfig<32, 4, 1, 1, 1, 2>>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 1, 1, 1, 4>, 2>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            // SpMM_N2M4_Kernel_API<N2M4TilingConfig<32, 4, 1, 2, 1, 2>>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 1, 2, 1, 4>, 2>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            // SpMM_N2M4_Kernel_API<N2M4TilingConfig<32, 4, 1, 4, 1, 2>>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 1, 4, 1, 4>, 2>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            // SpMM_N2M4_Kernel_API<N2M4TilingConfig<32, 2, 2, 4, 2, 2>>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 2, 2, 4, 2, 4>, 2>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            // warps = 4*2
            // TILE_M = 4*2*16 = 128 
            // TILE_N = 2*8*8 = 128
            // SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 2, 8, 2, 4>>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 2, 2, 4, 4, 4>, 2>(stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            break;
    }
    
    // 在内核启动后检查是否发生了错误
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    // 如果没有使用 split_k 则返回结果
    if (Split_K == 1)
        return Error;

    // 重新分配 Grid 和 Block，启动 Split_k 规约计算的内核
    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, KernelOutputPtr, M_Global, N_Global, Split_K);
    return cudaGetLastError();

}


