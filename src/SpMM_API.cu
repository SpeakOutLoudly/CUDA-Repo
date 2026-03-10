#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "TileConfig.h"        // 使用 Tile 配置与派生常量
#include "reduce_Kernel.cuh"
#include "SpMM_Kernel.cuh"

template<typename N2M4TilingConfig, int stages>
static cudaError_t SpMM_N2M4_Kernel_API(cudaStream_t stream,
                                        const half* Compressed_A,
                                        const half* B,
                                        const uint16_t* metadata,
                                        half* C,
                                        int M_Global,
                                        int N_Global,
                                        int K_Global,
                                        int Split_K);


// cuda 的启动函数
template<typename N2M4TilingConfig, int stages>
static cudaError_t SpMM_N2M4_Kernel_API(   cudaStream_t stream,
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
    if (stages > 1)
        computeSize -= ((N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)) / 4;

    int SHMEM_SZ = max(computeSize, resultSize);

    int device = 0;
    cudaError_t Error = cudaGetDevice(&device);
    if (Error != cudaSuccess)
        return Error;

    int maxOptinSharedMem = 0;
    Error = cudaDeviceGetAttribute(&maxOptinSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (Error != cudaSuccess)
        return Error;

    if (SHMEM_SZ > maxOptinSharedMem) {
        printf("[SpMM] requested dynamic shared memory %d exceeds opt-in limit %d (stages=%d, N=%d).\n",
               SHMEM_SZ, maxOptinSharedMem, stages, N_Global);
        return cudaErrorInvalidConfiguration;
    }

    // 进行 cuda 设置，表示 SpMM_Kernel_bitmap_v3 内核需要使用 SHMEM_SZ 大小的共享内存
    // cudaFuncAttributeMaxDynamicSharedMemorySize 表示进行设置的属性：函数属性-最大动态共享内存大小
    Error = cudaFuncSetAttribute(SpMM_N2M4_Kernel<N2M4TilingConfig, stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    if (Error != cudaSuccess)
        return Error;
    
    // 分块确定grid维度
    int dimN = max(N_Global / N2M4TilingConfig::TILE_N, 1);  // max(N_Global/N2M4TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int dimM = M_Global / N2M4TilingConfig::TILE_M * Split_K;  // 使用 Split_K 策略增加线程块的数量，提升占用率

    // 设置 Grid 内线程块的排布 
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy、
    // 设置线程块内线程的排布
    dim3 BlockDim(WARP_SIZE * N2M4TilingConfig::BLOCK_WARPS, 1, 1);  

    // 异步启动 GPU 上的主计算内核
    SpMM_N2M4_Kernel<N2M4TilingConfig, stages><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(Compressed_A, metadata, B, C, M_Global, N_Global, K_Global, Split_K);
    return cudaPeekAtLastError();
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


    // N2M4TilingConfig<mma_k, block_row_warps, block_col_warps, warp_col_tensors, warp_row_tensors, block_k_steps>
    // 各参数含义：
    // - mma_k: 单次 mma 指令在 K 方向处理的长度
    // - block_row_warps: 一个 thread block 在 M 方向排布的 warp 数
    // - block_col_warps: 一个 thread block 在 N 方向排布的 warp 数
    // - warp_col_tensors: 单个 warp 在 N 方向负责的 mma tile 数
    // - warp_row_tensors: 单个 warp 在 M 方向负责的 mma tile 数
    // - block_k_steps: 一个 block 在 K 方向累计的 mma 步数
    // 派生关系：
    // - TILE_M = 16 * (warp_row_tensors * block_row_warps)
    // - TILE_N = 8  * (warp_col_tensors * block_col_warps)
    // - TILE_K = mma_k * block_k_steps
    // - BLOCK_WARPS = block_row_warps * block_col_warps
    // - BLOCK_THREADS = 32 * BLOCK_WARPS
    cudaError_t Error = cudaSuccess;
    switch (N_Global) {
        case 8:
            // <16,4,1,1,1,4> -> TILE_M=64, TILE_N=8, TILE_K=64, BLOCK_THREADS=128
            Error = SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 1, 1, 1, 4>, 2>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            // <16,4,1,2,1,4> -> TILE_M=64, TILE_N=16, TILE_K=64, BLOCK_THREADS=128
            Error = SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 1, 2, 1, 4>, 2>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            // <16,4,1,4,1,4> -> TILE_M=64, TILE_N=32, TILE_K=64, BLOCK_THREADS=128
            Error = SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 1, 4, 1, 4>, 2>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            // <16,2,2,4,2,4> -> TILE_M=64, TILE_N=64, TILE_K=64, BLOCK_THREADS=128
            Error = SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 2, 2, 4, 2, 4>, 2>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            // <16,2,2,4,4,4> -> TILE_M=128, TILE_N=64, TILE_K=64, BLOCK_THREADS=128
            Error = SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 2, 2, 4, 4, 4>, 2>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;    
        // TODO 这里配置还要考虑一下        
        case 256:
            // 先复用 N=128 的保守配置；更激进的更宽 tile 需要单独验证正确性和资源占用。
            // <16,2,2,4,4,4> -> TILE_M=128, TILE_N=64, TILE_K=64, BLOCK_THREADS=128
            Error = SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 2, 2, 8, 4, 4>, 2>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;  
        case 512:
            // Keep the same 128x128x64 tile and 3-stage pipeline, but retest the
            // earlier 4x2 warp layout so each warp carries more N fragments and
            // fewer M fragments. This bypasses the direct-output specialization
            // and isolates the launch policy as the only active variable again.
            Error = SpMM_N2M4_Kernel_API<N2M4ConfigN128Wide, 3>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;  
        case 1024:
            // The previous TILE_N=1024 / 1024-thread launch is invalid on sm86/A40:
            // its dynamic shared-memory request is 280576B, far above the 101376B
            // opt-in limit. Keep the proven 128x128x64 / 256-thread tile, but
            // use the slimmer metadata ring to make a 4-stage pipeline legal.
            Error = SpMM_N2M4_Kernel_API<N2M4ConfigN128Wide, 4>(
                stream, Compressed_A, B, metadata, KernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            printf("[SpMM] unsupported N_Global=%d.\n", N_Global);
            return cudaErrorInvalidValue;
    }
    
    // 在内核启动后检查是否发生了错误
    if (Error != cudaSuccess)
        return Error;

    // 如果没有使用 split_k 则返回结果
    if (Split_K == 1)
        return Error;

    // 重新分配 Grid 和 Block，启动 Split_k 规约计算的内核
    dim3 GridDim(max((M_Global * N_Global) / 256, 1), 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, KernelOutputPtr, M_Global, N_Global, Split_K);
    return cudaGetLastError();

}


