#include "TileConfig.h"
#include "MMA_PTX.cuh"
#include "LoadAndStore.cuh"



// 每个 Warp 中的 MMA 计算流水线
template<typename N2M4TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputations(      float c[][REG_PER_C_TENSOR_16_8],       // 存储结果
                                                                uint32_t __restrict__ a[][4],           // 压缩矩阵 A
                                                                uint32_t __restrict__ b[][4],           // 激活矩阵 B
                                                                uint32_t __restrict__ metadata[],       // metadata
                                                                half* __restrict__ SharedMemoryPTR_A,   // 矩阵 A 的共享内存指针
                                                                half* __restrict__ SharedMemoryPTR_B,   // 矩阵 B 的共享内存指针
                                                                int warp_start_row,
                                                                int warp_start_col)
{
    // c 寄存器数组的原始类型是 float。但是 MMA 指令的封装 MMA_FP16_M16N8K16 的输入输出参数是 uint32_t*
    // 这行代码使用 reinterpret_cast 创建了一个 c 数组的 uint32_t “视图”，使得后续可以直接将它传递给 MMA 函数，而不会进行实际的数据转换
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_8] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_8]>(c);

// 在 Tile_K = 64 MMA_K_SP = 32 的情况下只进行两次迭代，双缓冲区的收益较低，所以这里不进行双缓冲区的设计
// 这里目前只支持 Tile_K = 64 的情况
#pragma unroll
    for (int k = 0; k < N2M4TilingConfig::BLOCK_K_STEPS; k += 2){  // 这里直接按照 K=64 一次进行迭代
        // K=64 的前半部分
        // 从共享内存中读取激活矩阵的数据分片到寄存器中
        B_FragLoadFromSharedToRegisters_K32<N2M4TilingConfig::WARP_COL_TENSORS>(b, SharedMemoryPTR_B, warp_start_col, k);
        // 从共享内存中读取权重矩阵的数据分片到寄存器中
        A_FragLoadFromSharedToRegisters_K32<N2M4TilingConfig::WARP_ROW_TENSORS>(a, SharedMemoryPTR_A, warp_start_row, k);
        // 进行计算

        // 这里是一个 Wrap 的执行流程
        #pragma unroll
        for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS; i++){
            #pragma unroll
            for (int j = 0; j < N2M4TilingConfig::WARP_COL_TENSORS; j++){
                MMA_FP16_SP_M16N8K32_0x0(c_uint32_t[i*N2M4TilingConfig::WARP_COL_TENSORS + j], a[i], b[j], metadata[i]);
            }
        }

        // K=64 的后半部分
        // 从共享内存中读取激活矩阵的数据分片到寄存器中
        B_FragLoadFromSharedToRegisters_K32<N2M4TilingConfig::WARP_COL_TENSORS>(b, SharedMemoryPTR_B, warp_start_col, k+1);
        // 从共享内存中读取权重矩阵的数据分片到寄存器中
        A_FragLoadFromSharedToRegisters_K32<N2M4TilingConfig::WARP_ROW_TENSORS>(a, SharedMemoryPTR_A, warp_start_row, k+1);

        #pragma unroll
        for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS; i++){
            #pragma unroll
            for (int j = 0; j < N2M4TilingConfig::WARP_COL_TENSORS; j++){
                MMA_FP16_SP_M16N8K32_0x1(c_uint32_t[i*N2M4TilingConfig::WARP_COL_TENSORS + j], a[i], b[j], metadata[i]);
            }
        }
    }
}




// 使用 M16N8K16 的 MMA 指令进行计算
// 使用双缓冲区
// 每个 Warp 中的 MMA 计算流水线
template<typename N2M4TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputations_K16(  float c[][REG_PER_C_TENSOR_16_8],       // 存储结果
                                                                uint32_t __restrict__ a[][2],           // 压缩矩阵 A
                                                                uint32_t __restrict__ b[][2],           // 激活矩阵 B
                                                                uint32_t __restrict__ metadata[],       // metadata
                                                                half* __restrict__ SharedMemoryPTR_A,   // 矩阵 A 的共享内存指针
                                                                half* __restrict__ SharedMemoryPTR_B,   // 矩阵 B 的共享内存指针
                                                                int warp_start_row,
                                                                int warp_start_col)
{
    // c 寄存器数组的原始类型是 float。但是 MMA 指令的封装 MMA_FP16_M16N8K16 的输入输出参数是 uint32_t*
    // 这行代码使用 reinterpret_cast 创建了一个 c 数组的 uint32_t “视图”，使得后续可以直接将它传递给 MMA 函数，而不会进行实际的数据转换
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_8] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_8]>(c);

    // 预取第一块数据
    B_FragLoadFromSharedToRegisters_K16<N2M4TilingConfig::WARP_COL_TENSORS>(b, SharedMemoryPTR_B, warp_start_col, 0);
    A_FragLoadFromSharedToRegisters_K16<N2M4TilingConfig::WARP_ROW_TENSORS>(a, SharedMemoryPTR_A, warp_start_row, 0);

    #pragma unroll
    for (int k = 0; k < N2M4TilingConfig::BLOCK_K_STEPS; k ++){  // 这里直接按照 K=64 一次进行迭代
        uint32_t __restrict__(*b_read)[2]  = b + ((k) % 2) * N2M4TilingConfig::WARP_COL_TENSORS;  // 标记当前 TensorCore 计算需要使用的数据空间
        uint32_t __restrict__(*b_write)[2] = b + ((k + 1) % 2) * N2M4TilingConfig::WARP_COL_TENSORS;  // 标记下次 TensorCore 计算需要使用的数据空间

        uint32_t __restrict__(*a_read)[2]  = a + ((k) % 2) * N2M4TilingConfig::WARP_ROW_TENSORS;  // 标记当前 TensorCore 计算需要使用的数据空间
        uint32_t __restrict__(*a_write)[2] = a + ((k + 1) % 2) * N2M4TilingConfig::WARP_ROW_TENSORS;  // 标记下次 TensorCore 计算需要使用的数据空间

        if (k + 1 < N2M4TilingConfig::BLOCK_K_STEPS) {  // 最后一次迭代不需要加载下一块数据
            B_FragLoadFromSharedToRegisters_K16<N2M4TilingConfig::WARP_COL_TENSORS>( b_write, SharedMemoryPTR_B, warp_start_col, k+1);
            A_FragLoadFromSharedToRegisters_K16<N2M4TilingConfig::WARP_ROW_TENSORS>(a_write, SharedMemoryPTR_A, warp_start_row, k+1);
        }

        // 这里是一个 Wrap 的执行流程
        #pragma unroll
        for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS; i++){
            #pragma unroll
            for (int j = 0; j < N2M4TilingConfig::WARP_COL_TENSORS; j++){
                if(k % 4 == 0){
                    MMA_FP16_SP_M16N8K16_0x0(c_uint32_t[i*N2M4TilingConfig::WARP_COL_TENSORS + j], a_read[i], b_read[j], metadata[i]);
                }else if(k % 4 == 1){
                    MMA_FP16_SP_M16N8K16_0x1(c_uint32_t[i*N2M4TilingConfig::WARP_COL_TENSORS + j], a_read[i], b_read[j], metadata[i]);
                }else if(k % 4 == 2){
                    MMA_FP16_SP_M16N8K16_0x2(c_uint32_t[i*N2M4TilingConfig::WARP_COL_TENSORS + j], a_read[i], b_read[j], metadata[i]);
                }else{
                    MMA_FP16_SP_M16N8K16_0x3(c_uint32_t[i*N2M4TilingConfig::WARP_COL_TENSORS + j], a_read[i], b_read[j], metadata[i]);
                }
            }
        }

    }
}









// // __global__ 函数，是从CPU端启动的GPU内核的入口函数
// // 控制内存管理、数据流控制、软件流水线以及最终结果的写回
// template<typename N2M4TilingConfig>
// __global__ void SpMM_N2M4_Kernel(   const half* Compressed_A,  // 
//                                     const uint16_t* metadata,
//                                     const half*  B,
//                                     half* C,
//                                     const int    M_Global,
//                                     const int    N_Global,
//                                     const int    K_Global,
//                                     const int    Split_K)
// {
//     // 1. 坐标计算
//     const int splitK_Idx  = blockIdx.y / (M_Global / N2M4TilingConfig::TILE_M);  // 计算出当前线程块属于第几个 Split_K
//     const int IsLastsplitK = (splitK_Idx == (Split_K - 1));  // 判断当前是不是最后一个 Split_K

//     // 坐标计算
//     const int tileIdxM = blockIdx.y % (M_Global / N2M4TilingConfig::TILE_M);  // 当前tile块在整个矩阵中的逻辑行数，注意这里使用了 Split_K
//     const int tileIdxN = blockIdx.x;  // 当前tile块在整个矩阵中的逻辑列数

//     // 2. 偏移量与规约次数计算
//     const int tileStartRow = tileIdxM * N2M4TilingConfig::TILE_M;  // 当前 Tile 在整个矩阵中对应的起始行数
//     const int tileStartCol = tileIdxN * N2M4TilingConfig::TILE_N;  // 当前 Tile 在整个矩阵中对应的起始列数

//     // 计算规约轴需要迭代的次数
//     const int tileKBlockNum = K_Global / N2M4TilingConfig::TILE_K;  // 在规约轴上一次处理 TILE_K 长度，当前 tile 需要计算多少次
//     const int AveragetileKBlock = (tileKBlockNum - 1) / Split_K + 1;  // 采取 Split_K 策略，算每个 Split_K 平均需要处理多少个 Tile，向上取整
//     // 如果当前是最后最后一个 Split_K 在计算时需要去除多余的填充块
//     const int RoundedKBlock    = AveragetileKBlock * Split_K;
//     const int PaddingKBlock    = RoundedKBlock - tileKBlockNum;
//     // 如果是最后一个 Split_K，则需要减少相应的迭代次数
//     int iterateK = 0;
//     if (IsLastsplitK)
//         iterateK = AveragetileKBlock - PaddingKBlock;
//     else
//         iterateK = AveragetileKBlock;

//     // 3. 共享内存布局与Warp级任务分配
//     // 开辟一段共享内存空间，其大小由启动内核时使用的 SEME_SZ 参数确定，并且声明这段内存是 fp16 类型的数组
//     // __align__(128) 表示内存对齐，起始地址必须是128字节的整数倍
//     extern __shared__ __align__(128) half sharedMem[];  // at least be 128 Bytes aligned

//     // 共享内存分配
//     // 这里只需要考虑输入数据的分配，计算完成后这些数据就可以直接被计算结果覆盖
//     // 使用双缓冲区设计
//     half* sharedMemMatrixA = &sharedMem[0];  // 压缩矩阵 A
//     half* sharedMemMatrixB = &sharedMem[(N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)*2];  // 激活矩阵 B
//     uint16_t* sharedMemMetadata =  reinterpret_cast<uint16_t*>(&sharedMem[(N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)*2+(N2M4TilingConfig::TILE_K * N2M4TilingConfig::TILE_N)*2]);  // metadata

//     // warp 任务分配
//     const unsigned int warpId = threadIdx.x / WARP_SIZE;  // 确定当前是第几个 warp 
//     int warpIdxM = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;  // warp 在线程块中的行号
//     int warpIdxN = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;  // warp 在线程块中的列号
//     int warpStartRow = N2M4TilingConfig::WARP_ROW_TENSORS * N2M4TilingConfig::MMA_M_SP * warpIdxM;  // 计算当前 Warp 的起始行是线程块中的第几行
//     int warpStartCol = N2M4TilingConfig::WARP_COL_TENSORS * N2M4TilingConfig::MMA_N_SP * warpIdxN;  // 计算当前 Warp 的起始列是线程块中的第几列

//     // 寄存器空间声明
//     uint32_t __restrict__ _a[N2M4TilingConfig::WARP_ROW_TENSORS][4];  // 稀疏矩阵
//     uint32_t __restrict__ _b[N2M4TilingConfig::WARP_COL_TENSORS][4];  // 稠密矩阵
//     uint32_t __restrict__ _metadata[N2M4TilingConfig::WARP_ROW_TENSORS];  // metadata

//     // 确定全局内存中的数据起始地址
//     const half* ATileGlobalPTR = Compressed_A + tileStartRow * K_Global / 2 + splitK_Idx * N2M4TilingConfig::MMA_M_SP * AveragetileKBlock * N2M4TilingConfig::TILE_K / 2;  // 权重矩阵的数据起始地址 
//     const half* BTileGlobalPTR = B + tileStartCol * K_Global + splitK_Idx * AveragetileKBlock * N2M4TilingConfig::TILE_K * 8;  // 激活矩阵的数据起始地址，注意这里是打包之后的数据，打包的时候是以 N2M4TilingConfig::TILE_K * 8 为一个单元进行打包的
//     const uint16_t* metadataTileGlobalPTR = metadata + tileStartRow * K_Global / 16 + splitK_Idx * N2M4TilingConfig::MMA_M_SP * AveragetileKBlock * N2M4TilingConfig::TILE_K / 16;  // metadata的数据起始地址 

//     // 计算流水线
//     // 共享内存数据预加载
//     async_copy_metadata_x_128B<N2M4TilingConfig::TILE_M, N2M4TilingConfig>(sharedMemMetadata, metadataTileGlobalPTR, K_Global);  // 加载 metadata
//     async_copy_matrixA_x_32<N2M4TilingConfig::TILE_M, N2M4TilingConfig>(sharedMemMatrixA, ATileGlobalPTR, K_Global / 2);  // 加载压缩权重矩阵 A
//     cp_async_group_commit();

//     async_copy_matrixB_x_64_V2<N2M4TilingConfig::TILE_N, N2M4TilingConfig>(sharedMemMatrixB, BTileGlobalPTR, K_Global);  // 加载激活矩阵 B
//     cp_async_group_commit();
    

//     // 初始化输出矩阵为 0， 这个操作与前面的数据加载操作完全并行执行
//     float c[N2M4TilingConfig::WARP_ROW_TENSORS * N2M4TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_8];
    
//     #pragma unroll
//     for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS * N2M4TilingConfig::WARP_COL_TENSORS; i++)
//         #pragma unroll
//         for (int j = 0; j < REG_PER_C_TENSOR_16_8; j++)
//             c[i][j] = 0.0f;

//     // 等待矩阵 A 和 metadata 加载到 SharedMem
//     cp_async_wait_group<1>(); // B loading done
//     __syncthreads();

//     metadata_FragLoadFromSharedToRegisters<N2M4TilingConfig::WARP_ROW_TENSORS>(_metadata, sharedMemMetadata, warpStartRow);  // 加载 metadata

//     // 等待加载完成
//     cp_async_wait_group<0>(); // B loading done
//     __syncthreads();


//     // 流水线主循环
//     for (int tileIdxK = 0; tileIdxK < iterateK; tileIdxK++){
//         // 全局地址更新
//         ATileGlobalPTR = ATileGlobalPTR + N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::TILE_K / 2; 
//         BTileGlobalPTR = BTileGlobalPTR + N2M4TilingConfig::TILE_K * 8;  // 注意这里是打包之后的数据
//         metadataTileGlobalPTR = metadataTileGlobalPTR + N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::TILE_K / 16;

//         // double buffer
//         // smem_write_B_PTR: 计算出本次迭代应该写入的共享内存缓冲区的地址。在第 k 次迭代，写入的是第 k+1 的数据，所以用 tile_id_k + 1。
//         // smem_read_B_PTR: 计算出本次迭代应该读取的共享内存缓冲区的地址。在第 k 次迭代，读取的是第 k 的数据，所以用 tile_id_k。
//         half* __restrict__ smem_write_B_PTR = sharedMemMatrixB;
//         half* __restrict__ smem_read_B_PTR  = sharedMemMatrixB;
//         smem_write_B_PTR = sharedMemMatrixB + ((tileIdxK + 1) % 2) * (N2M4TilingConfig::TILE_K * N2M4TilingConfig::TILE_N);  // The current write address of B
//         smem_read_B_PTR  = sharedMemMatrixB + ((tileIdxK) % 2) * (N2M4TilingConfig::TILE_K * N2M4TilingConfig::TILE_N);  // The current reading address of B

//         // 稀疏矩阵 A 的双缓冲区
//         half* __restrict__ smem_write_A_PTR = sharedMemMatrixA;
//         half* __restrict__ smem_read_A_PTR  = sharedMemMatrixA;
//         smem_write_A_PTR = sharedMemMatrixA + ((tileIdxK + 1) % 2) * (N2M4TilingConfig::TILE_K / 2 * N2M4TilingConfig::TILE_M);  // The current write address of B
//         smem_read_A_PTR  = sharedMemMatrixA + ((tileIdxK) % 2) * (N2M4TilingConfig::TILE_K / 2 * N2M4TilingConfig::TILE_M);  // The current reading address of B


//         // COPY indicator
//         // 判断是不是最后一个块，如果是最后一个块后续的读取操作就不继续进行
//         bool GlobalCopy = (tileIdxK + 1) < iterateK;


//         async_copy_metadata_x_128B<N2M4TilingConfig::TILE_M, N2M4TilingConfig>(sharedMemMetadata, metadataTileGlobalPTR, K_Global, GlobalCopy);  // 加载 metadata
//         async_copy_matrixA_x_32<N2M4TilingConfig::TILE_M, N2M4TilingConfig>(smem_write_A_PTR, ATileGlobalPTR, K_Global / 2, GlobalCopy);  // 加载压缩权重矩阵 A
//         cp_async_group_commit();

//         async_copy_matrixB_x_64_V2<N2M4TilingConfig::TILE_N, N2M4TilingConfig>(smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);  // 加载激活矩阵 B
//         cp_async_group_commit();

//         // 执行计算
//         PipelinedCoreComputations<N2M4TilingConfig>(c, _a, _b, _metadata, smem_read_A_PTR, smem_read_B_PTR, warpStartRow, warpStartCol);

//         cp_async_wait_group<1>();
//         __syncthreads();

//         metadata_FragLoadFromSharedToRegisters<N2M4TilingConfig::WARP_ROW_TENSORS>(_metadata, sharedMemMetadata, warpStartRow);  // 加载 metadata

//         cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
//         __syncthreads();

//     }



//     // 结果写回
//     // Store the C fragments to shared memory.
//     // 把 smem 这个 half* 指针，当作一个指向数组的指针来看待
//     // 这个数组的每个元素本身又是一个大小为 TILE_N + PADDING_SHARED_MEM_FOR_C 的 half 数组
//     // 注意这里进行存储的时候保存成行主序

//     half(*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C] = reinterpret_cast<half(*)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C]>(sharedMem);
//     StoreToSharedMemoryFromRegister_half<N2M4TilingConfig>(smem_CFrag, c);  // 将各个 Wrap 的结果写入共享内存中
//     // __syncthreads();

//     // 计算当前线程块负责的 C Tile 在全局内存中的起始地址
//     half* BlockGlobalPTR = C + splitK_Idx * (M_Global * N_Global) + tileStartRow * N_Global + tileStartCol;  // 注意这里也是行主序

//     // 将最终的结果写回全局内存
//     // #pragma unroll
//     // for (int i = warpId; i < N2M4TilingConfig::TILE_M; i += N2M4TilingConfig::BLOCK_WARPS){  // 一个 warp 负责一行
//     //     #pragma unroll
//     //     for (int j = threadIdx.x % WARP_SIZE; j < N2M4TilingConfig::TILE_N; j += WARP_SIZE){  // warp 内一个线程负责一行中的某一列
//     //         BlockGlobalPTR[j + i * N_Global] = (*(smem_CFrag + i))[j];
//     //     }
//     // }

//     StoreToGlobalMemoryFromShared<N2M4TilingConfig>(smem_CFrag, BlockGlobalPTR, N_Global);  // 将共享内存中的结果写回全局内存


// }




// __global__ 函数，是从CPU端启动的GPU内核的入口函数
// 控制内存管理、数据流控制、软件流水线以及最终结果的写回
// V1 给 metadata 添加双缓冲区，减少__syncthreads()
template<typename N2M4TilingConfig, int stages>
__global__ void SpMM_N2M4_Kernel(   const half* Compressed_A,  // 
                                    const uint16_t* metadata,
                                    const half*  B,
                                    half* C,
                                    const int    M_Global,
                                    const int    N_Global,
                                    const int    K_Global,
                                    const int    Split_K)
{
    // 1. 坐标计算
    const int splitK_Idx  = blockIdx.y / (M_Global / N2M4TilingConfig::TILE_M);  // 计算出当前线程块属于第几个 Split_K
    const int IsLastsplitK = (splitK_Idx == (Split_K - 1));  // 判断当前是不是最后一个 Split_K

    // 坐标计算
    const int tileIdxM = blockIdx.y % (M_Global / N2M4TilingConfig::TILE_M);  // 当前tile块在整个矩阵中的逻辑行数，注意这里使用了 Split_K
    const int tileIdxN = blockIdx.x;  // 当前tile块在整个矩阵中的逻辑列数

    // 2. 偏移量与规约次数计算
    const int tileStartRow = tileIdxM * N2M4TilingConfig::TILE_M;  // 当前 Tile 在整个矩阵中对应的起始行数
    const int tileStartCol = tileIdxN * N2M4TilingConfig::TILE_N;  // 当前 Tile 在整个矩阵中对应的起始列数

    // 计算规约轴需要迭代的次数
    const int tileKBlockNum = K_Global / N2M4TilingConfig::TILE_K;  // 在规约轴上一次处理 TILE_K 长度，当前 tile 需要计算多少次
    const int AveragetileKBlock = (tileKBlockNum - 1) / Split_K + 1;  // 采取 Split_K 策略，算每个 Split_K 平均需要处理多少个 Tile，向上取整
    // 如果当前是最后最后一个 Split_K 在计算时需要去除多余的填充块
    const int RoundedKBlock    = AveragetileKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - tileKBlockNum;
    // 如果是最后一个 Split_K，则需要减少相应的迭代次数
    int iterateK = 0;
    if (IsLastsplitK)
        iterateK = AveragetileKBlock - PaddingKBlock;
    else
        iterateK = AveragetileKBlock;

    // 3. 共享内存布局与Warp级任务分配
    // 开辟一段共享内存空间，其大小由启动内核时使用的 SEME_SZ 参数确定，并且声明这段内存是 fp16 类型的数组
    // __align__(128) 表示内存对齐，起始地址必须是128字节的整数倍
    extern __shared__ __align__(128) half sharedMem[];  // at least be 128 Bytes aligned

    // 共享内存分配
    // 这里只需要考虑输入数据的分配，计算完成后这些数据就可以直接被计算结果覆盖
    // 使用双缓冲区设计
    half* sharedMemMatrixA = &sharedMem[0];  // 压缩矩阵 A
    half* sharedMemMatrixB = &sharedMem[(N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)*stages];  // 激活矩阵 B
    uint16_t* sharedMemMetadata =  reinterpret_cast<uint16_t*>(&sharedMem[(N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)*stages+(N2M4TilingConfig::TILE_K * N2M4TilingConfig::TILE_N)*stages]);  // metadata

    // warp 任务分配
    const unsigned int warpId = threadIdx.x / WARP_SIZE;  // 确定当前是第几个 warp 
    int warpIdxM = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;  // warp 在线程块中的行号
    int warpIdxN = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;  // warp 在线程块中的列号
    int warpStartRow = N2M4TilingConfig::WARP_ROW_TENSORS * N2M4TilingConfig::MMA_M_SP * warpIdxM;  // 计算当前 Warp 的起始行是线程块中的第几行
    int warpStartCol = N2M4TilingConfig::WARP_COL_TENSORS * N2M4TilingConfig::MMA_N_SP * warpIdxN;  // 计算当前 Warp 的起始列是线程块中的第几列

    // 寄存器空间声明
    // K32 不使用双缓冲区 
    // uint32_t __restrict__ _a[N2M4TilingConfig::WARP_ROW_TENSORS][4];  // 稀疏矩阵
    // uint32_t __restrict__ _b[N2M4TilingConfig::WARP_COL_TENSORS][4];  // 稠密矩阵
    // uint32_t __restrict__ _metadata[N2M4TilingConfig::WARP_ROW_TENSORS];  // metadata

    // K16 使用双缓冲区
    uint32_t __restrict__ _a[N2M4TilingConfig::WARP_ROW_TENSORS*2][2];  // 稀疏矩阵
    uint32_t __restrict__ _b[N2M4TilingConfig::WARP_COL_TENSORS*2][2];  // 稠密矩阵
    uint32_t __restrict__ _metadata[N2M4TilingConfig::WARP_ROW_TENSORS];  // metadata

    // 确定全局内存中的数据起始地址
    const half* ATileGlobalPTR = Compressed_A + tileStartRow * K_Global / 2 + splitK_Idx * N2M4TilingConfig::MMA_M_SP * AveragetileKBlock * N2M4TilingConfig::TILE_K / 2;  // 权重矩阵的数据起始地址 
    const half* BTileGlobalPTR = B + tileStartCol * K_Global + splitK_Idx * AveragetileKBlock * N2M4TilingConfig::TILE_K * 8;  // 激活矩阵的数据起始地址，注意这里是打包之后的数据，打包的时候是以 N2M4TilingConfig::TILE_K * 8 为一个单元进行打包的
    const uint16_t* metadataTileGlobalPTR = metadata + tileStartRow * K_Global / 16 + splitK_Idx * N2M4TilingConfig::MMA_M_SP * AveragetileKBlock * N2M4TilingConfig::TILE_K / 16;  // metadata的数据起始地址 

    // 计算流水线
    // 共享内存数据预加载
    // 多级缓冲区预加载
    #pragma unroll
    for(int _stage = 0; _stage < min(stages-1, iterateK); _stage++){
        async_copy_metadata_x_128B<N2M4TilingConfig::TILE_M, N2M4TilingConfig>((sharedMemMetadata + _stage * (N2M4TilingConfig::TILE_K / 16 * N2M4TilingConfig::TILE_M)), metadataTileGlobalPTR, K_Global);  // 加载 metadata
        async_copy_matrixA_x_32<N2M4TilingConfig::TILE_M, N2M4TilingConfig>((sharedMemMatrixA + _stage * (N2M4TilingConfig::TILE_K / 2 * N2M4TilingConfig::TILE_M)), ATileGlobalPTR, K_Global / 2);  // 加载压缩权重矩阵 A
        async_copy_matrixB_x_64_V2<N2M4TilingConfig::TILE_N, N2M4TilingConfig>((sharedMemMatrixB + _stage * (N2M4TilingConfig::TILE_K * N2M4TilingConfig::TILE_N)), BTileGlobalPTR, K_Global);  // 加载激活矩阵 B
        cp_async_group_commit();

        ATileGlobalPTR = ATileGlobalPTR + N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::TILE_K / 2; 
        BTileGlobalPTR = BTileGlobalPTR + N2M4TilingConfig::TILE_K * 8;  // 注意这里是打包之后的数据
        metadataTileGlobalPTR = metadataTileGlobalPTR + N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::TILE_K / 16;
    }
    

    // 初始化输出矩阵为 0， 这个操作与前面的数据加载操作完全并行执行
    float c[N2M4TilingConfig::WARP_ROW_TENSORS * N2M4TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_8];
    
    #pragma unroll
    for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS * N2M4TilingConfig::WARP_COL_TENSORS; i++)
        #pragma unroll
        for (int j = 0; j < REG_PER_C_TENSOR_16_8; j++)
            c[i][j] = 0.0f;

   int prefetch = min(stages - 1, iterateK);

    // 确保 stage0 ready：prefetch==0 就不用等（iterateK==0 不会进主循环）
    if (prefetch <= 1) {
        cp_async_wait_group<0>();
    } else {
        // prefetch>=2：让只剩 1 组可能未完成
        // 因为 stages 是编译期常量，所以这里可以直接用 stages-2（注意：prefetch最多就是stages-1）
        cp_async_wait_group<stages - 2>();
    }
    __syncthreads();

    // 流水线主循环
    for (int tileIdxK = 0; tileIdxK < iterateK; tileIdxK++){
        // double buffer
        // smem_write_B_PTR: 计算出本次迭代应该写入的共享内存缓冲区的地址。在第 k 次迭代，写入的是第 k+1 的数据，所以用 tile_id_k + 1。
        // smem_read_B_PTR: 计算出本次迭代应该读取的共享内存缓冲区的地址。在第 k 次迭代，读取的是第 k 的数据，所以用 tile_id_k。
        half* __restrict__ smem_write_B_PTR = sharedMemMatrixB;
        half* __restrict__ smem_read_B_PTR  = sharedMemMatrixB;
        smem_write_B_PTR = sharedMemMatrixB + ((tileIdxK + (stages-1)) % stages) * (N2M4TilingConfig::TILE_K * N2M4TilingConfig::TILE_N);  // The current write address of B
        smem_read_B_PTR  = sharedMemMatrixB + ((tileIdxK) % stages) * (N2M4TilingConfig::TILE_K * N2M4TilingConfig::TILE_N);  // The current reading address of B

        // 稀疏矩阵 A 的双缓冲区
        half* __restrict__ smem_write_A_PTR = sharedMemMatrixA;
        half* __restrict__ smem_read_A_PTR  = sharedMemMatrixA;
        smem_write_A_PTR = sharedMemMatrixA + ((tileIdxK + (stages-1)) % stages) * (N2M4TilingConfig::TILE_K / 2 * N2M4TilingConfig::TILE_M);  // The current write address of A
        smem_read_A_PTR  = sharedMemMatrixA + ((tileIdxK) % stages) * (N2M4TilingConfig::TILE_K / 2 * N2M4TilingConfig::TILE_M);  // The current reading address of A

        // metadata 的双缓冲区
        uint16_t* __restrict__ smem_write_metadata_PTR = sharedMemMetadata;
        uint16_t* __restrict__ smem_read_metadata_PTR  = sharedMemMetadata;
        smem_write_metadata_PTR = sharedMemMetadata + ((tileIdxK + (stages-1)) % stages) * (N2M4TilingConfig::TILE_K / 16 * N2M4TilingConfig::TILE_M);  // The current write address of metadata
        smem_read_metadata_PTR  = sharedMemMetadata + ((tileIdxK) % stages) * (N2M4TilingConfig::TILE_K / 16 * N2M4TilingConfig::TILE_M);  // The current reading address of metadata


        // COPY indicator
        // 判断是不是最后一个块，如果是最后一个块后续的读取操作就不继续进行
        bool GlobalCopy = (tileIdxK + (stages-1)) < iterateK;

        if (GlobalCopy){
            async_copy_metadata_x_128B<N2M4TilingConfig::TILE_M, N2M4TilingConfig>(smem_write_metadata_PTR, metadataTileGlobalPTR, K_Global);  // 加载 metadata
            async_copy_matrixA_x_32<N2M4TilingConfig::TILE_M, N2M4TilingConfig>(smem_write_A_PTR, ATileGlobalPTR, K_Global / 2);  // 加载压缩权重矩阵 A
            async_copy_matrixB_x_64_V2<N2M4TilingConfig::TILE_N, N2M4TilingConfig>(smem_write_B_PTR, BTileGlobalPTR, K_Global);  // 加载激活矩阵 B
            cp_async_group_commit();

            // 全局地址更新
            ATileGlobalPTR = ATileGlobalPTR + N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::TILE_K / 2; 
            BTileGlobalPTR = BTileGlobalPTR + N2M4TilingConfig::TILE_K * 8;  // 注意这里是打包之后的数据
            metadataTileGlobalPTR = metadataTileGlobalPTR + N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::TILE_K / 16;

        }

        // 加载 metadata 到寄存器
        metadata_FragLoadFromSharedToRegisters<N2M4TilingConfig::WARP_ROW_TENSORS>(_metadata, smem_read_metadata_PTR, warpStartRow);
        // 执行计算
        PipelinedCoreComputations_K16<N2M4TilingConfig>(c, _a, _b, _metadata, smem_read_A_PTR, smem_read_B_PTR, warpStartRow, warpStartCol);

        if (GlobalCopy){
            cp_async_wait_group<stages - 2>();  // 数据加载完成
            __syncthreads();
        }
    }

    // // debug
    // #pragma unroll
    // for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS * N2M4TilingConfig::WARP_COL_TENSORS; i++)
    //     #pragma unroll
    //     for (int j = 0; j < REG_PER_C_TENSOR_16_8; j++)
    //         if(c[i][j] != 256.0f)
    //             printf("c[%d][%d] = %f\n", i, j, c[i][j]);

    // 结果写回
    // Store the C fragments to shared memory.
    // 把 smem 这个 half* 指针，当作一个指向数组的指针来看待
    // 这个数组的每个元素本身又是一个大小为 TILE_N + PADDING_SHARED_MEM_FOR_C 的 half 数组
    // 注意这里进行存储的时候保存成行主序

    __syncthreads();
    half(*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C] = reinterpret_cast<half(*)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C]>(sharedMem);
    StoreToSharedMemoryFromRegister_half<N2M4TilingConfig>(smem_CFrag, c);  // 将各个 Wrap 的结果写入共享内存中

    // 计算当前线程块负责的 C Tile 在全局内存中的起始地址
    half* BlockGlobalPTR = C + splitK_Idx * (M_Global * N_Global) + tileStartRow * N_Global + tileStartCol;  // 注意这里也是行主序

    // 将最终的结果写回全局内存
    // #pragma unroll
    // for (int i = warpId; i < N2M4TilingConfig::TILE_M; i += N2M4TilingConfig::BLOCK_WARPS){  // 一个 warp 负责一行
    //     #pragma unroll
    //     for (int j = threadIdx.x % WARP_SIZE; j < N2M4TilingConfig::TILE_N; j += WARP_SIZE){  // warp 内一个线程负责一行中的某一列
    //         BlockGlobalPTR[j + i * N_Global] = (*(smem_CFrag + i))[j];
    //     }
    // }

    StoreToGlobalMemoryFromShared<N2M4TilingConfig>(smem_CFrag, BlockGlobalPTR, N_Global);  // 将共享内存中的结果写回全局内存


}




