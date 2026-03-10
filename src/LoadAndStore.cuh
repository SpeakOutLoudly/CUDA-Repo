#pragma once  // 仅编译一次

#include "TileConfig.h"
#include "AsyncCopy_PTX.cuh"


// 加载激活矩阵， Tile_K * Tile_N 大小的子矩阵，每次加载 64*8 个数据
// 激活矩阵按照行主序的方式在 Global Memory 中进行存储，此处需要注意在读取时也要符合要求
// 设计理念：
// - 每 64*8 个数据为一个单元，一个 warp 单次进行 32*8 个数据的拷贝；
// - 行内采用异或置换(col ^ row)的列地址写入，减少 bank conflict；
// - 与 MatMulUtilities 的加载策略一致，便于后续 ldmatrix/B fragment 读取。
template<int NumNToCopy, typename N2M4TilingConfig>
__device__ __forceinline__ void async_copy_matrixB_x_64(    half* __restrict__ sharedMem_PTR,           // 共享内存起点
                                                            const half* __restrict__ globalMem_PTR,     // 全局内存起点
                                                            const int GlobalStride,                     // 全局内存行跨度
                                                            bool Pred = true)                           // 谓词：边界保护
{
    static_assert(NumNToCopy % 8 == 0, "NumNToCopy must be a multiple of 8");  // 这里要求加载的 N 维长度是 8 的倍数

    // 为了便于重排，需要按照行主序的方式进行存储，加载指令最好每次加载 128bit 8个fp16数
    int warp_id = threadIdx.x / 32;  // 当前是第几个 warp 
    int lane_id = threadIdx.x % 32;  // 0..31, 线程在 warp 内的 id

    // 在存储到 shared memory 中需要对数据进行映射，否则很容易导致 bank 冲突
    // 在逻辑上以 4*8 的二维方式组织
    int col     = lane_id % 8;            // 0..7
    int row1    = lane_id / 8;            // 0..3
    int row2    = row1 + 4;               // 4..7

    // 列置换，按行异或，规避 bank 冲突
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;

    // 8 行为一个拷贝单元；每个单元内，每个线程完成两次 128-bit（16B）拷贝
    // 计算一共需要多少个拷贝单元
    int CopyUnitsNums = NumNToCopy / async_copy_unit_N;

    // 每个单元需要一个 warp 进行拷贝，一个线程块有 BLOCK_ROW_WARPS * BLOCK_COL_WARPS 个 warp
    int MaxIteration = (CopyUnitsNums - 1) / (N2M4TilingConfig::BLOCK_WARPS) + 1;

#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {

        int unitIdx = i * (N2M4TilingConfig::BLOCK_WARPS) + warp_id;  // 当前是第几个拷贝单元
        bool AsyncCopyPredictor = unitIdx < CopyUnitsNums && Pred;
        // 计算当前拷贝单元的全局内存起点
        const half* __restrict__ unitGlobalMem_PTR = globalMem_PTR + unitIdx * async_copy_unit_N;         // 由于按照行主序进行存储，每个单元的起点都在同一行，这里直接计算行偏移就行
        half* __restrict__ unitSharedMem_PTR = sharedMem_PTR + unitIdx * 8 * 64;          // 目标地址按照单元数据进行跳跃，每个单元包含 8*64 个数据

        // 每次沿K方向拷贝 128-bit（8 个 half）
        cp_async<16>(   unitSharedMem_PTR + store_column1 * 8 + row1 * 64,    // 目标地址
                        unitGlobalMem_PTR + lane_id * GlobalStride,           // 源地址，由于是行主序，所以这里 GlobalStride = Global_N
                        AsyncCopyPredictor);
        cp_async<16>(   unitSharedMem_PTR + store_column2 * 8 + row2 * 64,
                        unitGlobalMem_PTR + (lane_id + 32) * GlobalStride,
                        AsyncCopyPredictor);
    }
}



// 加载激活矩阵， Tile_K * Tile_N 大小的子矩阵，每次加载 64*8 个数据
// 激活矩阵进行过pack，原始矩阵按照 64*8 分为一个子块，组内的数据先进行行主序存储，每组之间列主序排布，需要注意在读取时也要符合要求
// 设计理念：
// - 每 64*8 个数据为一个单元，一个 warp 单次进行 32*8 个数据的拷贝；
// - 行内采用异或置换(col ^ row)的列地址写入，减少 bank conflict；
// - 与 MatMulUtilities 的加载策略一致，便于后续 ldmatrix/B fragment 读取。
template<int NumNToCopy, typename N2M4TilingConfig>
__device__ __forceinline__ void async_copy_matrixB_x_64_V2( half* __restrict__ sharedMem_PTR,           // 共享内存起点
                                                            const half* __restrict__ globalMem_PTR,     // 全局内存起点
                                                            const int GlobalStride,                     // 全局内存行跨度
                                                            bool Pred = true)                           // 谓词：边界保护
{
    static_assert(NumNToCopy % 8 == 0, "NumNToCopy must be a multiple of 8");  // 这里要求加载的 N 维长度是 8 的倍数

    // 为了便于重排，需要按照行主序的方式进行存储，加载指令最好每次加载 128bit 8个fp16数
    int warp_id = threadIdx.x / 32;  // 当前是第几个 warp 
    int lane_id = threadIdx.x % 32;  // 0..31, 线程在 warp 内的 id

    // 在存储到 shared memory 中需要对数据进行映射，否则很容易导致 bank 冲突
    // 在逻辑上以 4*8 的二维方式组织
    int col     = lane_id % 8;            // 0..7
    int row1    = lane_id / 8;            // 0..3
    int row2    = row1 + 4;               // 4..7

    // 列置换，按行异或，规避 bank 冲突
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;

    // 8 行为一个拷贝单元；每个单元内，每个线程完成两次 128-bit（16B）拷贝
    // 计算一共需要多少个拷贝单元
    int CopyUnitsNums = NumNToCopy / async_copy_unit_N;

    // 每个单元需要一个 warp 进行拷贝，一个线程块有 BLOCK_ROW_WARPS * BLOCK_COL_WARPS 个 warp
    int MaxIteration = (CopyUnitsNums - 1) / (N2M4TilingConfig::BLOCK_WARPS) + 1;

#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {

        int unitIdx = i * (N2M4TilingConfig::BLOCK_WARPS) + warp_id;  // 当前是第几个拷贝单元
        bool AsyncCopyPredictor = unitIdx < CopyUnitsNums && Pred;
        // 计算当前拷贝单元的全局内存起点
        const half* __restrict__ unitGlobalMem_PTR = globalMem_PTR + unitIdx * async_copy_unit_N * GlobalStride;  // 矩阵进行过 pack 所以此处需要按照整个矩阵的 K 维度进行跳跃，GlobalStride = global_K
        half* __restrict__ unitSharedMem_PTR = sharedMem_PTR + unitIdx * 8 * 64;          // 目标地址按照单元数据进行跳跃，每个单元包含 8*64 个数据

        // 每次沿K方向拷贝 128-bit（8 个 half）
        cp_async<16>(   unitSharedMem_PTR + store_column1 * 8 + row1 * 64,    // 目标地址
                        unitGlobalMem_PTR + lane_id * 8,           // 源地址
                        AsyncCopyPredictor);
        cp_async<16>(   unitSharedMem_PTR + store_column2 * 8 + row2 * 64,
                        unitGlobalMem_PTR + (lane_id + 32) * 8,
                        AsyncCopyPredictor);
    }
}






// 加载权重矩阵 TILE_M * TILE_K 大小的子矩阵，由于经过压缩，所以实际上只加载 TILE_M * TILE_K / 2 的数据, TILE_K 一般等于 64
// 并且压缩之后的矩阵是以16*8的块为单位存储的，所以压缩的矩阵在存储时是完全连续的
template<int NumMToCopy, typename N2M4TilingConfig>
__device__ __forceinline__ void async_copy_matrixA_x_32(  half* __restrict__ sharedMem_PTR,         // 共享内存起点（按 K 主维步进）
                                                        const half* __restrict__ globalMem_PTR,     // 全局内存起点（按 GlobalStride 步进）
                                                        const int GlobalStride,                     // 全局内存行跨度（即 global_K / 2）
                                                        bool Pred = true)                           // 谓词：边界保护
{
    static_assert(NumMToCopy % 16 == 0, "NumMToCopy must be a multiple of 16");

    // 每个 Warp 每次加载 16*32 大小的数据
    int warp_id = threadIdx.x / 32;  // 当前是第几个 warp 
    int lane_id = threadIdx.x % 32;  // 0..31, 线程在 warp 内的 id

    // 在存储到 shared memory 中需要对数据进行映射，否则很容易导致 bank 冲突
    // 在逻辑上以 4*8 的二维方式组织
    int col     = lane_id % 8;            // 0..7
    int row1    = lane_id / 8;            // 0..3
    int row2    = row1 + 4;               // 4..7

    // 列置换，按行异或，规避 bank 冲突
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;


    // 16 行为一个拷贝单元；每个单元内，每个线程完成两次 128-bit（16B）拷贝
    // 计算一共需要多少个拷贝单元
    int CopyUnitsNums = NumMToCopy / async_copy_unit_M;

    // 每个单元需要一个 warp 进行拷贝，一个线程块有 BLOCK_ROW_WARPS * BLOCK_COL_WARPS 个 warp
    int MaxIteration = (CopyUnitsNums - 1) / (N2M4TilingConfig::BLOCK_WARPS) + 1;

#pragma unroll
    for (int i = 0; i < MaxIteration; i++){
        int unitIdx = i * (N2M4TilingConfig::BLOCK_WARPS) + warp_id;  // 当前是第几个拷贝单元
        bool AsyncCopyPredictor = unitIdx < CopyUnitsNums && Pred;

        const half* __restrict__ unitGlobalMem_PTR = globalMem_PTR + unitIdx * async_copy_unit_M * GlobalStride;   // 源地址按整个矩阵的K维度跳跃
        half* __restrict__ unitSharedMem_PTR = sharedMem_PTR + unitIdx * 16 * 32 ;      // 目标地址按单元进行跳跃，每个单元包含 16*32 个数据

        // 每次沿K方向拷贝 128-bit（8 个 half）
        // 因为在 shared memory 中进行存储时线程是按照4*8排布的，所以此时64个数存为一行
        cp_async<16>(   unitSharedMem_PTR + store_column1 * 8 + row1 * 64,    // 目标地址
                        unitGlobalMem_PTR + lane_id * 8,    // 源地址
                        AsyncCopyPredictor);

        cp_async<16>(   unitSharedMem_PTR + store_column2 * 8 + row2 * 64,    // 目标地址
                        unitGlobalMem_PTR + (lane_id + 32) * 8,    // 源地址
                        AsyncCopyPredictor);
    }

}

// 加载权重矩阵对应的 metadata，以 16*64 个数据为一个单元
// 2:4 的压缩格式下，一块 16*16 的权重对应 32B 的 metadata，只需要 2 条线程进行读取
// 所以 16*64 的子权重需要加载 128B 的 metadata，只需要 8 条线程
template<int NumMToCopy, typename N2M4TilingConfig>
__device__ __forceinline__ void async_copy_metadata_x_128B(     uint16_t* __restrict__ sharedMem_PTR,           // 共享内存起点（按 K 主维步进）
                                                                const uint16_t* __restrict__ globalMem_PTR,     // 全局内存起点（按 GlobalStride 步进）
                                                                const int GlobalStride,                     // 全局内存行跨度（即 global_K）
                                                                bool Pred = true)                           // 谓词：边界保护
{
    static_assert(NumMToCopy % 16 == 0, "NumMToCopy must be a multiple of 16");

    int group_id = threadIdx.x / 8;  // 每 8 条线程一组，加载 16 * 4 个 uint16 的 metadata
    int idx = threadIdx.x % 8;       // 表示是每组内的第几条线程

    int CopyUnitsNums = NumMToCopy / async_copy_unit_M;  // 计算需要多少个单元
    bool AsyncCopyPredictor = group_id < CopyUnitsNums && Pred;

    const uint16_t* __restrict__ unitGlobalMem_PTR = globalMem_PTR + group_id * GlobalStride;        // 源地址按 16*64 矩阵对应的 metadata 进行偏移
    uint16_t* __restrict__ unitSharedMem_PTR = sharedMem_PTR + group_id  * 64;      // 目标地址按照单元进行偏移，每个单元 64 个 uint16_t 数

    // 每次沿K方向拷贝 128-bit（8 个 uint16_t）
    cp_async<16>(   unitSharedMem_PTR + idx * 8,    // 目标地址
                    unitGlobalMem_PTR + idx * 8,    // 源地址
                    AsyncCopyPredictor);
}


/* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */
/* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */
/* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */


// 矩阵 b 在加载时要求需要列主序的方式进行排布
// 在global memory时矩阵的排布是 Tile_K * Tile_N 以行主序排列，但是实际需要的是列主序, 所以这里需要使用 .trans
// 这里一次性加载 Wrap_N * MMA_K 大小的数据，循环每次加载 MMA_N * MMA_K
template<int NumOfTensors>
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters_K32(    uint32_t __restrict__ Registers[][4],
                                                                        half* __restrict__ sharedMem_PTR,  // 这是当前 Tile 对应的起始地址
                                                                        int warp_start_N,  // 表示当前 N 方向的 warp 的起始列号
                                                                        int k_step_offset  // 表示当前在 K 维度计算到第几个 MMA_K,在加载到shared memory时64 K维度为一个单元，2个MMA_K
                                                                    )
{
    // 计算参与加载的每条线程负责的 shared memory 地址
    int lane_id = threadIdx.x % 32;  // 计算 warp 内的线程号

    // 在存储到 shared memory 时对数据进行了映射，此处做同样的映射
    // 在逻辑上以 4*8 的二维方式组织
    int col     = lane_id % 8;            // 0..7  逻辑列映射
    int row    = lane_id / 8 +  k_step_offset * 4;  // 0..3  逻辑行映射，在存储时第一个 MMA_K 的时候是 0~3，第二个MMA_K是4~7

    // 与 aysc 存储时相同的 XOR 置换
    int read_col = col ^ row;

    // 找到当前 warp 对应的shared memory起始地址
    // 对于矩阵 B，从global memory加载到shared memory中时对于 N 维需要确保是 8 的倍数，并且存储时是以 64*8 为单元存储的
    sharedMem_PTR += warp_start_N / 8 * 8 * 64;
    // 计算在当前 warp 中每条线程负责的地址，aysc 时线程按照 4*8 的方式进行排布，此处地址需要做同样的对应
    // read_col * 8 表示每条线程读取8个连续数据， row * 64 表示一行64个数据
    sharedMem_PTR += row * 64 + read_col * 8;  

    // 把标准的C++指针转换为以共享内存地址为基址的字节数偏移量
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sharedMem_PTR));
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {  // 加载 Wrap_N * MMA_K 大小的数据时将 Wrap_N 维度进行拆分， NumOfTensors = Wrap_N / MMA_N = Wrap_col_Tensor
        // 32条线程一次性加载 4 个 8*8 大小的矩阵
        // 这条命令的意思是每条线程读取各自对应 smem_local_ptr 地址下的连续 128 bit，并且存放到 4 个 32 bit的寄存器中
        // 由于一开始是行主序，此处需要添加 .trans
        asm volatile(   "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                        : "r"(smem_local_ptr)
                        : "memory");
        
        // 计算下次读取的地址，因为在进行存储时以 64*8 为单元，所以此处按照 64*8 进行偏移
        smem_local_ptr += 64 * 8 * sizeof(half);  // sizeof(half) 表示的是字节数
    } 
}


// 矩阵 A 在加载时要求需要行主序的方式进行排布
// 在加载到共享内存时矩阵的排布是 Warp_M * Tile_K
template<int NumOfTensors>
__device__ __forceinline__ void A_FragLoadFromSharedToRegisters_K32(uint32_t __restrict__ Registers[][4],
                                                                    half* __restrict__ sharedMem_PTR,
                                                                    int warp_start_M,
                                                                    int k_step_offset = 0)
{
    // 计算参与加载的每条线程负责的 shared memory 地址
    int lane_id = threadIdx.x % 32;

    // 在存储到 shared memory 时对数据进行了映射，此处做同样的映射
    // 在逻辑上以 4*8 的二维方式组织
    int col     = lane_id % 8;            // 0..7  逻辑列映射
    int row    = lane_id / 8 +  k_step_offset * 4;  // 0..3  逻辑行映射，在存储时第一个 MMA_K 的时候是 0~3，第二个MMA_K是4~7

    // 与 aysc 存储时相同的 XOR 置换
    int read_col = col ^ row;

    // 找到当前 warp 对应的shared memory起始地址
    // 对于矩阵 B，从global memory加载到shared memory中时对于 M 维需要确保是 16 的倍数，并且存储时是以 16*32 为单元存储的
    sharedMem_PTR += warp_start_M / 16 * 16 * 32;
    // 计算在当前 warp 中每条线程负责的地址，aysc 时线程按照 4*8 的方式进行排布，此处地址需要做同样的对应
    // read_col * 8 表示每条线程读取8个连续数据， row * 64 表示一行64个数据
    sharedMem_PTR += row * 64 + read_col * 8;  

    // 把标准的C++指针转换为以共享内存地址为基址的字节数偏移量
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sharedMem_PTR));

    //  这里的 NumOfTensors 是 WARP_ROW_TENSORS 
    #pragma unroll
    for (int i = 0; i<NumOfTensors; i++){
        // M16N8K32 的shape需要加载 16*32 个数据，压缩之后只需要 16*16 个数据，对应 4 个 8*8 的矩阵
        // 这条命令的意思是每条线程读取各自对应 smem_local_ptr 地址下的连续 8*8*16*4/32=128 bit，并且存放到四个 32 bit的寄存器中
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));
        
        // 计算下次读取的地址，因为在进行存储时以 16*32 为单元，所以此处按照 16*32 进行偏移
        smem_local_ptr += 16 * 32 * sizeof(half);  // sizeof(half) 表示的是字节数
    }
}


// 加载 metadata 到寄存器中，虽然计算时使用的是 K32，但是这里一次性加载 K=64 数量的 metadata
// 预先进行了交错重排，此处直接使用 ldmatrix 指令加载一个 8*8 的矩阵就行
template<int NumOfTensors>
__device__ __forceinline__ void metadata_FragLoadFromSharedToRegisters( uint32_t __restrict__ Registers[],
                                                                        uint16_t* __restrict__ sharedMem_PTR,
                                                                        int warp_start_M)
{
    int lane_id = threadIdx.x % 32;
    
    // 找到当前 warp 对应的shared memory起始地址
    // 对于 metadata ，从global memory加载到shared memory中时对于 M 维需要确保是 16 的倍数，并且存储时是以 16*4 为单元存储的
    sharedMem_PTR += warp_start_M / 16 * 16 * 4;

    // 计算在当前 warp 中每条线程负责的地址，aysc 时是直接连续存储的，所以这里直接 *8
    sharedMem_PTR += (lane_id & 7) * 8;

    // 把标准的C++指针转换为以共享内存地址为基址的字节数偏移量
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sharedMem_PTR));


    //  这里的 NumOfTensors 是 WARP_ROW_TENSORS 
    #pragma unroll
    for (int i = 0; i<NumOfTensors; i++){ // 这里每个 WARP_ROW_TENSORS 需要加载一次 metadata
        asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
            : "=r"(Registers[i])
            : "r"(smem_local_ptr));

        // 计算下次读取的地址，因为在进行存储时以 16*4 个uint16数为单元，所以此处按照 16*4 进行偏移
        smem_local_ptr += 16 * 4 * sizeof(uint16_t);  // sizeof(half) 表示的是字节数
    }

}


/* 将计算的结果从寄存器存放回 Shared Memory 中 */ /* 将计算的结果从寄存器存放回 Shared Memory 中 */ /* 将计算的结果从寄存器存放回 Shared Memory 中 */
/* 将计算的结果从寄存器存放回 Shared Memory 中 */ /* 将计算的结果从寄存器存放回 Shared Memory 中 */ /* 将计算的结果从寄存器存放回 Shared Memory 中 */
/* 将计算的结果从寄存器存放回 Shared Memory 中 */ /* 将计算的结果从寄存器存放回 Shared Memory 中 */ /* 将计算的结果从寄存器存放回 Shared Memory 中 */


// 将寄存器计算的结果存放回到 Shared Memory 中
// 后续可以考虑能否使用 float2 进行数据的存储
template<typename N2M4TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister(    float (*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C],  // 注意这里是行主序
                                    float c[][REG_PER_C_TENSOR_16_8])
{
    // 确定当前线程属于第几个 Wrap
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::WARP_ROW_TENSORS);  // 确定当前 Wrap 起始元素的行数
    int                Warp_j_offset = Warp_j * (N2M4TilingConfig::MMA_N_SP * N2M4TilingConfig::WARP_COL_TENSORS);  // 确定当前 Wrap 起始元素的列数
    // 计算当前线程数据 Wrap 中的第几个线程
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS; i++) {
#pragma unroll
        for (int j = 0; j < N2M4TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*8 Tensor
            int RegSetID        = i * N2M4TilingConfig::WARP_COL_TENSORS + j;
            int Tensor_i_offset = Warp_i_offset + i * N2M4TilingConfig::MMA_M_SP;  // 对应 Tensor 的起始行数
            int Tensor_j_offset = Warp_j_offset + j * N2M4TilingConfig::MMA_N_SP;  // 对应 Tensor 的起始列数
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_8; r++) {  // 遍历当前线程负责存储结果的寄存器
                // 16*8 的float矩阵一行 8 个float数对应8条线程
                // 将线程按照行主序的方式，以 4*8 的二维方式组织
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;  // *2 是因为线程负责存储两个连续的flaot结果
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                // if (r >= 4)
                //     col_offset += 8;
                
                // 寄存器由线程持有，所以 c[RegSetID] 表示当前线程负责第 RegSetID 个 MMA 计算结果的寄存器组
                (*(smem_CFrag + Tensor_i_offset + row_offset))[Tensor_j_offset + col_offset] = c[RegSetID][r];
            }
        }
    }
}

// 将寄存器结果写回 shared memory 中，这里修改 shared memory 中的数据类型为 half
template<typename N2M4TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister_half(   half (*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C],  // 注意这里是行主序
                                        float c[][REG_PER_C_TENSOR_16_8])
{
    // 确定当前线程属于第几个 Wrap
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::WARP_ROW_TENSORS);  // 确定当前 Wrap 起始元素的行数
    int                Warp_j_offset = Warp_j * (N2M4TilingConfig::MMA_N_SP * N2M4TilingConfig::WARP_COL_TENSORS);  // 确定当前 Wrap 起始元素的列数
    // 计算当前线程数据 Wrap 中的第几个线程
    int lane_id = threadIdx.x % WARP_SIZE;
    // 每个 warp 将负责计算的结果写回 shared memory
    #pragma unroll
    for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS; i++) {
    #pragma unroll
        for (int j = 0; j < N2M4TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*8 Tensor
            int RegSetID        = i * N2M4TilingConfig::WARP_COL_TENSORS + j;
            int Tensor_i_offset = Warp_i_offset + i * N2M4TilingConfig::MMA_M_SP;  // 对应 Tensor 的起始行数
            int Tensor_j_offset = Warp_j_offset + j * N2M4TilingConfig::MMA_N_SP;  // 对应 Tensor 的起始列数

            // 你原来的映射：REG_PER_C_TENSOR_16_8=4 时，正好是两行 × 两列
            const int col_base = (lane_id % 4) * 2;
            const int row0 = lane_id / 4;
            const int row1 = row0 + 8;

            // Store adjacent FP16 outputs as half2 to reduce shared-store traffic.
            *reinterpret_cast<half2*>(
                &(*(smem_CFrag + Tensor_i_offset + row0))[Tensor_j_offset + col_base]) =
                __halves2half2(__float2half_rn(c[RegSetID][0]),
                               __float2half_rn(c[RegSetID][1]));

            *reinterpret_cast<half2*>(
                &(*(smem_CFrag + Tensor_i_offset + row1))[Tensor_j_offset + col_base]) =
                __halves2half2(__float2half_rn(c[RegSetID][2]),
                               __float2half_rn(c[RegSetID][3]));
        }
    }
}

// 将 shared memory 结果写回 Global Memory，这里 shared memory 中的数据类型为 half
// if constexpr 的写法不适用于 C++ 11
template<typename N2M4TilingConfig>
__device__ __forceinline__ void
StoreToGlobalMemoryFromRegister_half(   half* __restrict__ BlockGlobalPTR,
                                        int N_Global,
                                        float c[][REG_PER_C_TENSOR_16_8])
{
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const int warpIdxM = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;
    const int warpIdxN = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;
    const int warpRowOffset =
        warpIdxM * (N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::WARP_ROW_TENSORS);
    const int warpColOffset =
        warpIdxN * (N2M4TilingConfig::MMA_N_SP * N2M4TilingConfig::WARP_COL_TENSORS);
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int col_base = (lane_id % 4) * 2;
    const int row0 = lane_id / 4;
    const int row1 = row0 + 8;

    #pragma unroll
    for (int i = 0; i < N2M4TilingConfig::WARP_ROW_TENSORS; i++) {
    #pragma unroll
        for (int j = 0; j < N2M4TilingConfig::WARP_COL_TENSORS; j++) {
            const int regSetId = i * N2M4TilingConfig::WARP_COL_TENSORS + j;
            const int tensorRow = warpRowOffset + i * N2M4TilingConfig::MMA_M_SP;
            const int tensorCol = warpColOffset + j * N2M4TilingConfig::MMA_N_SP;

            half* dst0 =
                BlockGlobalPTR + (tensorRow + row0) * N_Global + tensorCol + col_base;
            half* dst1 =
                BlockGlobalPTR + (tensorRow + row1) * N_Global + tensorCol + col_base;

            *reinterpret_cast<half2*>(dst0) =
                __halves2half2(__float2half_rn(c[regSetId][0]),
                               __float2half_rn(c[regSetId][1]));
            *reinterpret_cast<half2*>(dst1) =
                __halves2half2(__float2half_rn(c[regSetId][2]),
                               __float2half_rn(c[regSetId][3]));
        }
    }
}

__device__ __forceinline__ uint32_t PackHalf2ToUint(float x, float y)
{
    union {
        half2 h2;
        uint32_t u32;
    } packed;
    packed.h2 = __halves2half2(__float2half_rn(x), __float2half_rn(y));
    return packed.u32;
}

__device__ __forceinline__ void
StoreToGlobalMemoryFromRegister_half_Coalesced_N128Balanced(
    half* __restrict__ BlockGlobalPTR,
    int N_Global,
    float c[][REG_PER_C_TENSOR_16_8])
{
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const int warpIdxM = warpId / N2M4ConfigN128Balanced::BLOCK_COL_WARPS;
    const int warpIdxN = warpId % N2M4ConfigN128Balanced::BLOCK_COL_WARPS;
    const int warpRowStart =
        warpIdxM * (N2M4ConfigN128Balanced::MMA_M_SP * N2M4ConfigN128Balanced::WARP_ROW_TENSORS);
    const int warpColStart =
        warpIdxN * (N2M4ConfigN128Balanced::MMA_N_SP * N2M4ConfigN128Balanced::WARP_COL_TENSORS);

    const int lane = threadIdx.x % WARP_SIZE;
    const int rowInTensor = lane / 2;
    const int colChunk = lane % 2;
    const int srcRow = rowInTensor & 7;
    const int srcLaneBase = srcRow * 4;
    const int regOffset = (rowInTensor < 8) ? 0 : 2;
    const unsigned int mask = 0xffffffffu;

    #pragma unroll
    for (int i = 0; i < N2M4ConfigN128Balanced::WARP_ROW_TENSORS; i++) {
        #pragma unroll
        for (int j = 0; j < N2M4ConfigN128Balanced::WARP_COL_TENSORS / 2; j++) {
            const int regSetId = i * N2M4ConfigN128Balanced::WARP_COL_TENSORS + j * 2 + colChunk;

            uint4 packed;
            packed.x = PackHalf2ToUint(
                __shfl_sync(mask, c[regSetId][regOffset + 0], srcLaneBase + 0),
                __shfl_sync(mask, c[regSetId][regOffset + 1], srcLaneBase + 0));
            packed.y = PackHalf2ToUint(
                __shfl_sync(mask, c[regSetId][regOffset + 0], srcLaneBase + 1),
                __shfl_sync(mask, c[regSetId][regOffset + 1], srcLaneBase + 1));
            packed.z = PackHalf2ToUint(
                __shfl_sync(mask, c[regSetId][regOffset + 0], srcLaneBase + 2),
                __shfl_sync(mask, c[regSetId][regOffset + 1], srcLaneBase + 2));
            packed.w = PackHalf2ToUint(
                __shfl_sync(mask, c[regSetId][regOffset + 0], srcLaneBase + 3),
                __shfl_sync(mask, c[regSetId][regOffset + 1], srcLaneBase + 3));

            half* dst = BlockGlobalPTR
                + (warpRowStart + i * N2M4ConfigN128Balanced::MMA_M_SP + rowInTensor) * N_Global
                + warpColStart + j * 16 + colChunk * 8;

            *reinterpret_cast<uint4*>(dst) = packed;
        }
    }
}

// template<typename N2M4TilingConfig>
// __device__ __forceinline__ void
// StoreToGlobalMemoryFromShared_V0(  half (*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C],  // 注意这里是行主序
//                                 half* __restrict__ BlockGlobalPTR,   // 当前线程块负责的 C Tile 在全局内存中的起始地址
//                                 int N_Global)
// {
//     const int warpId = threadIdx.x / WARP_SIZE;
//     const int lane   = threadIdx.x % WARP_SIZE;

//     // warp 在 block 内的 2D 坐标
//     const int warpIdxM = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;  // warp 在线程块中的行号
//     const int warpIdxN = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;  // warp 在线程块中的列号

//     // 这个 warp 负责的输出 subtile 起点（相对整个 block tile）
//     constexpr int WARP_TILE_M = N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::WARP_ROW_TENSORS;
//     constexpr int WARP_TILE_N = N2M4TilingConfig::MMA_N_SP * N2M4TilingConfig::WARP_COL_TENSORS;

//     const int warpRowStart = warpIdxM * WARP_TILE_M;
//     const int warpColStart = warpIdxN * WARP_TILE_N;

//     // 由于本 warp 只读写自己区域，block 级 syncthreads 不再必须。
//     // 只需要保证本 warp 之前对 shared 的写入已完成即可：
//     __syncwarp();

//     // --------- 编译期按 WARP_TILE_N 分派向量宽度（写回以列方向向量化） ----------
//     if constexpr (WARP_TILE_N == 16 || WARP_TILE_N == 32 || WARP_TILE_N == 64 || WARP_TILE_N == 128) {
//         // 128-bit: 8 half = 16B
//         constexpr int ELEMS = 8;
//         static_assert((WARP_TILE_N % ELEMS) == 0, "WARP_TILE_N must be multiple of 8");

//         // 每条线程负责写回128bit=8个fp16，每个 warp 32条线程一次可以写回 16*16 个数据，以 16*16 大小为一个单元进行写回
//         constexpr int iterater_M = WARP_TILE_M / 16;
//         constexpr int iterater_N = WARP_TILE_N / 16;

//         // 32 条线程按逻辑划分为 16*2 的二维方式，每条线程负责写回 8 个数据
//         int row = lane / 2;  // 0..15
//         int col = lane % 2;  // 0..1

//         #pragma unroll
//         for (int r = 0; r<iterater_M; r++){
//             #pragma unroll
//             for (int c = 0; c<iterater_N; c++){
//                 half* dst = BlockGlobalPTR + (warpRowStart * N_Global + warpColStart) + (r * 16 * N_Global + c * 16) + row * N_Global + col * ELEMS;
//                 half* src = &(*(smem_CFrag + warpRowStart + r * 16 + row))[warpColStart + c * 16 + col * ELEMS];
//                 // 纯 128-bit 搬运
//                 *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
//             }
//         }
//     }else if constexpr (WARP_TILE_N == 8){
//         // 64-bit: 4 half
//         constexpr int ELEMS = 4;

//         // 每条线程负责写回 64bit=4个fp16，每个 warp 32条线程一次可以写回 16*8 个数据，以 16*8 大小为一个单元进行写回
//         constexpr int iterater_M = WARP_TILE_M / 16;
//         constexpr int iterater_N = WARP_TILE_N / 8;

//         // 32 条线程按逻辑划分为 16*2 的二维方式，每条线程负责写回 8 个数据
//         int row = lane / 2;  // 0..15
//         int col = lane % 2;  // 0..1

//         #pragma unroll
//         for (int r = 0; r<iterater_M; r++){
//             #pragma unroll
//             for (int c = 0; c<iterater_N; c++){
//                 half* dst = BlockGlobalPTR + (warpRowStart * N_Global + warpColStart) + (r * 16 * N_Global + c * 8) + row * N_Global + col * ELEMS;
//                 half* src = &(*(smem_CFrag + warpRowStart + r * 16 + row))[warpColStart + c * 8 + col * ELEMS];
//                 // 纯 128-bit 搬运
//                 *reinterpret_cast<uint64_t*>(dst) = *reinterpret_cast<const uint64_t*>(src);
//             }
//         }
//     }
// }




// 将 shared memory 结果写回 Global Memory
// 在编译时根据 TileingConfig 中 WARP_TILE_N 的值分派不同的实现，以适配不同的向量化宽度
// -------------------- 实现体：默认直接报错 --------------------
template<int WARP_TILE_N>
struct StoreImpl {
    template<typename N2M4TilingConfig>
    __device__ __forceinline__
    static void run(half (* /*smem_CFrag*/)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C],
                    half* /*BlockGlobalPTR*/, int /*N_Global*/) {
        static_assert(WARP_TILE_N == 8 || WARP_TILE_N == 16 || WARP_TILE_N == 32 ||
                      WARP_TILE_N == 64 || WARP_TILE_N == 128,
                      "Unsupported WARP_TILE_N");
    }
};

// -------------------- WARP_TILE_N == 8：64-bit --------------------
template<>
struct StoreImpl<8> {
    template<typename N2M4TilingConfig>
    __device__ __forceinline__
    static void run(half (*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C],
                    half* __restrict__ BlockGlobalPTR,
                    int N_Global)
    {
        const int warpId = threadIdx.x / WARP_SIZE;
        const int lane   = threadIdx.x % WARP_SIZE;

        const int warpIdxM = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;
        const int warpIdxN = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;

        const int WARP_TILE_M = N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::WARP_ROW_TENSORS;
        const int WARP_TILE_N = 8;

        const int warpRowStart = warpIdxM * WARP_TILE_M;
        const int warpColStart = warpIdxN * WARP_TILE_N;

        __syncwarp();

        const int ELEMS = 4; // 4 half = 64-bit

        const int iterater_M = WARP_TILE_M / 16;
        const int iterater_N = WARP_TILE_N / 8;

        int row = lane / 2;  // 0..15
        int col = lane % 2;  // 0..1

        #pragma unroll
        for (int r = 0; r < iterater_M; r++) {
            #pragma unroll
            for (int c = 0; c < iterater_N; c++) {
                half* dst = BlockGlobalPTR
                    + (warpRowStart * N_Global + warpColStart)
                    + (r * 16 * N_Global + c * 8)
                    + row * N_Global + col * ELEMS;

                half* src = &(*(smem_CFrag + warpRowStart + r * 16 + row))
                    [warpColStart + c * 8 + col * ELEMS];

                *reinterpret_cast<uint64_t*>(dst) =
                    *reinterpret_cast<const uint64_t*>(src);
            }
        }
    }
};

// -------------------- WARP_TILE_N ∈ {16,32,64,128}：128-bit --------------------
template<int WARP_TILE_N>
struct StoreImplVec128 {
    template<typename N2M4TilingConfig>
    __device__ __forceinline__
    static void run(half (*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C],
                    half* __restrict__ BlockGlobalPTR,
                    int N_Global)
    {
        const int warpId = threadIdx.x / WARP_SIZE;
        const int lane   = threadIdx.x % WARP_SIZE;

        const int warpIdxM = warpId / N2M4TilingConfig::BLOCK_COL_WARPS;
        const int warpIdxN = warpId % N2M4TilingConfig::BLOCK_COL_WARPS;

        const int WARP_TILE_M = N2M4TilingConfig::MMA_M_SP * N2M4TilingConfig::WARP_ROW_TENSORS;

        const int warpRowStart = warpIdxM * WARP_TILE_M;
        const int warpColStart = warpIdxN * WARP_TILE_N;

        __syncwarp();

        const int ELEMS = 8; // 8 half = 128-bit
        // 编译期检查（仍然是常量表达式）
        static_assert((WARP_TILE_N % ELEMS) == 0, "WARP_TILE_N must be multiple of 8");

        const int iterater_M = WARP_TILE_M / 16;
        const int iterater_N = WARP_TILE_N / 16;

        int row = lane / 2;  // 0..15
        int col = lane % 2;  // 0..1

        #pragma unroll
        for (int r = 0; r < iterater_M; r++) {
            #pragma unroll
            for (int c = 0; c < iterater_N; c++) {
                half* dst = BlockGlobalPTR
                    + (warpRowStart * N_Global + warpColStart)
                    + (r * 16 * N_Global + c * 16)
                    + row * N_Global + col * ELEMS;

                half* src = &(*(smem_CFrag + warpRowStart + r * 16 + row))
                    [warpColStart + c * 16 + col * ELEMS];

                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(src);
            }
        }
    }
};

// 把 16/32/64/128 映射到同一个 Vec128 实现
template<> struct StoreImpl<16> : StoreImplVec128<16> {};
template<> struct StoreImpl<32> : StoreImplVec128<32> {};
template<> struct StoreImpl<64> : StoreImplVec128<64> {};
template<> struct StoreImpl<128>: StoreImplVec128<128>{};

// -------------------- 你的原函数：只做一次编译期分派 --------------------
template<typename N2M4TilingConfig>
__device__ __forceinline__ void
StoreToGlobalMemoryFromShared(
    half (*smem_CFrag)[N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C],
    half* __restrict__ BlockGlobalPTR,
    int N_Global)
{
    // 编译期常量
    const int WARP_TILE_N =
        N2M4TilingConfig::MMA_N_SP * N2M4TilingConfig::WARP_COL_TENSORS;

    // 这里是编译期选择：只会实例化匹配的 StoreImpl<WARP_TILE_N>
    StoreImpl<WARP_TILE_N>::template run<N2M4TilingConfig>(smem_CFrag, BlockGlobalPTR, N_Global);
}





// 针对 MMA_SP_K=16 的版本
/* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */
/* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */
/* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */ /* 将数据从 shared memory 中加载到 reg */


// 矩阵 b 在加载时要求需要列主序的方式进行排布
// 在global memory时矩阵的排布是 Tile_K * Tile_N 以行主序排列，但是实际需要的是列主序, 所以这里需要使用 .trans
// 这里一次性加载 Wrap_N * MMA_K 大小的数据，循环每次加载 MMA_N * MMA_K
template<int NumOfTensors>
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters_K16(    uint32_t __restrict__ Registers[][2],
                                                                        half* __restrict__ sharedMem_PTR,  // 这是当前 Tile 对应的起始地址
                                                                        int warp_start_N,  // 表示当前 N 方向的 warp 的起始列号
                                                                        int k_step_offset  // 表示当前在 K 维度计算到第几个 MMA_K,在加载到shared memory时64 K维度为一个单元，4个MMA_K
                                                                    )
{
    // 计算参与加载的每条线程负责的 shared memory 地址
    int lane_id = threadIdx.x % 32;  // 计算 warp 内的线程号

    // 在存储到 shared memory 时对数据进行了映射，此处做同样的映射
    // 在逻辑上以 4*8 的二维方式组织
    int col     = lane_id % 8;            // 0..7  逻辑列映射
    int row    = lane_id / 8 +  k_step_offset * 2;  // 0..3  逻辑行映射，在存储时第一个 MMA_K 的时候是 0~3，第二个MMA_K是4~7

    // 与 aysc 存储时相同的 XOR 置换
    int read_col = col ^ row;

    // 找到当前 warp 对应的shared memory起始地址
    // 对于矩阵 B，从global memory加载到shared memory中时对于 N 维需要确保是 8 的倍数，并且存储时是以 64*8 为单元存储的
    sharedMem_PTR += warp_start_N / 8 * 8 * 64;
    // 计算在当前 warp 中每条线程负责的地址，aysc 时线程按照 4*8 的方式进行排布，此处地址需要做同样的对应
    // read_col * 8 表示每条线程读取8个连续数据， row * 64 表示一行64个数据
    sharedMem_PTR += row * 64 + read_col * 8;  

    // 把标准的C++指针转换为以共享内存地址为基址的字节数偏移量
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sharedMem_PTR));
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {  // 加载 Wrap_N * MMA_K 大小的数据时将 Wrap_N 维度进行拆分， NumOfTensors = Wrap_N / MMA_N = Wrap_col_Tensor
        // 32条线程一次性加载 4 个 8*8 大小的矩阵
        // 这条命令的意思是每条线程读取各自对应 smem_local_ptr 地址下的连续 128 bit，并且存放到 4 个 32 bit的寄存器中
        // 由于一开始是行主序，此处需要添加 .trans
        asm volatile(   "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(Registers[i][0]), "=r"(Registers[i][1])
                        : "r"(smem_local_ptr)
                        : "memory");
        
        // 计算下次读取的地址，因为在进行存储时以 64*8 为单元，所以此处按照 64*8 进行偏移
        smem_local_ptr += 64 * 8 * sizeof(half);  // sizeof(half) 表示的是字节数
    } 
}


template<int NumOfTensors>
__device__ __forceinline__ void A_FragLoadFromSharedToRegisters_K16(uint32_t __restrict__ Registers[][2],
                                                                    half* __restrict__ sharedMem_PTR,
                                                                    int warp_start_M,
                                                                    int k_step_offset = 0)
{
    // 计算参与加载的每条线程负责的 shared memory 地址
    int lane_id = threadIdx.x % 32;

    // 在存储到 shared memory 时对数据进行了映射，此处做同样的映射
    // 在逻辑上以 4*8 的二维方式组织
    int col     = lane_id % 8;            // 0..7  逻辑列映射
    int row    = lane_id / 8 +  k_step_offset * 2;  // 0..3  逻辑行映射，在存储时第一个 MMA_K 的时候是 0~3，第二个MMA_K是4~7

    // 与 aysc 存储时相同的 XOR 置换
    int read_col = col ^ row;

    // 找到当前 warp 对应的shared memory起始地址
    // 对于矩阵 B，从global memory加载到shared memory中时对于 M 维需要确保是 16 的倍数，并且存储时是以 16*32 为单元存储的
    sharedMem_PTR += warp_start_M / 16 * 16 * 32;
    // 计算在当前 warp 中每条线程负责的地址，aysc 时线程按照 4*8 的方式进行排布，此处地址需要做同样的对应
    // read_col * 8 表示每条线程读取8个连续数据， row * 64 表示一行64个数据
    sharedMem_PTR += row * 64 + read_col * 8;  

    // 把标准的C++指针转换为以共享内存地址为基址的字节数偏移量
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sharedMem_PTR));

    //  这里的 NumOfTensors 是 WARP_ROW_TENSORS 
    #pragma unroll
    for (int i = 0; i<NumOfTensors; i++){
        // M16N8K32 的shape需要加载 16*32 个数据，压缩之后只需要 16*16 个数据，对应 4 个 8*8 的矩阵
        // 这条命令的意思是每条线程读取各自对应 smem_local_ptr 地址下的连续 8*8*16*4/32=128 bit，并且存放到四个 32 bit的寄存器中
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1])
                     : "r"(smem_local_ptr));
        
        // 计算下次读取的地址，因为在进行存储时以 16*32 为单元，所以此处按照 16*32 进行偏移
        smem_local_ptr += 16 * 32 * sizeof(half);  // sizeof(half) 表示的是字节数
    }
}





















