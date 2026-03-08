#ifndef __SPARSE_MMA_PTR_CUH__
#define __SPARSE_MMA_PTR_CUH__


/* 这个文件只包含了 MMA.SP 相关的 PTX 指令封装 */

// 需要注意这条指令，在某些架构中有更好的性能
// mma.sp::ordered_metadata




#include <stdint.h> // for uint32_t

/*
 * @brief 使用 uint32_t 指针封装 PTX 指令 mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *
 * 该函数在支持稀疏张量核心的 GPU (Ampere 架构及以上) 上执行一个 16x8 的稀疏矩阵乘法累加操作。
 * 它直接操作以 32位无符号整数形式表示的寄存器数据。
 *
 * 运算: C = A * B + C (结果写回到 C)
 *
 * 矩阵维度:
 * - A: 16x16 (M x K), .f16, 行主序
 * - B: 16x8  (K x N), .f16, 列主序
 * - C: 16x8  (M x N), .f32
 *
 * 参数要求 (指针指向的数组大小):
 * - c: 必须指向一个包含 4 个 uint32_t 元素的数组。这些元素被视为 4 个 f32 寄存器，
 *      既作为输入累加器，也作为输出目标。
 * - a: 必须指向一个包含 2 个 uint32_t 元素的数组。每个元素代表一个 f16x2 寄存器。
 * - b: 必须指向一个包含 2 个 uint32_t 元素的数组。每个元素代表一个 f16x2 寄存器。
 * - metadata: 描述矩阵 A 的 2:4 稀疏结构的 32位元数据。
 *
 * @param c        [inout] 指向 C 矩阵分片寄存器的指针 (长度为4)。作为输入累加器，并被结果覆盖。
 * @param a        [in]    指向 A 矩阵分片寄存器的指针 (长度为2)。
 * @param b        [in]    指向 B 矩阵分片寄存器的指针 (长度为2)。
 * @param metadata [in]    32位的稀疏元数据。
 */
__device__ __forceinline__ void MMA_FP16_SP_M16N8K16_0x0(
    uint32_t __restrict__ c[],
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t metadata)
{
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%0, %1, %2, %3}, %8, 0x0;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "r"(metadata)
    );
}


__device__ __forceinline__ void MMA_FP16_SP_M16N8K16_0x1(
    uint32_t __restrict__ c[],
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t metadata)
{
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%0, %1, %2, %3}, %8, 0x1;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "r"(metadata)
    );
}


__device__ __forceinline__ void MMA_FP16_SP_M16N8K16_0x2(
    uint32_t __restrict__ c[],
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t metadata)
{
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%0, %1, %2, %3}, %8, 0x2;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "r"(metadata)
    );
}

__device__ __forceinline__ void MMA_FP16_SP_M16N8K16_0x3(
    uint32_t __restrict__ c[],
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t metadata)
{
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%0, %1, %2, %3}, %8, 0x3;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "r"(metadata)
    );
}


/*
 * @brief 使用 uint32_t 指针封装 PTX 指令 mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
 *
 * 该函数在支持稀疏张量核心的 GPU (Ampere 架构及以上) 上执行一个 16x8 的稀疏矩阵乘法累加操作。
 * 它直接操作以 32位无符号整数形式表示的寄存器数据。
 *
 * 运算: C = A * B + C (结果写回到 C)
 *
 * 矩阵维度:
 * - A: 16x32 (M x K), .f16, 行主序
 * - B: 32x8  (K x N), .f16, 列主序
 * - C: 16x8  (M x N), .f32
 *
 * 参数要求 (指针指向的数组大小):
 * - c: 必须指向一个包含 4 个 uint32_t 元素的数组。这些元素被视为 4 个 f32 寄存器，
 *      既作为输入累加器，也作为输出目标。
 * - a: 必须指向一个包含 4 个 uint32_t 元素的数组。每个元素代表一个 f16x2 寄存器。
 * - b: 必须指向一个包含 4 个 uint32_t 元素的数组。每个元素代表一个 f16x2 寄存器。
 * - metadata: 描述矩阵 A 的 2:4 稀疏结构的 32位元数据。
 *
 * @param c        [inout] 指向 C 矩阵分片寄存器的指针 (长度为4)。作为输入累加器，并被结果覆盖。
 * @param a        [in]    指向 A 矩阵分片寄存器的指针 (长度为4)。
 * @param b        [in]    指向 B 矩阵分片寄存器的指针 (长度为4)。
 * @param metadata [in]    32位的稀疏元数据。
 */
__device__ __forceinline__ void MMA_FP16_SP_M16N8K32_0x0(
    uint32_t __restrict__ c[],
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t metadata)
{
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%0, %1, %2, %3}, %12, 0x0;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(metadata)
    );
}

__device__ __forceinline__ void MMA_FP16_SP_M16N8K32_0x1(
    uint32_t __restrict__ c[],
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t metadata)
{
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%0, %1, %2, %3}, %12, 0x1;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(metadata)
    );
}

#endif // __SPARSE_MMA_PTR_CUH__