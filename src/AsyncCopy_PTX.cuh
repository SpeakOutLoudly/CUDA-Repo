#pragma once  // 仅编译一次

/* 这个文件包含进行异步读取的函数 */

// 目的：为 N:2M（2:4）稀疏内核提供基于 cp.async 的 A/B/metadata 异步加载封装，
//      以便在计算（mma.sp）与访存之间做双缓冲/流水隐藏延迟。

// 使用 cp.async 将全局内存搬运到共享内存（半精度版本）
// - SizeInBytes: 每个线程搬运的字节数（4/8/16，推荐 16 提升带宽利用）
// - pred_guard: 谓词屏蔽，越界时传 false 避免非法访存
template<int SizeInBytes>  // 模板参数：每线程搬运的字节数（4/8/16）
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)  // 半精度版本
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");  // 编译期校验
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);  // 将共享内存指针转换为共享地址空间整数
    asm volatile("{ \n"  // 发起异步拷贝的内联 PTX
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),  // 谓词寄存器：pred_guard 为真才执行
                 "r"(smem_int_ptr),             // 目标共享内存地址（整数）
                 "l"(global_ptr),               // 源全局内存地址（64位）
                 "n"(SizeInBytes));             // 编译期常量：拷贝字节数
}

template<int SizeInBytes>  // 模板参数：每线程搬运的字节数（4/8/16）
__device__ __forceinline__ void cp_async(uint16_t* smem_ptr, const uint16_t* global_ptr, bool pred_guard = true)  // uint16 版本
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");  // 编译期校验
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);  // 将共享内存指针转换为共享地址空间整数
    asm volatile("{ \n"  // 发起异步拷贝的内联 PTX
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),  // 谓词寄存器：pred_guard 为真才执行
                 "r"(smem_int_ptr),             // 目标共享内存地址（整数）
                 "l"(global_ptr),               // 源全局内存地址（64位）
                 "n"(SizeInBytes));             // 编译期常量：拷贝字节数
}



// 使用 cp.async 将全局内存搬运到共享内存（uint32 版本，用于 metadata）
template<int SizeInBytes>  // 模板参数：每线程搬运的字节数（4/8/16）
__device__ __forceinline__ void cp_async(uint32_t* smem_ptr, const uint32_t* global_ptr, bool pred_guard = true)  // uint32 版本
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");  // 编译期校验
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);  // 共享内存地址整数
    asm volatile("{ \n"  // 发起异步拷贝
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),  // 谓词
                 "r"(smem_int_ptr),             // 目的共享地址
                 "l"(global_ptr),               // 源全局地址
                 "n"(SizeInBytes));             // 拷贝字节数
}

// 使用 cp.async 将全局内存搬运到共享内存（uint64）
template<int SizeInBytes>  // 模板参数：每线程搬运的字节数（4/8/16）
__device__ __forceinline__ void cp_async(uint64_t* smem_ptr, const uint64_t* global_ptr, bool pred_guard = true)  // uint64 版本
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");  // 编译期校验
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);  // 共享内存地址整数
    asm volatile("{ \n"  // 发起异步拷贝
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),  // 谓词
                 "r"(smem_int_ptr),             // 目的共享地址
                 "l"(global_ptr),               // 源全局地址
                 "n"(SizeInBytes));             // 拷贝字节数
}


// 提交当前已发起的 cp.async 到同一组（不阻塞）
__device__ __forceinline__ void cp_async_group_commit()  // 提交当前组的 cp.async 指令（不阻塞）
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");  // PTX 提交组
}

// 阻塞直到除了最近 N 组外，其余组都已完成（N=0 等同等待所有已提交组完成）
template<int N>
__device__ __forceinline__ void cp_async_wait_group()  // 等待早期提交的组完成
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");  // PTX 等待组
}


