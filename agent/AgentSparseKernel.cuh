#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int PADDING_SHARED_MEM_FOR_C = 8;
constexpr int REG_PER_C_TENSOR_16_8 = 4;
constexpr int WARP_SIZE = 32;
constexpr int HALF_PER_128bit = 8;

// CUDA 12.2 ptxas rejects the current K16 sparse MMA inline PTX syntax in this file.
// Keep the K32 path enabled so the agent build can still target shapes such as N=1024.
#ifndef AGENT_ENABLE_SPARSE_K16_PATHS
#define AGENT_ENABLE_SPARSE_K16_PATHS 0
#endif

template <int _K,
          int _BLOCK_ROW_WARPS,
          int _BLOCK_COL_WARPS,
          int _WARP_COL_TENSORS,
          int _WARP_ROW_TENSORS = 1,
          int _BLOCK_K_STEPS = 4>
struct N2M4TilingConfig {
    static_assert(_K == 16 || _K == 32, "AgentSparseKernel supports sparse MMA K16 and K32 paths only");

    static constexpr int MMA_M_SP = 16;
    static constexpr int MMA_N_SP = 8;
    static constexpr int MMA_K_SP = _K;

    static constexpr int BLOCK_ROW_WARPS = _BLOCK_ROW_WARPS;
    static constexpr int BLOCK_COL_WARPS = _BLOCK_COL_WARPS;
    static constexpr int WARP_COL_TENSORS = _WARP_COL_TENSORS;
    static constexpr int WARP_ROW_TENSORS = _WARP_ROW_TENSORS;
    static constexpr int BLOCK_K_STEPS = _BLOCK_K_STEPS;

    static constexpr int BLOCK_WARPS = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
    static constexpr int TILE_M = MMA_M_SP * WARP_ROW_TENSORS * BLOCK_ROW_WARPS;
    static constexpr int TILE_N = MMA_N_SP * WARP_COL_TENSORS * BLOCK_COL_WARPS;
    static constexpr int TILE_K = MMA_K_SP * BLOCK_K_STEPS;
    static constexpr int FRAGMENT_REGS = MMA_K_SP / 8;
    static constexpr int METADATA_GROUPS = TILE_K / 64;

    static_assert((TILE_K % 64) == 0, "Sparse metadata path expects TILE_K to be a multiple of 64");
};

using N2M4ConfigN128Wide = N2M4TilingConfig<16, 4, 2, 8, 2, 4>;

template <int SizeInBytes, typename T>
__device__ __forceinline__ void AgentCpAsync(T* smem_ptr, const T* global_ptr, bool pred_guard = true) {
    static_assert(SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16, "Unsupported cp.async width");
    const unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n"
        :
        : "r"(static_cast<int>(pred_guard)), "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
}

__device__ __forceinline__ void AgentCpAsyncCommit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void AgentCpAsyncWait() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
}

template <int NumNToCopy, typename Config>
__device__ __forceinline__ void AgentAsyncCopyPackedB(half* shared_mem_ptr,
                                                      const half* global_mem_ptr,
                                                      int global_stride,
                                                      bool pred = true) {
    static_assert(NumNToCopy % 8 == 0, "Packed B copy expects N to be a multiple of 8");

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = lane_id % 8;
    const int row0 = lane_id / 8;
    const int row1 = row0 + 4;
    const int store_col0 = col ^ row0;
    const int store_col1 = col ^ row1;

    const int copy_units = NumNToCopy / 8;
    const int max_iterations = (copy_units + Config::BLOCK_WARPS - 1) / Config::BLOCK_WARPS;

    #pragma unroll
    for (int iter = 0; iter < max_iterations; ++iter) {
        const int unit_idx = iter * Config::BLOCK_WARPS + warp_id;
        const bool valid = pred && (unit_idx < copy_units);
        const half* src = global_mem_ptr + unit_idx * 8 * global_stride;
        half* dst = shared_mem_ptr + unit_idx * 8 * 64;

        AgentCpAsync<16>(dst + store_col0 * 8 + row0 * 64, src + lane_id * 8, valid);
        AgentCpAsync<16>(dst + store_col1 * 8 + row1 * 64, src + (lane_id + 32) * 8, valid);
    }
}

template <int NumMToCopy, typename Config>
__device__ __forceinline__ void AgentAsyncCopyCompressedA(half* shared_mem_ptr,
                                                          const half* global_mem_ptr,
                                                          int global_stride,
                                                          bool pred = true) {
    static_assert(NumMToCopy % 16 == 0, "Compressed A copy expects M to be a multiple of 16");

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = lane_id % 8;
    const int row0 = lane_id / 8;
    const int row1 = row0 + 4;
    const int store_col0 = col ^ row0;
    const int store_col1 = col ^ row1;

    const int copy_units = NumMToCopy / 16;
    const int max_iterations = (copy_units + Config::BLOCK_WARPS - 1) / Config::BLOCK_WARPS;

    #pragma unroll
    for (int iter = 0; iter < max_iterations; ++iter) {
        const int unit_idx = iter * Config::BLOCK_WARPS + warp_id;
        const bool valid = pred && (unit_idx < copy_units);
        const half* src = global_mem_ptr + unit_idx * 16 * global_stride;
        half* dst = shared_mem_ptr + unit_idx * 16 * 32;

        AgentCpAsync<16>(dst + store_col0 * 8 + row0 * 64, src + lane_id * 8, valid);
        AgentCpAsync<16>(dst + store_col1 * 8 + row1 * 64, src + (lane_id + 32) * 8, valid);
    }
}

template <int NumMToCopy, typename Config>
__device__ __forceinline__ void AgentAsyncCopyMetadata(uint16_t* shared_mem_ptr,
                                                       const uint16_t* global_mem_ptr,
                                                       int global_stride,
                                                       bool pred = true) {
    static_assert(NumMToCopy % 16 == 0, "Metadata copy expects M to be a multiple of 16");

    const int group_id = threadIdx.x / 8;
    const int lane_id = threadIdx.x % 8;
    const int copy_units = NumMToCopy / 16;
    const bool valid = pred && (group_id < copy_units);

    const uint16_t* src = global_mem_ptr + group_id * global_stride;
    uint16_t* dst = shared_mem_ptr + group_id * 64 * Config::METADATA_GROUPS;

    #pragma unroll
    for (int metadata_group = 0; metadata_group < Config::METADATA_GROUPS; ++metadata_group) {
        AgentCpAsync<16>(dst + metadata_group * 64 + lane_id * 8,
                         src + metadata_group * 64 + lane_id * 8,
                         valid);
    }
}

template <int NumTensors>
__device__ __forceinline__ void AgentLoadBFragmentsK16(uint32_t regs[][2],
                                                       half* shared_mem_ptr,
                                                       int warp_start_n,
                                                       int k_step_offset) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = lane_id % 8;
    const int row = lane_id / 8 + k_step_offset * 2;
    const int read_col = col ^ row;

    shared_mem_ptr += (warp_start_n / 8) * 8 * 64;
    shared_mem_ptr += row * 64 + read_col * 8;
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_mem_ptr));

    #pragma unroll
    for (int tensor = 0; tensor < NumTensors; ++tensor) {
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(regs[tensor][0]), "=r"(regs[tensor][1])
                     : "r"(smem_ptr)
                     : "memory");
        smem_ptr += 64 * 8 * sizeof(half);
    }
}

template <int NumTensors>
__device__ __forceinline__ void AgentLoadAFragmentsK16(uint32_t regs[][2],
                                                       half* shared_mem_ptr,
                                                       int warp_start_m,
                                                       int k_step_offset) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = lane_id % 8;
    const int row = lane_id / 8 + k_step_offset * 2;
    const int read_col = col ^ row;

    shared_mem_ptr += (warp_start_m / 16) * 16 * 32;
    shared_mem_ptr += row * 64 + read_col * 8;
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_mem_ptr));

    #pragma unroll
    for (int tensor = 0; tensor < NumTensors; ++tensor) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(regs[tensor][0]), "=r"(regs[tensor][1])
                     : "r"(smem_ptr)
                     : "memory");
        smem_ptr += 16 * 32 * sizeof(half);
    }
}

template <int NumTensors>
__device__ __forceinline__ void AgentLoadBFragmentsK32(uint32_t regs[][4],
                                                       half* shared_mem_ptr,
                                                       int warp_start_n,
                                                       int k_step_offset) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = lane_id % 8;
    const int row = lane_id / 8 + k_step_offset * 4;
    const int read_col = col ^ row;

    shared_mem_ptr += (warp_start_n / 8) * 8 * 64;
    shared_mem_ptr += row * 64 + read_col * 8;
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_mem_ptr));

    #pragma unroll
    for (int tensor = 0; tensor < NumTensors; ++tensor) {
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(regs[tensor][0]), "=r"(regs[tensor][1]), "=r"(regs[tensor][2]), "=r"(regs[tensor][3])
                     : "r"(smem_ptr)
                     : "memory");
        smem_ptr += 64 * 8 * sizeof(half);
    }
}

template <int NumTensors>
__device__ __forceinline__ void AgentLoadAFragmentsK32(uint32_t regs[][4],
                                                       half* shared_mem_ptr,
                                                       int warp_start_m,
                                                       int k_step_offset) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = lane_id % 8;
    const int row = lane_id / 8 + k_step_offset * 4;
    const int read_col = col ^ row;

    shared_mem_ptr += (warp_start_m / 16) * 16 * 32;
    shared_mem_ptr += row * 64 + read_col * 8;
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_mem_ptr));

    #pragma unroll
    for (int tensor = 0; tensor < NumTensors; ++tensor) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(regs[tensor][0]), "=r"(regs[tensor][1]), "=r"(regs[tensor][2]), "=r"(regs[tensor][3])
                     : "r"(smem_ptr)
                     : "memory");
        smem_ptr += 16 * 32 * sizeof(half);
    }
}

template <int NumTensors, typename Config>
__device__ __forceinline__ void AgentLoadMetadata(uint32_t regs[][Config::METADATA_GROUPS],
                                                  uint16_t* shared_mem_ptr,
                                                  int warp_start_m) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    shared_mem_ptr += (warp_start_m / 16) * Config::TILE_K;
    shared_mem_ptr += (lane_id & 7) * 8;
    uint32_t tensor_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_mem_ptr));

    #pragma unroll
    for (int tensor = 0; tensor < NumTensors; ++tensor) {
        uint32_t group_smem_ptr = tensor_smem_ptr;
        #pragma unroll
        for (int metadata_group = 0; metadata_group < Config::METADATA_GROUPS; ++metadata_group) {
            asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                         : "=r"(regs[tensor][metadata_group])
                         : "r"(group_smem_ptr)
                         : "memory");
            group_smem_ptr += 64 * sizeof(uint16_t);
        }
        tensor_smem_ptr += Config::TILE_K * sizeof(uint16_t);
    }
}

#if AGENT_ENABLE_SPARSE_K16_PATHS
__device__ __forceinline__ void AgentSparseMmaK16_0(uint32_t c[],
                                                    const uint32_t a[],
                                                    const uint32_t b[],
                                                    uint32_t metadata) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(metadata));
}

__device__ __forceinline__ void AgentSparseMmaK16_1(uint32_t c[],
                                                    const uint32_t a[],
                                                    const uint32_t b[],
                                                    uint32_t metadata) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x1;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(metadata));
}

__device__ __forceinline__ void AgentSparseMmaK16_2(uint32_t c[],
                                                    const uint32_t a[],
                                                    const uint32_t b[],
                                                    uint32_t metadata) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x2;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(metadata));
}

__device__ __forceinline__ void AgentSparseMmaK16_3(uint32_t c[],
                                                    const uint32_t a[],
                                                    const uint32_t b[],
                                                    uint32_t metadata) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x3;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(metadata));
}
#endif

__device__ __forceinline__ void AgentSparseMmaK32_0(uint32_t c[],
                                                    const uint32_t a[],
                                                    const uint32_t b[],
                                                    uint32_t metadata) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, %1, %2, %3}, %12, 0x0;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(metadata));
}

__device__ __forceinline__ void AgentSparseMmaK32_1(uint32_t c[],
                                                    const uint32_t a[],
                                                    const uint32_t b[],
                                                    uint32_t metadata) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, %1, %2, %3}, %12, 0x1;\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(metadata));
}

template <int SparseK>
struct AgentTensorCoreLoopImpl;

#if AGENT_ENABLE_SPARSE_K16_PATHS
template <>
struct AgentTensorCoreLoopImpl<16> {
    template <typename Config>
    __device__ __forceinline__ static void run(
        float accum[][REG_PER_C_TENSOR_16_8],
        uint32_t a_regs[Config::WARP_ROW_TENSORS * 2][2],
        uint32_t b_regs[Config::WARP_COL_TENSORS * 2][2],
        uint32_t metadata_regs[][Config::METADATA_GROUPS],
        half* shared_a,
        half* shared_b,
        int warp_start_row,
        int warp_start_col) {
        auto accum_u32 = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_8]>(accum);

        AgentLoadBFragmentsK16<Config::WARP_COL_TENSORS>(b_regs, shared_b, warp_start_col, 0);
        AgentLoadAFragmentsK16<Config::WARP_ROW_TENSORS>(a_regs, shared_a, warp_start_row, 0);

        #pragma unroll
        for (int k = 0; k < Config::BLOCK_K_STEPS; ++k) {
            uint32_t(*b_read)[2] = b_regs + (k % 2) * Config::WARP_COL_TENSORS;
            uint32_t(*b_write)[2] = b_regs + ((k + 1) % 2) * Config::WARP_COL_TENSORS;
            uint32_t(*a_read)[2] = a_regs + (k % 2) * Config::WARP_ROW_TENSORS;
            uint32_t(*a_write)[2] = a_regs + ((k + 1) % 2) * Config::WARP_ROW_TENSORS;

            if (k + 1 < Config::BLOCK_K_STEPS) {
                AgentLoadBFragmentsK16<Config::WARP_COL_TENSORS>(b_write, shared_b, warp_start_col, k + 1);
                AgentLoadAFragmentsK16<Config::WARP_ROW_TENSORS>(a_write, shared_a, warp_start_row, k + 1);
            }

            #pragma unroll
            for (int i = 0; i < Config::WARP_ROW_TENSORS; ++i) {
                #pragma unroll
                for (int j = 0; j < Config::WARP_COL_TENSORS; ++j) {
                    uint32_t* c_frag = accum_u32[i * Config::WARP_COL_TENSORS + j];
                    const uint32_t metadata = metadata_regs[i][k / 4];
                    if ((k & 3) == 0) {
                        AgentSparseMmaK16_0(c_frag, a_read[i], b_read[j], metadata);
                    } else if ((k & 3) == 1) {
                        AgentSparseMmaK16_1(c_frag, a_read[i], b_read[j], metadata);
                    } else if ((k & 3) == 2) {
                        AgentSparseMmaK16_2(c_frag, a_read[i], b_read[j], metadata);
                    } else {
                        AgentSparseMmaK16_3(c_frag, a_read[i], b_read[j], metadata);
                    }
                }
            }
        }
    }
};
#endif

template <>
struct AgentTensorCoreLoopImpl<32> {
    template <typename Config>
    __device__ __forceinline__ static void run(
        float accum[][REG_PER_C_TENSOR_16_8],
        uint32_t a_regs[Config::WARP_ROW_TENSORS * 2][4],
        uint32_t b_regs[Config::WARP_COL_TENSORS * 2][4],
        uint32_t metadata_regs[][Config::METADATA_GROUPS],
        half* shared_a,
        half* shared_b,
        int warp_start_row,
        int warp_start_col) {
        auto accum_u32 = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_8]>(accum);

        AgentLoadBFragmentsK32<Config::WARP_COL_TENSORS>(b_regs, shared_b, warp_start_col, 0);
        AgentLoadAFragmentsK32<Config::WARP_ROW_TENSORS>(a_regs, shared_a, warp_start_row, 0);

        #pragma unroll
        for (int k = 0; k < Config::BLOCK_K_STEPS; ++k) {
            uint32_t(*b_read)[4] = b_regs + (k % 2) * Config::WARP_COL_TENSORS;
            uint32_t(*b_write)[4] = b_regs + ((k + 1) % 2) * Config::WARP_COL_TENSORS;
            uint32_t(*a_read)[4] = a_regs + (k % 2) * Config::WARP_ROW_TENSORS;
            uint32_t(*a_write)[4] = a_regs + ((k + 1) % 2) * Config::WARP_ROW_TENSORS;

            if (k + 1 < Config::BLOCK_K_STEPS) {
                AgentLoadBFragmentsK32<Config::WARP_COL_TENSORS>(b_write, shared_b, warp_start_col, k + 1);
                AgentLoadAFragmentsK32<Config::WARP_ROW_TENSORS>(a_write, shared_a, warp_start_row, k + 1);
            }

            #pragma unroll
            for (int i = 0; i < Config::WARP_ROW_TENSORS; ++i) {
                #pragma unroll
                for (int j = 0; j < Config::WARP_COL_TENSORS; ++j) {
                    uint32_t* c_frag = accum_u32[i * Config::WARP_COL_TENSORS + j];
                    const uint32_t metadata = metadata_regs[i][k / 2];
                    if ((k & 1) == 0) {
                        AgentSparseMmaK32_0(c_frag, a_read[i], b_read[j], metadata);
                    } else {
                        AgentSparseMmaK32_1(c_frag, a_read[i], b_read[j], metadata);
                    }
                }
            }
        }
    }
};

template <typename Config>
__device__ __forceinline__ void AgentStoreAccumulatorToShared(
    half (*shared_c)[Config::TILE_N + PADDING_SHARED_MEM_FOR_C],
    float accum[][REG_PER_C_TENSOR_16_8]) {
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    const int warp_m = warp_id / Config::BLOCK_COL_WARPS;
    const int warp_n = warp_id % Config::BLOCK_COL_WARPS;
    const int warp_row_base = warp_m * (Config::MMA_M_SP * Config::WARP_ROW_TENSORS);
    const int warp_col_base = warp_n * (Config::MMA_N_SP * Config::WARP_COL_TENSORS);
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col_base = (lane_id % 4) * 2;
    const int row0 = lane_id / 4;
    const int row1 = row0 + 8;

    #pragma unroll
    for (int i = 0; i < Config::WARP_ROW_TENSORS; ++i) {
        #pragma unroll
        for (int j = 0; j < Config::WARP_COL_TENSORS; ++j) {
            const int frag_idx = i * Config::WARP_COL_TENSORS + j;
            const int tensor_row = warp_row_base + i * Config::MMA_M_SP;
            const int tensor_col = warp_col_base + j * Config::MMA_N_SP;

            *reinterpret_cast<half2*>(&shared_c[tensor_row + row0][tensor_col + col_base]) =
                __halves2half2(__float2half_rn(accum[frag_idx][0]), __float2half_rn(accum[frag_idx][1]));
            *reinterpret_cast<half2*>(&shared_c[tensor_row + row1][tensor_col + col_base]) =
                __halves2half2(__float2half_rn(accum[frag_idx][2]), __float2half_rn(accum[frag_idx][3]));
        }
    }
}

template <int WarpTileN>
struct AgentSharedStore;

template <>
struct AgentSharedStore<8> {
    template <typename Config>
    __device__ __forceinline__ static void run(
        half (*shared_c)[Config::TILE_N + PADDING_SHARED_MEM_FOR_C],
        half* global_ptr,
        int global_n) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp_m = warp_id / Config::BLOCK_COL_WARPS;
        const int warp_n = warp_id % Config::BLOCK_COL_WARPS;
        const int warp_tile_m = Config::MMA_M_SP * Config::WARP_ROW_TENSORS;
        const int row_base = warp_m * warp_tile_m;
        const int col_base = warp_n * 8;
        const int row = lane / 2;
        const int col_group = lane % 2;

        __syncwarp();

        #pragma unroll
        for (int tile_m = 0; tile_m < warp_tile_m / 16; ++tile_m) {
            half* dst = global_ptr + (row_base + tile_m * 16 + row) * global_n + col_base + col_group * 4;
            half* src = &shared_c[row_base + tile_m * 16 + row][col_base + col_group * 4];
            *reinterpret_cast<uint64_t*>(dst) = *reinterpret_cast<const uint64_t*>(src);
        }
    }
};

template <int WarpTileN>
struct AgentSharedStoreVec128 {
    template <typename Config>
    __device__ __forceinline__ static void run(
        half (*shared_c)[Config::TILE_N + PADDING_SHARED_MEM_FOR_C],
        half* global_ptr,
        int global_n) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp_m = warp_id / Config::BLOCK_COL_WARPS;
        const int warp_n = warp_id % Config::BLOCK_COL_WARPS;
        const int warp_tile_m = Config::MMA_M_SP * Config::WARP_ROW_TENSORS;
        const int row_base = warp_m * warp_tile_m;
        const int col_base = warp_n * WarpTileN;
        const int row = lane / 2;
        const int col_group = lane % 2;

        __syncwarp();

        #pragma unroll
        for (int tile_m = 0; tile_m < warp_tile_m / 16; ++tile_m) {
            #pragma unroll
            for (int tile_n = 0; tile_n < WarpTileN / 16; ++tile_n) {
                half* dst = global_ptr + (row_base + tile_m * 16 + row) * global_n + col_base + tile_n * 16 + col_group * 8;
                half* src = &shared_c[row_base + tile_m * 16 + row][col_base + tile_n * 16 + col_group * 8];
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            }
        }
    }
};

template <>
struct AgentSharedStore<16> : AgentSharedStoreVec128<16> {};
template <>
struct AgentSharedStore<32> : AgentSharedStoreVec128<32> {};
template <>
struct AgentSharedStore<64> : AgentSharedStoreVec128<64> {};
template <>
struct AgentSharedStore<128> : AgentSharedStoreVec128<128> {};

template <typename Config>
__device__ __forceinline__ void AgentStoreSharedToGlobal(
    half (*shared_c)[Config::TILE_N + PADDING_SHARED_MEM_FOR_C],
    half* global_ptr,
    int global_n) {
    AgentSharedStore<Config::MMA_N_SP * Config::WARP_COL_TENSORS>::template run<Config>(shared_c, global_ptr, global_n);
}

template <typename Config>
struct AgentOutputStorePolicy {
    static constexpr bool kUseDirectStore = false;

    __device__ __forceinline__ static void run(
        half* shared_mem,
        half* global_ptr,
        int global_n,
        float accum[][REG_PER_C_TENSOR_16_8]) {
        __syncthreads();
        auto shared_c = reinterpret_cast<half(*)[Config::TILE_N + PADDING_SHARED_MEM_FOR_C]>(shared_mem);
        AgentStoreAccumulatorToShared<Config>(shared_c, accum);
        AgentStoreSharedToGlobal<Config>(shared_c, global_ptr, global_n);
    }
};

template <typename Config, int stages>
__global__ void SpMM_N2M4_Kernel(const half* compressed_a,
                                 const uint16_t* metadata,
                                 const half* packed_b,
                                 half* c,
                                 int M_Global,
                                 int N_Global,
                                 int K_Global,
                                 int Split_K) {
    const int tiles_m = M_Global / Config::TILE_M;
    const int split_k_idx = blockIdx.y / tiles_m;
    const bool last_split = split_k_idx == (Split_K - 1);
    const int tile_idx_m = blockIdx.y % tiles_m;
    const int tile_idx_n = blockIdx.x;
    const int tile_row = tile_idx_m * Config::TILE_M;
    const int tile_col = tile_idx_n * Config::TILE_N;

    const int total_k_tiles = K_Global / Config::TILE_K;
    const int avg_k_tiles = (total_k_tiles - 1) / Split_K + 1;
    const int rounded_k_tiles = avg_k_tiles * Split_K;
    const int padding_k_tiles = rounded_k_tiles - total_k_tiles;
    const int iterate_k = last_split ? (avg_k_tiles - padding_k_tiles) : avg_k_tiles;

    constexpr int metadata_stages = (stages > 1) ? (stages - 1) : 1;
    constexpr int shared_a_stage_elems = Config::TILE_M * Config::TILE_K / 2;
    constexpr int shared_b_stage_elems = Config::TILE_K * Config::TILE_N;
    constexpr int shared_metadata_stage_elems = Config::TILE_M * Config::TILE_K / 16;

    extern __shared__ __align__(128) half shared_mem[];
    half* shared_a = shared_mem;
    half* shared_b = shared_a + shared_a_stage_elems * stages;
    uint16_t* shared_metadata =
        reinterpret_cast<uint16_t*>(shared_b + shared_b_stage_elems * stages);

    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    const int warp_m = warp_id / Config::BLOCK_COL_WARPS;
    const int warp_n = warp_id % Config::BLOCK_COL_WARPS;
    const int warp_row = warp_m * Config::WARP_ROW_TENSORS * Config::MMA_M_SP;
    const int warp_col = warp_n * Config::WARP_COL_TENSORS * Config::MMA_N_SP;

    uint32_t a_regs[Config::WARP_ROW_TENSORS * 2][Config::FRAGMENT_REGS];
    uint32_t b_regs[Config::WARP_COL_TENSORS * 2][Config::FRAGMENT_REGS];
    uint32_t metadata_regs[Config::WARP_ROW_TENSORS][Config::METADATA_GROUPS];

    const half* a_global = compressed_a + tile_row * K_Global / 2 +
                           split_k_idx * Config::MMA_M_SP * avg_k_tiles * Config::TILE_K / 2;
    const half* b_global = packed_b + tile_col * K_Global +
                           split_k_idx * avg_k_tiles * Config::TILE_K * 8;
    const uint16_t* metadata_global = metadata + tile_row * K_Global / 16 +
                                      split_k_idx * Config::MMA_M_SP * avg_k_tiles * Config::TILE_K / 16;

    const int prefetch_tiles = min(stages - 1, iterate_k);
    #pragma unroll
    for (int stage = 0; stage < stages - 1; ++stage) {
        if (stage < prefetch_tiles) {
            AgentAsyncCopyMetadata<Config::TILE_M, Config>(shared_metadata + stage * shared_metadata_stage_elems,
                                                           metadata_global,
                                                           K_Global);
            AgentAsyncCopyCompressedA<Config::TILE_M, Config>(shared_a + stage * shared_a_stage_elems,
                                                              a_global,
                                                              K_Global / 2);
            AgentAsyncCopyPackedB<Config::TILE_N, Config>(shared_b + stage * shared_b_stage_elems,
                                                          b_global,
                                                          K_Global);
            AgentCpAsyncCommit();

            a_global += Config::MMA_M_SP * Config::TILE_K / 2;
            b_global += Config::TILE_K * 8;
            metadata_global += Config::MMA_M_SP * Config::TILE_K / 16;
        }
    }

    float accum[Config::WARP_ROW_TENSORS * Config::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_8];
    #pragma unroll
    for (int frag = 0; frag < Config::WARP_ROW_TENSORS * Config::WARP_COL_TENSORS; ++frag) {
        #pragma unroll
        for (int reg = 0; reg < REG_PER_C_TENSOR_16_8; ++reg) {
            accum[frag][reg] = 0.0f;
        }
    }

    if (prefetch_tiles <= 1) {
        AgentCpAsyncWait<0>();
    } else {
        AgentCpAsyncWait<stages - 2>();
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < iterate_k; ++tile_k) {
        const int read_stage = tile_k % stages;
        const int write_stage = (tile_k + stages - 1) % stages;
        const int read_metadata_stage = tile_k % metadata_stages;
        const int write_metadata_stage = (tile_k + stages - 1) % metadata_stages;

        half* read_a = shared_a + read_stage * shared_a_stage_elems;
        half* read_b = shared_b + read_stage * shared_b_stage_elems;
        uint16_t* read_metadata = shared_metadata + read_metadata_stage * shared_metadata_stage_elems;
        half* write_a = shared_a + write_stage * shared_a_stage_elems;
        half* write_b = shared_b + write_stage * shared_b_stage_elems;
        uint16_t* write_metadata = shared_metadata + write_metadata_stage * shared_metadata_stage_elems;

        AgentLoadMetadata<Config::WARP_ROW_TENSORS, Config>(metadata_regs, read_metadata, warp_row);

        const bool issue_copy = (tile_k + stages - 1) < iterate_k;
        if (issue_copy) {
            AgentAsyncCopyMetadata<Config::TILE_M, Config>(write_metadata, metadata_global, K_Global);
            AgentAsyncCopyCompressedA<Config::TILE_M, Config>(write_a, a_global, K_Global / 2);
            AgentAsyncCopyPackedB<Config::TILE_N, Config>(write_b, b_global, K_Global);
            AgentCpAsyncCommit();

            a_global += Config::MMA_M_SP * Config::TILE_K / 2;
            b_global += Config::TILE_K * 8;
            metadata_global += Config::MMA_M_SP * Config::TILE_K / 16;
        }

        AgentTensorCoreLoopImpl<Config::MMA_K_SP>::template run<Config>(
            accum, a_regs, b_regs, metadata_regs, read_a, read_b, warp_row, warp_col);

        if (issue_copy) {
            AgentCpAsyncWait<stages - 2>();
            __syncthreads();
        }
    }

    half* global_c = c + split_k_idx * (M_Global * N_Global) + tile_row * N_Global + tile_col;
    AgentOutputStorePolicy<Config>::run(shared_mem, global_c, N_Global, accum);
}

__global__ void SplitK_Reduction(half* C, half* reduction_workspace, int M_Global, int N_Global, int Split_K) {
    const int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * HALF_PER_128bit;
    const int total_elements = M_Global * N_Global;
    if (base_idx >= total_elements) {
        return;
    }

    float2 accum[HALF_PER_128bit / 2];
    #pragma unroll
    for (int i = 0; i < HALF_PER_128bit / 2; ++i) {
        accum[i] = make_float2(0.0f, 0.0f);
    }

    for (int split = 0; split < Split_K; ++split) {
        const half2* src = reinterpret_cast<const half2*>(reduction_workspace + split * total_elements + base_idx);
        #pragma unroll
        for (int i = 0; i < HALF_PER_128bit / 2; ++i) {
            const float2 value = __half22float2(src[i]);
            accum[i].x += value.x;
            accum[i].y += value.y;
        }
    }

    half2* dst = reinterpret_cast<half2*>(C + base_idx);
    #pragma unroll
    for (int i = 0; i < HALF_PER_128bit / 2; ++i) {
        dst[i] = __floats2half2_rn(accum[i].x, accum[i].y);
    }
}
