#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <cstdio>

#include "AgentSparseKernel.cuh"
#include "backends/cuSparseLt_backend.cuh"
#include "utils.h"

namespace {

struct AgentLaunchContext {
    bool initialized = false;
    bool config_printed = false;
    int sm = 0;
    int M = 0;
    int N = 0;
    int K = 0;
    const half* dense_a_device = nullptr;
    const half* dense_b_row_major_device = nullptr;
    CuSparseLtSpmmContext* fallback_ctx = nullptr;
};

AgentLaunchContext g_agent_ctx;

bool ShouldPrintConfig() {
    return GetEnvIntOrDefault("SPMM_PRINT_CONFIG", 1) != 0;
}

template <typename N2M4TilingConfig, int stages>
void PrintLaunchConfig(const char* mode, const char* label, int N, int split_k) {
    if (!ShouldPrintConfig() || g_agent_ctx.config_printed) {
        return;
    }

    std::printf(
        "[SpMM] launch_config mode=%s label=%s N=%d SplitK=%d tile=(K=%d,row_warps=%d,col_warps=%d,"
        "warp_col=%d,warp_row=%d,block_k_steps=%d,stages=%d)\n",
        mode,
        label,
        N,
        split_k,
        N2M4TilingConfig::MMA_K_SP,
        N2M4TilingConfig::BLOCK_ROW_WARPS,
        N2M4TilingConfig::BLOCK_COL_WARPS,
        N2M4TilingConfig::WARP_COL_TENSORS,
        N2M4TilingConfig::WARP_ROW_TENSORS,
        N2M4TilingConfig::BLOCK_K_STEPS,
        stages);
    g_agent_ctx.config_printed = true;
}

void PrintFallbackConfig(int N, int split_k) {
    if (!ShouldPrintConfig() || g_agent_ctx.config_printed) {
        return;
    }

    std::printf(
        "[SpMM] launch_config mode=%s label=%s N=%d SplitK=%d tile=(K=%d,row_warps=%d,col_warps=%d,"
        "warp_col=%d,warp_row=%d,block_k_steps=%d,stages=%d)\n",
        "fallback",
        "cusparseLt",
        N,
        split_k,
        0,
        0,
        0,
        0,
        0,
        0,
        0);
    g_agent_ctx.config_printed = true;
}

cudaError_t EnsureFallbackContext() {
    if (g_agent_ctx.fallback_ctx != nullptr) {
        return cudaSuccess;
    }

    g_agent_ctx.fallback_ctx = new CuSparseLtSpmmContext{};
    const cudaError_t error = InitCuSparseLtContext(g_agent_ctx.fallback_ctx,
                                                    g_agent_ctx.M,
                                                    g_agent_ctx.N,
                                                    g_agent_ctx.K,
                                                    const_cast<half*>(g_agent_ctx.dense_a_device),
                                                    const_cast<half*>(g_agent_ctx.dense_b_row_major_device));
    if (error == cudaSuccess) {
        return cudaSuccess;
    }

    delete g_agent_ctx.fallback_ctx;
    g_agent_ctx.fallback_ctx = nullptr;
    return error;
}

template <typename N2M4TilingConfig>
bool ShapeCompatible(int M_Global, int K_Global) {
    return (M_Global % N2M4TilingConfig::TILE_M) == 0 && (K_Global % N2M4TilingConfig::TILE_K) == 0;
}

template <typename N2M4TilingConfig, int stages>
cudaError_t LaunchSparseKernel(cudaStream_t stream,
                               const half* Compressed_A,
                               const half* B,
                               const uint16_t* metadata,
                               half* C,
                               int M_Global,
                               int N_Global,
                               int K_Global,
                               int Split_K) {
    int computeSize = (N2M4TilingConfig::TILE_N * N2M4TilingConfig::TILE_K) * static_cast<int>(sizeof(half)) *
                          stages +
                      (N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2) *
                          static_cast<int>(sizeof(half)) * stages +
                      ((N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)) / 4 * stages;

    const int resultSize = AgentOutputStorePolicy<N2M4TilingConfig>::kUseDirectStore
                               ? 0
                               : (N2M4TilingConfig::TILE_M * (N2M4TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C)) *
                                     static_cast<int>(sizeof(half));
    if (stages > 1) {
        computeSize -= ((N2M4TilingConfig::TILE_M * N2M4TilingConfig::TILE_K / 2)) / 4;
    }

    const int sharedMemSize = std::max(computeSize, resultSize);

    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        return error;
    }

    int maxOptinSharedMem = 0;
    error = cudaDeviceGetAttribute(&maxOptinSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess) {
        return error;
    }

    if (sharedMemSize > maxOptinSharedMem) {
        std::printf("[SpMM] requested dynamic shared memory %d exceeds opt-in limit %d (stages=%d, N=%d).\n",
                    sharedMemSize,
                    maxOptinSharedMem,
                    stages,
                    N_Global);
        return cudaErrorInvalidConfiguration;
    }

    error = cudaFuncSetAttribute(SpMM_N2M4_Kernel<N2M4TilingConfig, stages>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sharedMemSize);
    if (error != cudaSuccess) {
        return error;
    }

    const int dimN = std::max(N_Global / N2M4TilingConfig::TILE_N, 1);
    const int dimM = (M_Global / N2M4TilingConfig::TILE_M) * Split_K;
    const dim3 grid(dimN, dimM, 1);
    const dim3 block(WARP_SIZE * N2M4TilingConfig::BLOCK_WARPS, 1, 1);

    SpMM_N2M4_Kernel<N2M4TilingConfig, stages>
        <<<grid, block, sharedMemSize, stream>>>(Compressed_A, metadata, B, C, M_Global, N_Global, K_Global, Split_K);
    return cudaPeekAtLastError();
}

template <typename N2M4TilingConfig, int stages>
cudaError_t LaunchConfiguredKernel(cudaStream_t stream,
                                   const half* Compressed_A,
                                   const half* B,
                                   const uint16_t* metadata,
                                   half* C,
                                   int M_Global,
                                   int N_Global,
                                   int K_Global,
                                   int Split_K) {
    if (!ShapeCompatible<N2M4TilingConfig>(M_Global, K_Global)) {
        return cudaErrorInvalidValue;
    }

    PrintLaunchConfig<N2M4TilingConfig, stages>("generic", "sp_tc", N_Global, Split_K);
    return LaunchSparseKernel<N2M4TilingConfig, stages>(
        stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
}

cudaError_t LaunchN1024Variant(cudaStream_t stream,
                               const half* Compressed_A,
                               const half* B,
                               const uint16_t* metadata,
                               half* C,
                               int M_Global,
                               int N_Global,
                               int K_Global,
                               int Split_K) {
    const char* variant = GetEnvStringOrDefault("SPMM_N1024_VARIANT", "k32_64x64_s2");

    if (std::strcmp(variant, "k32_64x64_s2") == 0) {
        return LaunchConfiguredKernel<N2M4TilingConfig<32, 2, 2, 4, 2, 2>, 2>(
            stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
    }
    if (std::strcmp(variant, "k32_64x64_s3") == 0) {
        return LaunchConfiguredKernel<N2M4TilingConfig<32, 2, 2, 4, 2, 2>, 3>(
            stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
    }
    if (std::strcmp(variant, "k32_64x64_nwide_s2") == 0) {
        return LaunchConfiguredKernel<N2M4TilingConfig<32, 2, 1, 8, 2, 2>, 2>(
            stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
    }
    if (std::strcmp(variant, "k32_64x64_mwide_s2") == 0) {
        return LaunchConfiguredKernel<N2M4TilingConfig<32, 1, 2, 4, 4, 2>, 2>(
            stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
    }
    if (std::strcmp(variant, "k32_128x64_s2") == 0) {
        return LaunchConfiguredKernel<N2M4TilingConfig<32, 2, 2, 4, 4, 2>, 2>(
            stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
    }
    if (std::strcmp(variant, "k32_64x128_s2") == 0) {
        return LaunchConfiguredKernel<N2M4TilingConfig<32, 2, 2, 8, 2, 2>, 2>(
            stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
    }
    std::printf("[SpMM] unknown SPMM_N1024_VARIANT=%s\n", variant);
    return cudaErrorInvalidValue;
}

cudaError_t LaunchGenericSparseKernel(cudaStream_t stream,
                                      const half* Compressed_A,
                                      const half* B,
                                      const uint16_t* metadata,
                                      half* C,
                                      int M_Global,
                                      int N_Global,
                                      int K_Global,
                                      int Split_K) {
    if (g_agent_ctx.sm < 80 || (K_Global % 64) != 0) {
        return cudaErrorInvalidValue;
    }

    switch (N_Global) {
#if AGENT_ENABLE_SPARSE_K16_PATHS
        case 8:
            return LaunchConfiguredKernel<N2M4TilingConfig<16, 4, 1, 1, 1, 4>, 2>(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
        case 16:
            return LaunchConfiguredKernel<N2M4TilingConfig<16, 4, 1, 2, 1, 4>, 2>(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
        case 32:
            return LaunchConfiguredKernel<N2M4TilingConfig<16, 4, 1, 4, 1, 4>, 2>(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
        case 64:
            return LaunchConfiguredKernel<N2M4TilingConfig<16, 2, 2, 4, 2, 4>, 2>(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
        case 128:
            return LaunchConfiguredKernel<N2M4TilingConfig<16, 2, 2, 4, 4, 4>, 2>(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
        case 256:
            return LaunchConfiguredKernel<N2M4TilingConfig<16, 2, 2, 8, 4, 4>, 2>(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
        case 512:
            return LaunchConfiguredKernel<N2M4ConfigN128Wide, 3>(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
#endif
        case 1024:
            return LaunchN1024Variant(
                stream, Compressed_A, B, metadata, C, M_Global, N_Global, K_Global, Split_K);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace

void AgentDestroyLaunchContext() {
    if (!g_agent_ctx.initialized) {
        return;
    }

    if (g_agent_ctx.fallback_ctx != nullptr) {
        DestroyCuSparseLtContext(g_agent_ctx.fallback_ctx);
        delete g_agent_ctx.fallback_ctx;
        g_agent_ctx.fallback_ctx = nullptr;
    }

    g_agent_ctx = AgentLaunchContext{};
}

cudaError_t AgentConfigureLaunchContext(const half* dense_a_device,
                                        const half* dense_b_row_major_device,
                                        int M,
                                        int N,
                                        int K,
                                        bool dense_b_all_ones) {
    (void)dense_b_all_ones;
    AgentDestroyLaunchContext();

    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        return error;
    }

    int major = 0;
    int minor = 0;
    error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (error != cudaSuccess) {
        return error;
    }

    g_agent_ctx.initialized = true;
    g_agent_ctx.sm = major * 10 + minor;
    g_agent_ctx.M = M;
    g_agent_ctx.N = N;
    g_agent_ctx.K = K;
    g_agent_ctx.dense_a_device = dense_a_device;
    g_agent_ctx.dense_b_row_major_device = dense_b_row_major_device;
    return cudaSuccess;
}

cudaError_t SpMM_N2M4_Launch(cudaStream_t stream,
                             const half* Compressed_A,
                             const half* B,
                             const uint16_t* metadata,
                             half* C,
                             const int M_Global,
                             const int N_Global,
                             const int K_Global,
                             half* Split_Reduction,
                             const int Split_K) {
    if (!g_agent_ctx.initialized) {
        return cudaErrorInvalidDeviceFunction;
    }

    half* kernelOutputPtr = (Split_K == 1) ? C : Split_Reduction;
    if (kernelOutputPtr == nullptr) {
        return cudaErrorInvalidValue;
    }

    const cudaError_t genericError = LaunchGenericSparseKernel(
        stream, Compressed_A, B, metadata, kernelOutputPtr, M_Global, N_Global, K_Global, Split_K);
    if (genericError == cudaSuccess) {
        if (Split_K == 1) {
            return cudaSuccess;
        }

        const dim3 grid(std::max((M_Global * N_Global) / 256, 1), 1, 1);
        const dim3 block(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<grid, block, 0, stream>>>(C, kernelOutputPtr, M_Global, N_Global, Split_K);
        return cudaGetLastError();
    }

    if (genericError != cudaErrorInvalidValue && genericError != cudaErrorInvalidConfiguration) {
        return genericError;
    }

    const cudaError_t fallbackError = EnsureFallbackContext();
    if (fallbackError != cudaSuccess) {
        return fallbackError;
    }

    PrintFallbackConfig(N_Global, Split_K);
    return RunCuSparseLtMatmul(g_agent_ctx.fallback_ctx, C, stream);
}
