#pragma once

#include <cstdio>
#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

struct CuSparseLtSpmmContext {
    bool initialized = false;
    int M = 0;
    int N = 0;
    int K = 0;

    half* dA_dense = nullptr;
    half* dB_dense = nullptr;

    cusparseLtHandle_t handle{};
    cusparseLtMatDescriptor_t matA{};
    cusparseLtMatDescriptor_t matB{};
    cusparseLtMatDescriptor_t matC{};
    cusparseLtMatDescriptor_t matD{};
    cusparseLtMatmulDescriptor_t matmul{};
    cusparseLtMatmulAlgSelection_t alg_sel{};
    cusparseLtMatmulPlan_t plan{};

    void* dA_compressed = nullptr;
    void* dWorkspace = nullptr;
    void* dCompressBuffer = nullptr;
};

static constexpr cusparseComputeType AGENT_CUSPARSELT_COMPUTE_TYPE = CUSPARSE_COMPUTE_32F;

inline cudaError_t ReportCuSparseLtError(const char* expr, cusparseStatus_t status, int line) {
    std::printf("cuSPARSELt error at line %d for %s: status=%d\n", line, expr, static_cast<int>(status));
    return cudaErrorUnknown;
}

#define AGENT_CUSPARSELT_CHECK(expr)                                         \
    do {                                                                     \
        const cusparseStatus_t _status = (expr);                             \
        if (_status != CUSPARSE_STATUS_SUCCESS) {                            \
            return ReportCuSparseLtError(#expr, _status, __LINE__);          \
        }                                                                    \
    } while (0)

inline cudaError_t InitCuSparseLtContext(CuSparseLtSpmmContext* ctx,
                                         int M,
                                         int N,
                                         int K,
                                         half* dA_dense,
                                         half* dB_dense) {
    if (ctx == nullptr) {
        return cudaErrorInvalidValue;
    }

    std::memset(ctx, 0, sizeof(*ctx));
    ctx->M = M;
    ctx->N = N;
    ctx->K = K;
    ctx->dA_dense = dA_dense;
    ctx->dB_dense = dB_dense;

    AGENT_CUSPARSELT_CHECK(cusparseLtInit(&ctx->handle));
    AGENT_CUSPARSELT_CHECK(cusparseLtStructuredDescriptorInit(&ctx->handle,
                                                              &ctx->matA,
                                                              M,
                                                              K,
                                                              K,
                                                              16,
                                                              CUDA_R_16F,
                                                              CUSPARSE_ORDER_ROW,
                                                              CUSPARSELT_SPARSITY_50_PERCENT));
    AGENT_CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(&ctx->handle,
                                                         &ctx->matB,
                                                         K,
                                                         N,
                                                         N,
                                                         16,
                                                         CUDA_R_16F,
                                                         CUSPARSE_ORDER_ROW));
    AGENT_CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(&ctx->handle,
                                                         &ctx->matC,
                                                         M,
                                                         N,
                                                         N,
                                                         16,
                                                         CUDA_R_16F,
                                                         CUSPARSE_ORDER_ROW));
    AGENT_CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(&ctx->handle,
                                                         &ctx->matD,
                                                         M,
                                                         N,
                                                         N,
                                                         16,
                                                         CUDA_R_16F,
                                                         CUSPARSE_ORDER_ROW));
    AGENT_CUSPARSELT_CHECK(cusparseLtMatmulDescriptorInit(&ctx->handle,
                                                          &ctx->matmul,
                                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                          &ctx->matA,
                                                          &ctx->matB,
                                                          &ctx->matC,
                                                          &ctx->matD,
                                                          AGENT_CUSPARSELT_COMPUTE_TYPE));
    AGENT_CUSPARSELT_CHECK(cusparseLtMatmulAlgSelectionInit(&ctx->handle,
                                                            &ctx->alg_sel,
                                                            &ctx->matmul,
                                                            CUSPARSELT_MATMUL_ALG_DEFAULT));
    AGENT_CUSPARSELT_CHECK(cusparseLtMatmulPlanInit(&ctx->handle,
                                                    &ctx->plan,
                                                    &ctx->matmul,
                                                    &ctx->alg_sel));

    size_t workspace_size = 0;
    size_t compressed_size = 0;
    size_t compress_buffer_size = 0;
    AGENT_CUSPARSELT_CHECK(cusparseLtMatmulGetWorkspace(&ctx->handle, &ctx->plan, &workspace_size));
    AGENT_CUSPARSELT_CHECK(
        cusparseLtSpMMACompressedSize(&ctx->handle, &ctx->plan, &compressed_size, &compress_buffer_size));

    cudaMalloc(&ctx->dA_compressed, compressed_size);
    cudaMalloc(&ctx->dWorkspace, workspace_size);
    cudaMalloc(&ctx->dCompressBuffer, compress_buffer_size);

    int is_valid = 1;
    int* d_valid = nullptr;
    cudaMalloc(&d_valid, sizeof(int));
    AGENT_CUSPARSELT_CHECK(cusparseLtSpMMAPruneCheck(&ctx->handle, &ctx->matmul, dA_dense, d_valid, 0));
    cudaMemcpy(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_valid);

    if (is_valid != 0) {
        std::printf("cuSPARSELt prune check failed for M=%d K=%d N=%d\n", M, K, N);
        return cudaErrorInvalidValue;
    }

    AGENT_CUSPARSELT_CHECK(
        cusparseLtSpMMACompress(&ctx->handle, &ctx->plan, dA_dense, ctx->dA_compressed, ctx->dCompressBuffer, 0));

    ctx->initialized = true;
    return cudaSuccess;
}

inline cudaError_t RunCuSparseLtMatmul(CuSparseLtSpmmContext* ctx, half* dOutput, cudaStream_t stream = 0) {
    if (ctx == nullptr || !ctx->initialized) {
        return cudaErrorInvalidDeviceFunction;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    AGENT_CUSPARSELT_CHECK(cusparseLtMatmul(&ctx->handle,
                                            &ctx->plan,
                                            &alpha,
                                            ctx->dA_compressed,
                                            ctx->dB_dense,
                                            &beta,
                                            dOutput,
                                            dOutput,
                                            ctx->dWorkspace,
                                            &stream,
                                            1));
    return cudaGetLastError();
}

inline float BenchmarkCuSparseLt(CuSparseLtSpmmContext* ctx,
                                 half* dOutput,
                                 int warmup_iterations,
                                 int benchmark_iterations) {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iterations; ++i) {
        RunCuSparseLtMatmul(ctx, dOutput, 0);
    }

    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iterations; ++i) {
        RunCuSparseLtMatmul(ctx, dOutput, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms;
}

inline void DestroyCuSparseLtContext(CuSparseLtSpmmContext* ctx) {
    if (ctx == nullptr || !ctx->initialized) {
        return;
    }

    cudaFree(ctx->dA_compressed);
    cudaFree(ctx->dWorkspace);
    cudaFree(ctx->dCompressBuffer);
    cusparseLtMatmulPlanDestroy(&ctx->plan);
    cusparseLtMatDescriptorDestroy(&ctx->matA);
    cusparseLtMatDescriptorDestroy(&ctx->matB);
    cusparseLtMatDescriptorDestroy(&ctx->matC);
    cusparseLtMatDescriptorDestroy(&ctx->matD);
    cusparseLtDestroy(&ctx->handle);
    std::memset(ctx, 0, sizeof(*ctx));
}
