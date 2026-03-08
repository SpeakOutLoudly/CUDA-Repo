#include <cusparseLt.h>
#include <cuda_runtime.h>

float RunCuSparseLtExample(int M, int N, int K,
                           half* dA_dense,
                           half* dB_dense,
                           half* dC,
                           half* dD) {
    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC, matD;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cusparseLtInit(&handle);

    cusparseLtStructuredDescriptorInit(
        &handle, &matA,
        M, K, K, 16,
        CUDA_R_16F, CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT);

    cusparseLtDenseDescriptorInit(
        &handle, &matB,
        K, N, N, 16,
        CUDA_R_16F, CUSPARSE_ORDER_ROW);

    cusparseLtDenseDescriptorInit(
        &handle, &matC,
        M, N, N, 16,
        CUDA_R_16F, CUSPARSE_ORDER_ROW);

    cusparseLtDenseDescriptorInit(
        &handle, &matD,
        M, N, N, 16,
        CUDA_R_16F, CUSPARSE_ORDER_ROW);

    cusparseLtMatmulDescriptorInit(
        &handle, &matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD,
        CUSPARSE_COMPUTE_32F);

    cusparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);

    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);

    size_t workspace_size = 0, compressed_size = 0, compress_buffer_size = 0;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compress_buffer_size);

    void* dA_compressed = nullptr;
    void* dWorkspace = nullptr;
    void* dCompressBuffer = nullptr;
    cudaMalloc(&dA_compressed, compressed_size);
    cudaMalloc(&dWorkspace, workspace_size);
    cudaMalloc(&dCompressBuffer, compress_buffer_size);

    int is_valid = 1;
    int* d_valid = nullptr;
    cudaMalloc(&d_valid, sizeof(int));
    cusparseLtSpMMAPruneCheck(&handle, &matmul, dA_dense, d_valid, 0);
    cudaMemcpy(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost);
    if (is_valid != 0) {
        printf("cuSPARSELt prune check failed.\n");
        exit(-1);
    }

    cusparseLtSpMMACompress(&handle, &plan, dA_dense, dA_compressed, dCompressBuffer, 0);

    float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream = 0;
    half* dC_input = (dC != nullptr) ? dC : dD;

    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cusparseLtMatmul(&handle, &plan,
                         &alpha,
                         dA_compressed,
                         dB_dense,
                         &beta,
                         dC_input,
                         dD,
                         dWorkspace,
                         &stream,
                         1);
    }

    cudaEventRecord(start, stream);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        cusparseLtMatmul(&handle, &plan,
                         &alpha,
                         dA_compressed,
                         dB_dense,
                         &beta,
                         dC_input,
                         dD,
                         dWorkspace,
                         &stream,
                         1);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds_cusparseLt = 0.0f;
    cudaEventElapsedTime(&milliseconds_cusparseLt, start, stop);

    cudaFree(dA_compressed);
    cudaFree(dWorkspace);
    cudaFree(dCompressBuffer);
    cudaFree(d_valid);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    cusparseLtDestroy(&handle);

    return milliseconds_cusparseLt;
}