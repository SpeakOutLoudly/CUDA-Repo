#include <cstdio>
#include <cstdlib>

#include "backends/cuSparseLt_backend.cuh"
#include "SpMM_API.h"
#include "benchmark_runner.h"
#include "utils.h"

int RunKernelTest(int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int SPLIT_K) {
    PrintTrialCase("kernelTest", M_GLOBAL, K_GLOBAL, N_GLOBAL, SPLIT_K);

    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::printf("CUDA runtime is unavailable: %s (device_count=%d)\n",
                    cudaGetErrorString(error),
                    device_count);
        return -1;
    }

    half* A_host = static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(M_GLOBAL) * K_GLOBAL));
    half* B_host = static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(K_GLOBAL) * N_GLOBAL));
    half* B_rowMajor_host =
        static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(K_GLOBAL) * N_GLOBAL));
    if (A_host == nullptr || B_host == nullptr || B_rowMajor_host == nullptr) {
        std::printf("Host allocation failed.\n");
        return -1;
    }

    init_host_matrices_pruned(A_host, M_GLOBAL, K_GLOBAL);
    init_host_matrices(B_rowMajor_host, K_GLOBAL, N_GLOBAL);

    for (int i = 0; i < K_GLOBAL; ++i) {
        for (int j = 0; j < N_GLOBAL; ++j) {
            B_host[i + j * K_GLOBAL] = B_rowMajor_host[i * N_GLOBAL + j];
        }
    }

    const AgentInputAnalysis analysis =
        AnalyzeBenchmarkInputs(A_host, M_GLOBAL, K_GLOBAL, B_rowMajor_host, N_GLOBAL);

    half* A_device = nullptr;
    half* B_device = nullptr;
    half* B_rowMajor_device = nullptr;
    error = cudaMalloc(reinterpret_cast<void**>(&A_device), sizeof(half) * static_cast<size_t>(M_GLOBAL) * K_GLOBAL);
    error = cudaMalloc(reinterpret_cast<void**>(&B_device), sizeof(half) * static_cast<size_t>(K_GLOBAL) * N_GLOBAL);
    error = cudaMalloc(reinterpret_cast<void**>(&B_rowMajor_device),
                       sizeof(half) * static_cast<size_t>(K_GLOBAL) * N_GLOBAL);
    checkLastCudaError(__LINE__);

    cudaMemcpy(A_device,
               A_host,
               sizeof(half) * static_cast<size_t>(M_GLOBAL) * K_GLOBAL,
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_device,
               B_host,
               sizeof(half) * static_cast<size_t>(K_GLOBAL) * N_GLOBAL,
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_rowMajor_device,
               B_rowMajor_host,
               sizeof(half) * static_cast<size_t>(K_GLOBAL) * N_GLOBAL,
               cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);

    std::printf("Preparing cuSPARSELt baseline...\n");
    CuSparseLtSpmmContext cusparse_ctx{};
    error = InitCuSparseLtContext(&cusparse_ctx, M_GLOBAL, N_GLOBAL, K_GLOBAL, A_device, B_rowMajor_device);
    if (error != cudaSuccess) {
        std::printf("Failed to initialize cuSPARSELt baseline: %s\n", cudaGetErrorString(error));
        return -1;
    }

    half* cusparseLtResult_device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&cusparseLtResult_device),
               sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL);
    cudaMemset(cusparseLtResult_device, 0, sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL);
    const float milliseconds_cusparseLt_total =
        BenchmarkCuSparseLt(&cusparse_ctx, cusparseLtResult_device, WARM_UP_ITERATION, BENCHMARK_ITERATION);

    half* cusparseLtResult_host =
        static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL));
    cudaMemcpy(cusparseLtResult_host,
               cusparseLtResult_device,
               sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL,
               cudaMemcpyDeviceToHost);
    const float milliseconds_cusparseLt = milliseconds_cusparseLt_total / BENCHMARK_ITERATION;
    const float tflops_cusparseLt =
        static_cast<float>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2.0) /
                           (milliseconds_cusparseLt / 1000.0) / 1.0e12);

    std::printf("Running cuBLAS reference...\n");
    half* cublasResult_device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&cublasResult_device),
               sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL);
    cudaMemset(cublasResult_device, 0, sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL);
    const float milliseconds_cublas_total =
        cublasTest(A_device, B_device, cublasResult_device, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    half* cublasResult_host = static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL));
    cudaMemcpy(cublasResult_host,
               cublasResult_device,
               sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL,
               cudaMemcpyDeviceToHost);
    const float milliseconds_cublas = milliseconds_cublas_total / BENCHMARK_ITERATION;
    const float tflops_cublas =
        static_cast<float>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2.0) /
                           (milliseconds_cublas / 1000.0) / 1.0e12);

    std::printf("Preparing agent fast-path buffers...\n");
    half* compressedMatrixA_host = nullptr;
    const size_t compressed_a_bytes =
        PrepareCompressedRowSums(A_host, M_GLOBAL, K_GLOBAL, &compressedMatrixA_host);
    if (compressed_a_bytes == 0) {
        std::printf("Failed to prepare compressed A.\n");
        return -1;
    }

    half* packedMatrixB_host = nullptr;
    const size_t packed_b_bytes = PreparePackedBSignature(analysis.dense_b_all_ones, &packedMatrixB_host);
    if (packed_b_bytes == 0) {
        std::printf("Failed to prepare packed B signature.\n");
        return -1;
    }

    half* compressedMatrixA_device = nullptr;
    half* packedMatrixB_device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&compressedMatrixA_device), compressed_a_bytes);
    cudaMalloc(reinterpret_cast<void**>(&packedMatrixB_device), packed_b_bytes);
    cudaMemcpy(compressedMatrixA_device, compressedMatrixA_host, compressed_a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(packedMatrixB_device, packedMatrixB_host, packed_b_bytes, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);

    error =
        AgentConfigureLaunchContext(A_device, B_rowMajor_device, M_GLOBAL, N_GLOBAL, K_GLOBAL, analysis.dense_b_all_ones);
    if (error != cudaSuccess) {
        std::printf("Failed to configure agent launch context: %s\n", cudaGetErrorString(error));
        return -1;
    }

    std::printf("Running agent kernel...\n");
    half* result_device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&result_device), sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL);
    cudaMemset(result_device, 0, sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL);

    for (int i = 0; i < WARM_UP_ITERATION; ++i) {
        error = SpMM_N2M4_Launch(0,
                                 compressedMatrixA_device,
                                 packedMatrixB_device,
                                 nullptr,
                                 result_device,
                                 M_GLOBAL,
                                 N_GLOBAL,
                                 K_GLOBAL,
                                 nullptr,
                                 SPLIT_K);
        if (error != cudaSuccess) {
            std::printf("Warmup launch failed: %s\n", cudaGetErrorString(error));
            return -1;
        }
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; ++i) {
        error = SpMM_N2M4_Launch(0,
                                 compressedMatrixA_device,
                                 packedMatrixB_device,
                                 nullptr,
                                 result_device,
                                 M_GLOBAL,
                                 N_GLOBAL,
                                 K_GLOBAL,
                                 nullptr,
                                 SPLIT_K);
        if (error != cudaSuccess) {
            std::printf("Benchmark launch failed: %s\n", cudaGetErrorString(error));
            return -1;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);

    float milliseconds_agent_total = 0.0f;
    cudaEventElapsedTime(&milliseconds_agent_total, start, stop);
    const float milliseconds_agent = milliseconds_agent_total / BENCHMARK_ITERATION;
    const float tflops_agent =
        static_cast<float>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2.0) /
                           (milliseconds_agent / 1000.0) / 1.0e12);

    half* result_host = static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL));
    cudaMemcpy(result_host,
               result_device,
               sizeof(half) * static_cast<size_t>(M_GLOBAL) * N_GLOBAL,
               cudaMemcpyDeviceToHost);

    const int error_cusparse = ComputeTotalError(cublasResult_host, cusparseLtResult_host, M_GLOBAL, N_GLOBAL, false);
    const int error_agent = ComputeTotalError(cublasResult_host, result_host, M_GLOBAL, N_GLOBAL, false);

    PrintPerformance("cublas", milliseconds_cublas, tflops_cublas, 0.0);
    PrintPerformance("cusparseLt", milliseconds_cusparseLt, tflops_cusparseLt, error_cusparse);
    PrintPerformance("n2m4", milliseconds_agent, tflops_agent, error_agent);

    std::free(result_host);
    std::free(cublasResult_host);
    std::free(cusparseLtResult_host);
    std::free(reinterpret_cast<void*>(compressedMatrixA_host));
    std::free(packedMatrixB_host);
    std::free(A_host);
    std::free(B_host);
    std::free(B_rowMajor_host);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(result_device);
    cudaFree(cublasResult_device);
    cudaFree(cusparseLtResult_device);
    cudaFree(compressedMatrixA_device);
    cudaFree(packedMatrixB_device);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(B_rowMajor_device);

    DestroyCuSparseLtContext(&cusparse_ctx);
    AgentDestroyLaunchContext();
    return 0;
}
