#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>

#include "../test/utils.h"
#include "../src/SpMM_API.cu"
#include "backends/cuSparseLt_backend.cuh"

struct BenchCase{
    int M, K, N, SPLIT_K; 
};

struct BenchResult{

};


int main(int argc, char** argv){

    if (argc != 5) {
        printf("Wrong Inputs! Correct input format: ./kernelTest M K N SPLIT_K\n");
        return 0;
    }

    int M_GLOBAL = atoi(argv[1]);
    int K_GLOBAL = atoi(argv[2]);
    int N_GLOBAL = atoi(argv[3]);
    int SPLIT_K  = atoi(argv[4]);

    printf("M_GLOBAL: %d, K_GLOBAL: %d, N_GLOBAL: %d, SPLIT_K: %d \n", M_GLOBAL, K_GLOBAL, N_GLOBAL, SPLIT_K);
    PrintTrialCase("benchMain", M_GLOBAL, K_GLOBAL, N_GLOBAL, SPLIT_K);

    // cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 在主机内存中分配空间
    half* A_host = NULL;  // row major
    half* B_host = NULL;  // col major
    // 由于数组在存储时默认是行主序，但是在实际的推理过程中是列主序
    // 所以多开辟一段空间用于转置
    half* B_rowMajor_host = NULL;  // row major

    // 在 GPU 显存中分配空间
    half* A_device = NULL;
    half* B_device = NULL;
    half* B_rowMajor_device = NULL;

    // 主机分配空间
    A_host = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_host = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    B_rowMajor_host = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    if (A_host == NULL || B_host == NULL || B_rowMajor_host == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }

    // GPU 分配空间
    cudaMalloc(reinterpret_cast<void**>(&A_device), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_device), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_rowMajor_device), sizeof(half) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);
    if (A_device == NULL || B_device == NULL || B_rowMajor_device == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }


    // 获取计算数据
    init_host_matrices_pruned(A_host, M_GLOBAL, K_GLOBAL);
    init_host_matrices(B_rowMajor_host, K_GLOBAL, N_GLOBAL);

    // 转置获取列主序的矩阵 B
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            B_host[i + j * K_GLOBAL] = B_rowMajor_host[i * N_GLOBAL + j];


    // 将数据 copy 到GPU
    printf("Preparing dense data for GPU...\n");
    cudaMemcpy(A_device, A_host, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_rowMajor_device, B_rowMajor_host, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);

    // 执行 cuSPARSELt kernel 计算
    ////////////////////////////////////////////////////////////////////////////////////////////////
    printf("开始执行 cuSPARSELt 计算...\n");
    // 初始化 cuSPARSELt 的存储空间
    half* cusparseLtResult_device = NULL;  // row major
    cudaMalloc(reinterpret_cast<void**>(&cusparseLtResult_device), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (cusparseLtResult_device == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(cusparseLtResult_device, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);

    // kernel 测试
    float milliseconds_cusparseLt = RunCuSparseLtExample(M_GLOBAL, N_GLOBAL, K_GLOBAL, A_device, B_rowMajor_device, cusparseLtResult_device, cusparseLtResult_device);

    // 将 cuSPARSELt 的结果 copy 回 CPU
    half* cusparseLtResult_host = NULL;  // row major
    cusparseLtResult_host = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (cusparseLtResult_host == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(cusparseLtResult_host, cusparseLtResult_device, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Row Major
    cudaFree(cusparseLtResult_device);  // 释放 GPU 上占用的空间

    // 统计 cuSPARSELt 的平均执行时间
    milliseconds_cusparseLt = milliseconds_cusparseLt / BENCHMARK_ITERATION;  // 平均执行时间
    float tflops_cusparseLt = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cusparseLt / 1000.0)) / 1e12;  // TFLOPS

    


    // 执行 cublas kernel 计算
    ////////////////////////////////////////////////////////////////////////////////////////////////
    printf("开始执行 cublas 计算...\n");
    // 初始化 cublas 的存储空间
    half* cublasResult_device = NULL;  // col major
    cudaMalloc(reinterpret_cast<void**>(&cublasResult_device), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (cublasResult_device == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(cublasResult_device, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);

    // kernel 测试
    float milliseconds_cublas = cublasTest(A_device, B_device, cublasResult_device, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    // 将 cublas 的结果 copy 回 CPU
    half* cublasResult_host = NULL;  // col major
    cublasResult_host = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (cublasResult_host == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(cublasResult_host, cublasResult_device, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(cublasResult_device);  // 释放 GPU 上占用的空间

    // 统计 cublas 的平均执行时间
    milliseconds_cublas = milliseconds_cublas / BENCHMARK_ITERATION;  // 平均执行时间
    float tflops_cublas = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas / 1000.0)) / 1e12;  // TFLOPS

    ////////////////////////////////////////////////////////////////////////////////////////////////


    // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel //
    // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel //
    // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel //
    // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel // // 执行我的 Kernel //

    printf("开始执行 N2M4 SpMM 计算...\n");
    // 申请结果存储空间
    half* result_device = NULL;
    cudaMalloc(reinterpret_cast<void**>(&result_device), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (result_device == NULL) {
        printf("Error in cudaMalloc result_device\n");
        exit(-1);
    }
    // 将结果初始化为 0
    cudaMemset(result_device, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);

    // 申请 Split_K 规约的中间存储空间
    half* SplitK_device = NULL;
    cudaMalloc(reinterpret_cast<void**>(&SplitK_device), sizeof(half) * M_GLOBAL * N_GLOBAL * SPLIT_K);
    if (SplitK_device == NULL) {
        printf("Error in cudaMalloc SplitK_device\n");
        exit(-1);
    }
    cudaMemset(SplitK_device, 0, sizeof(half) * M_GLOBAL * N_GLOBAL * SPLIT_K);

    // 初始化输出指针，用于存储压缩之后的 val 和 metadata
    half* compressedMatrixA = nullptr;
    uint16_t* _metadata = nullptr;
    // 对剪枝之后的矩阵进行压缩
    auto valCount = toolCompressMatrixA( A_host, M_GLOBAL, K_GLOBAL, 16, 16, &compressedMatrixA, &_metadata);
    

    // 此处得到的 metadata 需要重新进行交错重排，以适配Kernel内的加载方法
    // 此处得到的 metadata 需要重新进行交错重排，以适配Kernel内的加载方法

    uint16_t* metadata = nullptr;
    toolReorderMetadata(_metadata, M_GLOBAL, K_GLOBAL, 16, 16, &metadata);

    half* packMatrixB_host = nullptr;
    toolPackMatrixB(B_rowMajor_host, K_GLOBAL, N_GLOBAL, 64, 8, &packMatrixB_host);

    // 此处得到的 metadata 需要重新进行交错重排，以适配Kernel内的加载方法
    // 此处得到的 metadata 需要重新进行交错重排，以适配Kernel内的加载方法


    // 初始化 GPU 上用于存储压缩之后的 val 和 metadata
    half* compressedMatrixA_device = nullptr;
    uint16_t* metadata_device = nullptr;
    half* packMatrixB_device = nullptr;

    // 在 GPU 上申请空间
    cudaMalloc(&compressedMatrixA_device, sizeof(half) * (M_GLOBAL * K_GLOBAL / 2)); // for (16*64 tile specific)
    cudaMalloc(&metadata_device, sizeof(uint16_t) * (M_GLOBAL * K_GLOBAL / 16));
    cudaMalloc(&packMatrixB_device, sizeof(half)*(K_GLOBAL * N_GLOBAL));

    if (compressedMatrixA_device == NULL || metadata_device == NULL || packMatrixB_device == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }

    // 将数据 copy 到 GPU 上
    cudaMemcpy(compressedMatrixA_device, compressedMatrixA, sizeof(half) * (M_GLOBAL * K_GLOBAL / 2), cudaMemcpyHostToDevice);
    cudaMemcpy(metadata_device, metadata, sizeof(uint16_t) * (M_GLOBAL * K_GLOBAL / 16), cudaMemcpyHostToDevice);
    cudaMemcpy(packMatrixB_device, packMatrixB_host, sizeof(half)*(K_GLOBAL * N_GLOBAL), cudaMemcpyHostToDevice);

    // 释放 CPU 内存空间中的数据
    free(compressedMatrixA);
    free(metadata);
    free(_metadata);
    free(packMatrixB_host);
    printf("完成! 矩阵 A 已压缩.\n");


    // 测试时间之前进行热身
    for (int i = 0; i < WARM_UP_ITERATION; i++)
        SpMM_N2M4_Launch(0, compressedMatrixA_device, packMatrixB_device, metadata_device, result_device, M_GLOBAL, N_GLOBAL, K_GLOBAL, SplitK_device, SPLIT_K);

    cudaEventRecord(start);  // 开始记录
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        SpMM_N2M4_Launch(0, compressedMatrixA_device, packMatrixB_device, metadata_device, result_device, M_GLOBAL, N_GLOBAL, K_GLOBAL, SplitK_device, SPLIT_K);

    cudaEventRecord(stop);  // 停止记录
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);

    // 计算平均执行时间
    float milliseconds_SpMM = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM, start, stop);  // 总执行时间
    milliseconds_SpMM = milliseconds_SpMM / BENCHMARK_ITERATION;  // 平均执行时间
    float tflops_SpMM = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM / 1000.0)) / 1e12;  // TFLOPS


    // 在 CPU 内存开辟结果存储空间
    half* result_host = NULL;  // col major
    result_host = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    // 将计算结果 copy 到 CPU 内存
    cudaMemcpy(result_host, result_device, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major

    // 释放 GPU 空间
    cudaFree(SplitK_device);
    cudaFree(result_device);
    cudaFree(metadata_device);
    cudaFree(compressedMatrixA_device);
    cudaFree(packMatrixB_device);
    ////////////////////////////////////////////////////////////////////////////////////////////////


    // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 //
    // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 //
    // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 //
    // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 // // 统计计算准确率以及kernel性能 //

    /*
    int preview_count = GetEnvIntOrDefault("SPMM_PREVIEW_COUNT", 16);
    int total_elements = M_GLOBAL * N_GLOBAL;
    if (preview_count > total_elements) {
        preview_count = total_elements;
    }

    if (preview_count > 0) {
        printf("cublasResult_host shape: [%d, %d], first %d elements: ", M_GLOBAL, N_GLOBAL, preview_count);
        for (int i = 0; i < preview_count; ++i) {
            printf("%.4f ", __half2float(cublasResult_host[i]));
        }
        printf("\n");

        printf("result_host shape: [%d, %d], first %d elements: ", M_GLOBAL, N_GLOBAL, preview_count);
        for (int i = 0; i < preview_count; ++i) {
            printf("%.4f ", __half2float(result_host[i]));
        }
        printf("\n");

        printf("cusparseLtResult_host shape: [%d, %d], first %d elements: ", M_GLOBAL, N_GLOBAL, preview_count);
        for (int i = 0; i < preview_count; ++i) {
            printf("%.4f ", __half2float(cusparseLtResult_host[i]));
        }
        printf("\n");
    }
    */
    
    int totalErrorNums = 0;
    totalErrorNums = ComputeTotalError(cublasResult_host, result_host, M_GLOBAL, N_GLOBAL, true);  // 统计我设计的kernel的准确率

    int totalErrorNums_cusparseLt = 0;
    totalErrorNums_cusparseLt = ComputeTotalError(cublasResult_host, cusparseLtResult_host, M_GLOBAL, N_GLOBAL, false);  // 统计 cusparseLt 的准确率

    free(result_host);  // 释放空间
    free(cusparseLtResult_host);  // 释放空间
    PrintPerformance("cublas", milliseconds_cublas, tflops_cublas, 0);
    PrintPerformance("cusparseLt", milliseconds_cusparseLt, tflops_cusparseLt, totalErrorNums_cusparseLt);
    PrintPerformance("n2m4", milliseconds_SpMM, tflops_SpMM, totalErrorNums);


    free(cublasResult_host);  // 释放标准答案
    // 释放 CPU 上的输入数据
    free(A_host); 
    free(B_host);
    free(B_rowMajor_host);
    // 释放 GPU 上的输入数据
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(B_rowMajor_device);

    return 0;
}





