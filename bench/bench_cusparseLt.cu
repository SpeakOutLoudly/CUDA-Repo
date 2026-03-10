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
    PrintTrialCase("bench_cusparseLt", M_GLOBAL, K_GLOBAL, N_GLOBAL, SPLIT_K);

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
    
    free(cusparseLtResult_host);  // 释放空间
    PrintPerformance("cusparseLt", milliseconds_cusparseLt, tflops_cusparseLt, 0);


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





