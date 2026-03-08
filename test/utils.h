#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <iomanip>
#include <algorithm> 
#include <fstream>
#include <random>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#define WARM_UP_ITERATION 5  // 热身次数
#define BENCHMARK_ITERATION 10  // 测试次数



// 捕获并报告由 CUDA API 调用产生的错误
void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


// 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 //
// 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 //
// 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 //
// 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 // // 矩阵生成 //


// 模拟生成 2:4 剪枝的矩阵
__host__ void init_host_matrices_pruned(half* matrix, int row, int col)
{
    static std::mt19937 gen{std::random_device{}()};          // 线程安全随机引擎
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int total = row * col;
    for (int i = 0; i < total; i += 4) {
        // 产生 0-3 的随机排列
        int idx[4] = {0,1,2,3};
        for (int j = 3; j > 0; --j) {
            int r = rand() % (j + 1);
            std::swap(idx[j], idx[r]);
        }
        // 前 2 个下标置 0，后 2 个置随机值
        for (int j = 0; j < 4; ++j) {
            if (j < 2)
                matrix[i + idx[j]] = __float2half_rn(0.0f);
            else
                // if((i + idx[j]) % col / 64 == 0){
                //     matrix[i + idx[j]] = __float2half_rn(3.0f);
                // }else{
                //     matrix[i + idx[j]] = __float2half_rn(3.0f);
                // }
                // matrix[i + idx[j]] = __float2half_rn(dist(gen));
                // matrix[i + idx[j]] = __float2half_rn(float((i + idx[j]) % col));
                // matrix[i + idx[j]] = __float2half_rn(float((i + idx[j]) / col));
                matrix[i + idx[j]] = __float2half_rn(1.0f);
        }
    }
}



/* -------------------- 随机初始化矩阵 -------------------- */
// 模拟生成对应的激活矩阵
__host__ void init_host_matrices(half* matrix, int row, int col)
{
    static std::mt19937 gen{std::random_device{}()};          // 线程安全随机引擎
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++){
            // matrix[i*col + j] = __float2half_rn(float((i*col + j) % 10));
            // matrix[i*col + j] = __float2half_rn(float(i % 2 + 1));
            // matrix[i*col + j] = __float2half_rn(dist(gen));
            matrix[i*col + j] = __float2half_rn(1.0f);
        }
    }
}



// __host__ 函数，完全在CPU上运行
// 用于权重矩阵压缩
// 将 2:4 剪枝之后的稠密矩阵 denseMatrixA，压缩生成非 0 元素矩阵 compressedMatrixA，以及记录非 0 元素位置的 matadata
__host__ int toolCompressMatrixA(   half* denseMatrixA,
                                    int Global_M,
                                    int Global_K,
                                    int Block_M,  // 16
                                    int Block_K,  // 16
                                    half** compressedMatrixA,
                                    uint16_t** matadata)
{
    int blockRowNums = Global_M / Block_M;  // 行方向上有多少块
    int blockColNums = Global_K / Block_K;  // 列方向上有多少块


    *compressedMatrixA = (half*)malloc(Global_M * Global_K/2 * sizeof(half));  // 非 0 数据的存储空间
    *matadata = (uint16_t*)malloc(blockRowNums * blockColNums * 16 * sizeof(uint16_t));  // metadata的存储空间

    // 申请失败则返回 -1 报错
    if (*compressedMatrixA == nullptr || *matadata == nullptr ) {
        printf("初始化空间失败");
        return -1;
    }

    int valCount = 0;
    int metadataCount = 0;
    for(int blockRow = 0; blockRow < blockRowNums; blockRow++){
        for(int blockCol = 0; blockCol < blockColNums; blockCol++){  // 遍历所有块
            // 遍历块内数据进行压缩
            for(int row = 0; row < Block_M; row++){
                uint16_t _metadata = 0;  // 每行非 0 元素的位置使用一个 metadata 进行表示
                int tmpOffset = 0;
                for(int _col=0; _col<Block_K; _col += 4){  // 每 4 个数压缩两个非 0 值
                    int zeroNum = 0;  // 记录 0 的个数
                    for(int offset=0; offset<4; offset++){

                        int col = _col + offset;
                        int realRow = blockRow * Block_M + row;  // 计算当前数据是整个矩阵的第几行
                        int realCol = blockCol * Block_K + col;  // 计算当前数据是整个矩阵的第几列
                        
                        half val = denseMatrixA[realRow * Global_K + realCol];  // 读取对应数据

                        if(zeroNum < 2 && __half2float(val) == 0.0f){  //如果当前不足两个 0 值并且当前值为 0 则跳过
                            zeroNum += 1;
                            continue;
                        }else{  // 否则则记录 val 以及对应的 metadata
                            (*compressedMatrixA)[valCount] = val;
                            valCount += 1;

                            int tmpMetadata = offset << (2*tmpOffset);
                            tmpOffset += 1;
                            _metadata = _metadata|tmpMetadata;
                        }
                    }
                }
                (*matadata)[metadataCount] = _metadata;
                metadataCount += 1;
            }
        }
    }

    return valCount;
}



// 对 metadata 进行重排
__host__ void toolReorderMetadata(  uint16_t* matadata,
                                    int Global_M,
                                    int Global_K,
                                    int Block_M,  // 16
                                    int Block_K,  // 16
                                    uint16_t** reorderMatadata)
{
    int blockRowNums = Global_M / Block_M;  // 行方向上有多少块
    int blockColNums = Global_K / Block_K;  // 列方向上有多少块

    *reorderMatadata = (uint16_t*)malloc(blockRowNums * blockColNums * 16 * sizeof(uint16_t));  // metadata的存储空间，每块16个uint16_t

    // 申请失败则返回
    if (*reorderMatadata == nullptr ) {
        printf("初始化空间失败");
        return;
    }

    for(int blockRow = 0; blockRow < blockRowNums; blockRow++){
        for(int blockCol = 0; blockCol < blockColNums; blockCol += 4){  // 按行主序遍历每个块，每 4 个块进行重排打包
            // 计算当前 4 个块的 metadata 在原始 metadata 中的起始位置
            int idx = (blockRow * blockColNums + blockCol) * 16;  // 每块有 16 个 uint16_t
            for(int i=0; i<8; i++){
                for(int j=0; j<8; j++){
                    int offset = (j % 2) * 8 + (j / 2) * 16 + i;  // 当前 4 个块内的偏移
                    (*reorderMatadata)[idx + i * 8 + j] = matadata[idx + offset];  // 将原始 metadata 中对应位置的值放到重排后 metadata 的对应位置上
                }
            }
        }
    }
}

// 对 matrixB 进行 pack
__host__ void toolPackMatrixB(  half* MatrixB,  // matrixB 是行主序存储的
                                int Global_K,
                                int Global_N,
                                int Block_K,  // 64
                                int Block_N,  // 8
                                half** packMatrixB)
{
    int blockRowNums = Global_K / Block_K;  // 行方向上有多少块
    int blockColNums = Global_N / Block_N;  // 列方向上有多少块

    *packMatrixB = (half*)malloc(Global_K * Global_N * sizeof(half));  // 申请存储空间

    // 申请失败则返回
    if (*packMatrixB == nullptr ) {
        printf("初始化空间失败");
        return;
    }

    half* dst = *packMatrixB;

    // 块间列主序：blockId = blockCol * blockRowNums + blockRow
    for (int bc = 0; bc < blockColNums; ++bc) {
        for (int br = 0; br < blockRowNums; ++br) {
            size_t blockId = (size_t)bc * blockRowNums + br;
            size_t baseDst = blockId * (size_t)Block_K * Block_N;

            int baseRow = br * Block_K;
            int baseCol = bc * Block_N;

            // 块内行主序
            for (int r = 0; r < Block_K; ++r) {
                const half* srcRowPtr = MatrixB + (size_t)(baseRow + r) * Global_N + baseCol;
                half* dstRowPtr = dst + baseDst + (size_t)r * Block_N;
                for (int c = 0; c < Block_N; ++c) {
                    dstRowPtr[c] = srcRowPtr[c];
                }
            }
        }
    }


}


// 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // 
// 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // 
// 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // 
// 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // // 统计kernel计算正确性 // 

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t ErrorRate: %.2lf\n",
           KernelName,
           milliseconds,
           tflops,
           error);
}


// 假设：standResult 来自 cuBLAS（列主序, MxN）
//       kernelResult 是行主序（row-major, MxN）
int ComputeTotalError(const half* standResult,
                      const half* kernelResult,
                      int m, int n,
                      bool storeResult)
{
    int totalError = 0;

    std::ofstream outkernelResult("/data/zhb/code/cudaCode/test/debugData/ans/myKernel.txt");
    std::ofstream outstandResult("/data/zhb/code/cudaCode/test/debugData/ans/cublas.txt");

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            // 逻辑元素 (i, j)
            float s = __half2float(standResult[j * m + i]);   // column-major
            float k = __half2float(kernelResult[i * n + j]);  // row-major

            if (fabsf(s - k) > 1e-1f) {
                totalError += 1;
                // printf("Error at (%d, %d): standResult = %f, kernelResult = %f\n", i, j, s, k);
            }

            if(storeResult){
                outstandResult << s << "   ";
                outkernelResult << k << "   ";
            }

        }

        if(storeResult){
            outkernelResult << "\n";
            outstandResult << "\n";
        }
    }

    return totalError;
}









// cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 //
// cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 //
// cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 //
// cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 // // cublas 测试 //


void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

// 这里传入的都是 GPU 端的地址
// A 是行主序
// B 是列主序
// C 是列主序
float cublasTest(const half* A, const half* B, half* C, int M, int N, int K){

    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 初始化 cubla
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);

    // 启用Tensor Core
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();
    
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);  // 使用默认算法

    // 热身
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(   handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, 
                                        &alpha, A, CUDA_R_16F, K,
                                        B, CUDA_R_16F, K,
                                        &beta, C, CUDA_R_16F, M,
                                        CUDA_R_32F,
                                        CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }

    // 测试
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(   handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, 
                        &alpha, A, CUDA_R_16F, K,
                        B, CUDA_R_16F, K,
                        &beta, C, CUDA_R_16F, M,
                        CUDA_R_32F,
                        CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    // 统计执行时间
    float milliseconds_cublas_tc = 0;
    cudaEventElapsedTime(&milliseconds_cublas_tc, start, stop);
    
    return milliseconds_cublas_tc;
}


////////////////////////////////////////////////////////////////////////////////////

























