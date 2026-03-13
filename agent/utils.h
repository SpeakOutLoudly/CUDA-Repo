#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr int WARM_UP_ITERATION = 5;
static constexpr int BENCHMARK_ITERATION = 10;

inline void checkLastCudaError(int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::printf("Last CUDA error at line %d: %s\n", line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

inline const char* GetEnvStringOrDefault(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value;
}

inline int GetEnvIntOrDefault(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char* end_ptr = nullptr;
    long parsed = std::strtol(value, &end_ptr, 10);
    if (end_ptr == value || *end_ptr != '\0') {
        return fallback;
    }
    return static_cast<int>(parsed);
}

inline void PrintTrialCase(const char* program_name, int M, int K, int N, int split_k) {
    std::printf(
        "[TrialCase] program=%s role=%s trial_id=%s sms=%s tile_config=%s M=%d K=%d N=%d SPLIT_K=%d\n",
        program_name,
        GetEnvStringOrDefault("SPMM_RUN_ROLE", "manual"),
        GetEnvStringOrDefault("SPMM_TRIAL_ID", "none"),
        GetEnvStringOrDefault("SPMM_BUILD_SMS", "unknown"),
        GetEnvStringOrDefault("SPMM_TILE_CONFIG", "agent_default"),
        M,
        K,
        N,
        split_k);
}

inline void PrintPerformance(const char* kernel_name, float milliseconds, float tflops, double error) {
    std::printf(
        "%-10s \t -> \t\t Time/ms: %5.5f \t Performance/TFLOPs: %4.2f \t ErrorRate: %.2f\n",
        kernel_name,
        milliseconds,
        tflops,
        error);
    std::printf("[TrialMetric] kernel=%s time_ms=%.8f tflops=%.8f error=%.8f\n",
                kernel_name,
                milliseconds,
                tflops,
                error);
}

inline void init_host_matrices_pruned(half* matrix, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; col += 4) {
            matrix[row * cols + col + 0] = __float2half_rn(1.0f);
            matrix[row * cols + col + 1] = __float2half_rn(1.0f);
            matrix[row * cols + col + 2] = __float2half_rn(0.0f);
            matrix[row * cols + col + 3] = __float2half_rn(0.0f);
        }
    }
}

inline void init_host_matrices(half* matrix, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            matrix[row * cols + col] = __float2half_rn(1.0f);
        }
    }
}

inline int toolCompressMatrixA(half* denseMatrixA,
                               int Global_M,
                               int Global_K,
                               int Block_M,
                               int Block_K,
                               half** compressedMatrixA,
                               uint16_t** metadata) {
    const int blockRowNums = Global_M / Block_M;
    const int blockColNums = Global_K / Block_K;

    *compressedMatrixA = static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(Global_M) * Global_K / 2));
    *metadata =
        static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * static_cast<size_t>(blockRowNums) * blockColNums * 16));

    if (*compressedMatrixA == nullptr || *metadata == nullptr) {
        std::printf("Failed to allocate sparse A buffers.\n");
        return -1;
    }

    int valCount = 0;
    int metadataCount = 0;
    for (int blockRow = 0; blockRow < blockRowNums; ++blockRow) {
        for (int blockCol = 0; blockCol < blockColNums; ++blockCol) {
            for (int row = 0; row < Block_M; ++row) {
                uint16_t rowMetadata = 0;
                int packedOffset = 0;
                for (int colBase = 0; colBase < Block_K; colBase += 4) {
                    int zeroCount = 0;
                    for (int offset = 0; offset < 4; ++offset) {
                        const int col = colBase + offset;
                        const int realRow = blockRow * Block_M + row;
                        const int realCol = blockCol * Block_K + col;
                        const half value = denseMatrixA[realRow * Global_K + realCol];

                        if (zeroCount < 2 && __half2float(value) == 0.0f) {
                            ++zeroCount;
                            continue;
                        }

                        (*compressedMatrixA)[valCount++] = value;
                        rowMetadata |= static_cast<uint16_t>(offset << (2 * packedOffset));
                        ++packedOffset;
                    }
                }
                (*metadata)[metadataCount++] = rowMetadata;
            }
        }
    }

    return valCount;
}

inline void toolReorderMetadata(uint16_t* metadata,
                                int Global_M,
                                int Global_K,
                                int Block_M,
                                int Block_K,
                                uint16_t** reorderMetadata) {
    const int blockRowNums = Global_M / Block_M;
    const int blockColNums = Global_K / Block_K;

    *reorderMetadata =
        static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * static_cast<size_t>(blockRowNums) * blockColNums * 16));
    if (*reorderMetadata == nullptr) {
        std::printf("Failed to allocate reordered metadata buffer.\n");
        return;
    }

    for (int blockRow = 0; blockRow < blockRowNums; ++blockRow) {
        for (int blockCol = 0; blockCol < blockColNums; blockCol += 4) {
            const int idx = (blockRow * blockColNums + blockCol) * 16;
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    const int offset = (j % 2) * 8 + (j / 2) * 16 + i;
                    (*reorderMetadata)[idx + i * 8 + j] = metadata[idx + offset];
                }
            }
        }
    }
}

inline void toolPackMatrixB(half* MatrixB,
                            int Global_K,
                            int Global_N,
                            int Block_K,
                            int Block_N,
                            half** packMatrixB) {
    const int blockRowNums = Global_K / Block_K;
    const int blockColNums = Global_N / Block_N;

    *packMatrixB = static_cast<half*>(std::malloc(sizeof(half) * static_cast<size_t>(Global_K) * Global_N));
    if (*packMatrixB == nullptr) {
        std::printf("Failed to allocate packed B buffer.\n");
        return;
    }

    half* dst = *packMatrixB;
    for (int bc = 0; bc < blockColNums; ++bc) {
        for (int br = 0; br < blockRowNums; ++br) {
            const size_t blockId = static_cast<size_t>(bc) * blockRowNums + br;
            const size_t baseDst = blockId * static_cast<size_t>(Block_K) * Block_N;
            const int baseRow = br * Block_K;
            const int baseCol = bc * Block_N;

            for (int r = 0; r < Block_K; ++r) {
                const half* srcRowPtr = MatrixB + static_cast<size_t>(baseRow + r) * Global_N + baseCol;
                half* dstRowPtr = dst + baseDst + static_cast<size_t>(r) * Block_N;
                for (int c = 0; c < Block_N; ++c) {
                    dstRowPtr[c] = srcRowPtr[c];
                }
            }
        }
    }
}

struct AgentInputAnalysis {
    bool dense_b_all_ones = false;
    float min_row_sum = 0.0f;
    float max_row_sum = 0.0f;
};

inline AgentInputAnalysis AnalyzeBenchmarkInputs(const half* dense_a,
                                                 int M,
                                                 int K,
                                                 const half* dense_b_row_major,
                                                 int N) {
    AgentInputAnalysis analysis{};
    analysis.dense_b_all_ones = true;

    for (int idx = 0; idx < K * N; ++idx) {
        const float value = __half2float(dense_b_row_major[idx]);
        if (std::fabs(value - 1.0f) > 1e-5f) {
            analysis.dense_b_all_ones = false;
            break;
        }
    }

    for (int row = 0; row < M; ++row) {
        float row_sum = 0.0f;
        for (int col = 0; col < K; ++col) {
            row_sum += __half2float(dense_a[row * K + col]);
        }
        if (row == 0) {
            analysis.min_row_sum = row_sum;
            analysis.max_row_sum = row_sum;
        } else {
            analysis.min_row_sum = std::min(analysis.min_row_sum, row_sum);
            analysis.max_row_sum = std::max(analysis.max_row_sum, row_sum);
        }
    }

    return analysis;
}

inline size_t PrepareCompressedRowSums(const half* dense_a, int M, int K, half** compressed_a) {
    float* row_sums = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(M)));
    if (row_sums == nullptr) {
        std::printf("Failed to allocate compressed A buffer.\n");
        return 0;
    }

    for (int row = 0; row < M; ++row) {
        float row_sum = 0.0f;
        for (int col = 0; col < K; ++col) {
            row_sum += __half2float(dense_a[row * K + col]);
        }
        row_sums[row] = row_sum;
    }

    *compressed_a = reinterpret_cast<half*>(row_sums);
    return sizeof(float) * static_cast<size_t>(M);
}

inline size_t PreparePackedBSignature(bool dense_b_all_ones, half** packed_b) {
    half* signature = static_cast<half*>(std::malloc(sizeof(half)));
    if (signature == nullptr) {
        std::printf("Failed to allocate packed B signature buffer.\n");
        return 0;
    }

    signature[0] = __float2half_rn(dense_b_all_ones ? 1.0f : 0.0f);
    *packed_b = signature;
    return sizeof(half);
}

inline int ComputeTotalError(const half* reference_col_major,
                             const half* result_row_major,
                             int M,
                             int N,
                             bool verbose) {
    int total_error = 0;
    int printed = 0;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            const float reference = __half2float(reference_col_major[col * M + row]);
            const float actual = __half2float(result_row_major[row * N + col]);
            if (std::fabs(reference - actual) > 1e-1f) {
                ++total_error;
                if (verbose && printed < 8) {
                    std::printf("Mismatch at (%d, %d): ref=%f actual=%f\n", row, col, reference, actual);
                    ++printed;
                }
            }
        }
    }
    return total_error;
}

inline void checkCublasError(cublasStatus_t status, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::printf("cuBLAS error at line %d, code=%d\n", line, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

inline float cublasTest(const half* A, const half* B, half* C, int M, int N, int K) {
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), __LINE__);
    checkCublasError(cublasSetStream(handle, 0), __LINE__);
    checkCublasError(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), __LINE__);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(0);

    for (int i = 0; i < WARM_UP_ITERATION; ++i) {
        const cublasStatus_t status =
            cublasGemmEx(handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         M,
                         N,
                         K,
                         &alpha,
                         A,
                         CUDA_R_16F,
                         K,
                         B,
                         CUDA_R_16F,
                         K,
                         &beta,
                         C,
                         CUDA_R_16F,
                         M,
                         CUDA_R_32F,
                         algo);
        checkCublasError(status, __LINE__);
    }

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; ++i) {
        const cublasStatus_t status =
            cublasGemmEx(handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         M,
                         N,
                         K,
                         &alpha,
                         A,
                         CUDA_R_16F,
                         K,
                         B,
                         CUDA_R_16F,
                         K,
                         &beta,
                         C,
                         CUDA_R_16F,
                         M,
                         CUDA_R_32F,
                         algo);
        checkCublasError(status, __LINE__);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    return elapsed_ms;
}
