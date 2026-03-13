#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void AgentDestroyLaunchContext();

cudaError_t AgentConfigureLaunchContext(const half* dense_a_device,
                                        const half* dense_b_row_major_device,
                                        int M,
                                        int N,
                                        int K,
                                        bool dense_b_all_ones);

cudaError_t SpMM_N2M4_Launch(cudaStream_t stream,
                             const half* compressed_a,
                             const half* packed_b,
                             const uint16_t* metadata,
                             half* C,
                             int M_Global,
                             int N_Global,
                             int K_Global,
                             half* Split_Reduction,
                             int Split_K);
