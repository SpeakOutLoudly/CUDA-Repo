/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

// used for the reduction of result matrix if Split-K is used
// Reduction_Workspace:     (M_Global, N_Global, Split_K),  row major
// C:                       (M_Global, N_Global),           row major
// Each thread deals with 8 output elements, each elements is the sum of Split_K elements
// Each Warp: 32 threads_per_warp * 8 half_per_threads -> 256 half_per_warp
// Each GPU: 108 SM -> 108 warp -> 108*256 = 27648
// GridSize = ceil((M_Global * N_Global) / 256)

#include "TileConfig.h"
#include <cuda_fp16.h>

#define ELEMENT_PER_THREADBLOCK 256

__global__ void SplitK_Reduction(half* C, half* Reduction_Workspace, int M_Global, int N_Global, int Split_K)
{
    const int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * HALF_PER_128bit;
    const int total_elements = M_Global * N_Global;
    if (base_idx >= total_elements)
        return;

    half* c_ptr = C + base_idx;
    half* r_ptr = Reduction_Workspace + base_idx;

    float2 results[HALF_PER_128bit / 2];
#pragma unroll
    for (int j = 0; j < HALF_PER_128bit / 2; j++)
        results[j] = make_float2(0.0f, 0.0f);

    for (int i = 0; i < Split_K; i++) {
        const half2* src = reinterpret_cast<const half2*>(r_ptr + i * total_elements);
#pragma unroll
        for (int j = 0; j < HALF_PER_128bit / 2; j++) {
            float2 value = __half22float2(src[j]);
            results[j].x += value.x;
            results[j].y += value.y;
        }
    }

    half2* dst = reinterpret_cast<half2*>(c_ptr);
#pragma unroll
    for (int j = 0; j < HALF_PER_128bit / 2; j++)
        dst[j] = __floats2half2_rn(results[j].x, results[j].y);
}
