// Copyright (C) 2026 Silvet.icu
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Shared by rowcol_scaled_gemm_bf16.cu and attention_bf16.cu.
// See docs/gemm-kernel.md.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <mma.h>

// stored value -> fp32 (exact for all three)
static __device__ __forceinline__ float bf2f(__nv_bfloat16 v) {
    return __uint_as_float(static_cast<unsigned int>(__bfloat16_as_ushort(v)) << 16);
}

static __device__ __forceinline__ float bf2f(float v) { return v; }

static __device__ __forceinline__ float bf2f(c10::Float8_e4m3fn v) {
    return static_cast<float>(v);
}

// power of two mapping |max| into [1024, 2048)
static __device__ __forceinline__ float pow2_scale(float mx) {
    if (!(mx > 0.f)) return 1.f;
    float ex = floorf(__log2f(2048.0f / mx));
    if (ex > 127.f) ex = 127.f;
    return exp2f(ex);
}

// fp16 operands, fp32 accumulation; a 16-row strip per call

// C[16][N] = A[16][K] @ B[N][K]^T   (B row-major, i.e. the weight layout)
static __device__ __forceinline__ void gemm_16_nt(const half* A, int ldA,
                                                  const half* B, int ldB,
                                                  int K, int N,
                                                  float* C, int ldC) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    for (int n = 0; n < N; n += 16) {
        wmma::fill_fragment(cf, 0.f);
        for (int k = 0; k < K; k += 16) {
            wmma::load_matrix_sync(af, A + k, ldA);
            wmma::load_matrix_sync(bf, B + n * ldB + k, ldB);
            wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(C + n, cf, ldC, wmma::mem_row_major);
    }
}

// C[16][N] += A[16][K] @ B[K][N]    (B row-major)
static __device__ __forceinline__ void gemm_16_nn_acc(const half* A, int ldA,
                                                      const half* B, int ldB,
                                                      int K, int N,
                                                      float* C, int ldC) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    for (int n = 0; n < N; n += 16) {
        wmma::load_matrix_sync(cf, C + n, ldC, wmma::mem_row_major);
        for (int k = 0; k < K; k += 16) {
            wmma::load_matrix_sync(af, A + k, ldA);
            wmma::load_matrix_sync(bf, B + k * ldB + n, ldB);
            wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(C + n, cf, ldC, wmma::mem_row_major);
    }
}
