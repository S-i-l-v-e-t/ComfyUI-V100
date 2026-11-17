// Copyright (C) 2026 Silvet.icu
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <climits>
#include <vector>

// bf2f / pow2_scale are shared with the attention kernel.
// See docs/gemm-kernel.md.
#include "scaled_common.cuh"

// Only e4m3 is wired up; e5m2 has 3 mantissa bits and no such exact widening.
static __host__ inline bool is_fp8(const torch::Tensor& t) {
    return t.scalar_type()==at::kFloat8_e4m3fn;
}
// Accepted storage types for an operand: bf16/fp32, or (weights only) fp8 e4m3.
static __host__ inline bool is_act_dtype(c10::ScalarType st) {
    return st==at::kBFloat16 || st==at::kFloat;
}
static __host__ inline bool is_w_dtype(c10::ScalarType st) {
    return is_act_dtype(st) || st==at::kFloat8_e4m3fn;
}
// Raises the matching kernel for whichever of the three storage types shows up.
static __host__ inline const void* in_ptr(const torch::Tensor& t) {
    return t.const_data_ptr();
}

// A row per CTA. Reduce locally in registers and warp/shared memory; no
// per-element global atomics. BT rows correspond to B columns.
// `ld` is the padded tile width: the input is read at the real `cols` stride,
// the padded tile is written at `ld`, so padding never needs its own pass.
template <typename In>
__device__ __forceinline__ void pack_one_row(const In* __restrict__ x, half* __restrict__ out,
                                            float* __restrict__ maxima, int* __restrict__ unsafe,
                                            int row, int cols, int ld, bool allow_wide_range, int* __restrict__ bad_rows) {
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    unsigned int mn=0x7f800000u, mx=0u, flags=0u;
    for (int col=tid;col<cols;col+=256) {
        unsigned int u=__float_as_uint(bf2f(x[int64_t(row)*cols+col])) & 0x7fffffffu;
        if (u>=0x7f800000u) flags|=1u;
        else if (u) {
            mn=min(mn,u); mx=max(mx,u);
            if (u<0x00800000u) flags|=2u;
        }
    }
    for (int d=16;d>0;d>>=1) {
        mn=min(mn,__shfl_down_sync(0xffffffffu,mn,d));
        mx=max(mx,__shfl_down_sync(0xffffffffu,mx,d));
        flags|=__shfl_down_sync(0xffffffffu,flags,d);
    }
    __shared__ unsigned int smn[8],smx[8],sflags[8];
    __shared__ float rowmax;
    __shared__ float rowscale;
    if (lane==0) { smn[warp]=mn; smx[warp]=mx; sflags[warp]=flags; }
    __syncthreads();
    if (warp==0) {
        mn=lane<8?smn[lane]:0x7f800000u;
        mx=lane<8?smx[lane]:0u;
        flags=lane<8?sflags[lane]:0u;
        for (int d=16;d>0;d>>=1) {
            mn=min(mn,__shfl_down_sync(0xffffffffu,mn,d));
            mx=max(mx,__shfl_down_sync(0xffffffffu,mx,d));
            flags|=__shfl_down_sync(0xffffffffu,flags,d);
        }
        if (lane==0) {
            rowmax=mx?__uint_as_float(mx):0.f;
            rowscale=pow2_scale(rowmax);
            maxima[row]=rowscale;
            bool is_bad=flags || (!allow_wide_range && mx && double(__uint_as_float(mx))>double(__uint_as_float(mn))*1073217536.0);
            if (bad_rows) bad_rows[row]=int(is_bad);
            if (is_bad) atomicOr(unsafe,1);
        }
    }
    __syncthreads();
    for (int col=tid;col<cols;col+=256) {
        out[int64_t(row)*ld+col]=__float2half_rn(bf2f(x[int64_t(row)*cols+col])*rowscale);
    }
}

// Same as pack_one_row but the whole row is handled by a single warp, so short
// rows (a few hundred columns) skip the block-wide reduction and its two
// __syncthreads. min/max/or are order independent, so the result is identical.
template <typename In>
__device__ __forceinline__ void pack_one_row_warp(const In* __restrict__ x, half* __restrict__ out,
                                                  float* __restrict__ maxima, int* __restrict__ unsafe,
                                                  int row, int cols, int ld, bool allow_wide_range, int* __restrict__ bad_rows) {
    int lane=threadIdx.x&31;
    unsigned int mn=0x7f800000u, mx=0u, flags=0u;
    for (int col=lane;col<cols;col+=32) {
        unsigned int u=__float_as_uint(bf2f(x[int64_t(row)*cols+col])) & 0x7fffffffu;
        if (u>=0x7f800000u) flags|=1u;
        else if (u) {
            mn=min(mn,u); mx=max(mx,u);
            if (u<0x00800000u) flags|=2u;
        }
    }
    #pragma unroll
    for (int d=16;d>0;d>>=1) {
        mn=min(mn,__shfl_down_sync(0xffffffffu,mn,d));
        mx=max(mx,__shfl_down_sync(0xffffffffu,mx,d));
        flags|=__shfl_down_sync(0xffffffffu,flags,d);
    }
    float rowscale=1.f;
    if (lane==0) {
        float rowmax=mx?__uint_as_float(mx):0.f;
        rowscale=pow2_scale(rowmax);
        maxima[row]=rowscale;
        bool is_bad=flags || (!allow_wide_range && mx && double(__uint_as_float(mx))>double(__uint_as_float(mn))*1073217536.0);
        if (bad_rows) bad_rows[row]=int(is_bad);
        if (is_bad) atomicOr(unsafe,1);
    }
    rowscale=__shfl_sync(0xffffffffu,rowscale,0);
    for (int col=lane;col<cols;col+=32) {
        out[int64_t(row)*ld+col]=__float2half_rn(bf2f(x[int64_t(row)*cols+col])*rowscale);
    }
}

// Warps per block for the short-row pack kernels (bandwidth bound, not warp limited).
static int pack_warps() { return 8; }

template <typename In>
__global__ void row_pack_ws(const In* __restrict__ x, half* __restrict__ out,
                            float* __restrict__ maxima, int* __restrict__ unsafe,
                            int rows, int cols, int ld, bool allow_wide_range, int* __restrict__ bad_rows) {
    int row=int(blockIdx.x)*(int(blockDim.x)>>5)+(int(threadIdx.x>>5));
    if (row<rows) pack_one_row_warp(x,out,maxima,unsafe,row,cols,ld,allow_wide_range,bad_rows);
}

template <typename InA, typename InW>
__global__ void row_pack2_ws(const InA* __restrict__ xa, half* __restrict__ oa,
                             float* __restrict__ ma, int* __restrict__ ua, int* __restrict__ ba,
                             const InW* __restrict__ xb, half* __restrict__ ob,
                             float* __restrict__ mb, int* __restrict__ ub, int* __restrict__ bb,
                             int rows_a, int rows_b, int cols, int ld, bool allow_wide_range) {
    int gid=int(blockIdx.x)*(int(blockDim.x)>>5)+(int(threadIdx.x>>5));
    if (gid<rows_a) pack_one_row_warp(xa,oa,ma,ua,gid,cols,ld,allow_wide_range,ba);
    else if (gid<rows_a+rows_b) pack_one_row_warp(xb,ob,mb,ub,gid-rows_a,cols,ld,allow_wide_range,bb);
}

template <typename In>
__global__ void row_pack(const In* __restrict__ x, half* __restrict__ out,
                         float* __restrict__ maxima, int* __restrict__ unsafe,
                         int rows, int cols, int ld, bool allow_wide_range, int* __restrict__ bad_rows) {
    pack_one_row(x, out, maxima, unsafe, int(blockIdx.x), cols, ld, allow_wide_range, bad_rows);
}

// A and B packing share one launch; the per-row body is still pack_one_row.
template <typename InA, typename InW>
__global__ void row_pack2(const InA* __restrict__ xa, half* __restrict__ oa,
                          float* __restrict__ ma, int* __restrict__ ua, int* __restrict__ ba,
                          const InW* __restrict__ xb, half* __restrict__ ob,
                          float* __restrict__ mb, int* __restrict__ ub, int* __restrict__ bb,
                          int rows_a, int cols, int ld, bool allow_wide_range) {
    int gid=int(blockIdx.x);
    if (gid<rows_a) pack_one_row(xa, oa, ma, ua, gid, cols, ld, allow_wide_range, ba);
    else pack_one_row(xb, ob, mb, ub, gid-rows_a, cols, ld, allow_wide_range, bb);
}

// Packs A and W in one launch. Returns {ha,ma,ua,hb,mb,ub,[marksA,marksB]}.
std::vector<torch::Tensor> pack_pair(torch::Tensor a, torch::Tensor w, bool allow_wide_range,
                                     bool return_marks, int64_t m_pad, int64_t n_pad, int64_t k_pad) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && a.device()==w.device(),"same device expected");
    TORCH_CHECK(is_act_dtype(a.scalar_type()) &&
                (a.scalar_type()==w.scalar_type() || w.scalar_type()==at::kFloat8_e4m3fn),
                "expected bf16/fp32 A with matching B, or fp8 e4m3 weights");
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && a.dim()==2 && w.dim()==2,"contiguous 2D expected");
    TORCH_CHECK(a.size(1)==w.size(1),"contraction dim mismatch");
    int64_t m=a.size(0), n=w.size(0), k=a.size(1);
    int64_t mp=m_pad>0?m_pad:m, np=n_pad>0?n_pad:n, kp=k_pad>0?k_pad:k;
    TORCH_CHECK(mp>=m && np>=n && kp>=k,"invalid padded shape");
    TORCH_CHECK(mp<=INT_MAX && np<=INT_MAX && kp<=INT_MAX && m+n<=INT_MAX,"shape too large");
    const c10::cuda::CUDAGuard guard(a.device());
    auto ha=torch::empty({mp,kp},a.options().dtype(at::kHalf));
    auto hb=torch::empty({np,kp},a.options().dtype(at::kHalf));
    if (kp>k) { ha.narrow(1,k,kp-k).zero_(); hb.narrow(1,k,kp-k).zero_(); }
    if (mp>m) ha.narrow(0,m,mp-m).zero_();
    if (np>n) hb.narrow(0,n,np-n).zero_();
    auto ma=torch::ones({mp},a.options().dtype(at::kFloat));
    auto mb=torch::ones({np},a.options().dtype(at::kFloat));
    auto ua=torch::zeros({},a.options().dtype(at::kInt));
    auto ub=torch::zeros({},a.options().dtype(at::kInt));
    auto marksA=return_marks?torch::zeros({mp},a.options().dtype(at::kInt)):torch::Tensor();
    auto marksB=return_marks?torch::zeros({np},a.options().dtype(at::kInt)):torch::Tensor();
    auto launch=[&](auto* ap, auto* wp){
        using InA=typename std::remove_const<typename std::remove_pointer<decltype(ap)>::type>::type;
        using InW=typename std::remove_const<typename std::remove_pointer<decltype(wp)>::type>::type;
        if (k<=2048) {
            int wb=pack_warps();
            row_pack2_ws<InA,InW><<<int((m+n+wb-1)/wb),wb*32,0,at::cuda::getCurrentCUDAStream()>>>(
                ap,reinterpret_cast<half*>(ha.data_ptr<at::Half>()),ma.data_ptr<float>(),ua.data_ptr<int>(),return_marks?marksA.data_ptr<int>():nullptr,
                wp,reinterpret_cast<half*>(hb.data_ptr<at::Half>()),mb.data_ptr<float>(),ub.data_ptr<int>(),return_marks?marksB.data_ptr<int>():nullptr,
                int(m),int(n),int(k),kp?int(kp):int(k),allow_wide_range);
        } else {
            row_pack2<InA,InW><<<int(m+n),256,0,at::cuda::getCurrentCUDAStream()>>>(
                ap,reinterpret_cast<half*>(ha.data_ptr<at::Half>()),ma.data_ptr<float>(),ua.data_ptr<int>(),return_marks?marksA.data_ptr<int>():nullptr,
                wp,reinterpret_cast<half*>(hb.data_ptr<at::Half>()),mb.data_ptr<float>(),ub.data_ptr<int>(),return_marks?marksB.data_ptr<int>():nullptr,
                int(m),int(k),kp?int(kp):int(k),allow_wide_range);
        }
    };
    if (is_fp8(w)) {
        auto* wp=reinterpret_cast<const c10::Float8_e4m3fn*>(w.data_ptr<c10::Float8_e4m3fn>());
        if (a.scalar_type()==at::kFloat)
            launch(reinterpret_cast<const float*>(a.data_ptr<float>()),wp);
        else
            launch(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),wp);
    } else if (a.scalar_type()==at::kFloat)
        launch(reinterpret_cast<const float*>(a.data_ptr<float>()),reinterpret_cast<const float*>(w.data_ptr<float>()));
    else
        launch(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_marks) return {ha,ma,ua,hb,mb,ub,marksA,marksB};
    return {ha,ma,ua,hb,mb,ub};
}

// Writes the padded fp16 tile [rows_pad, cols_pad]; padding is part of the same pass.
std::vector<torch::Tensor> pack_rows(torch::Tensor x, bool allow_wide_range, bool return_marks, int64_t rows_pad, int64_t cols_pad) {
    TORCH_CHECK(x.is_cuda() && is_w_dtype(x.scalar_type()) && x.is_contiguous() && x.dim()==2 && x.size(0)>0 && x.size(1)>0,"expected nonempty contiguous bf16/fp32/(fp8) CUDA matrix");
    TORCH_CHECK(x.size(0)<=INT_MAX && x.size(1)<=INT_MAX,"shape too large");
    int64_t rows=x.size(0),cols=x.size(1);
    int64_t rp=rows_pad>0?rows_pad:rows, cp=cols_pad>0?cols_pad:cols;
    TORCH_CHECK(rp>=rows && cp>=cols && rp<=INT_MAX && cp<=INT_MAX,"invalid padded shape");
    const c10::cuda::CUDAGuard guard(x.device());
    auto h=torch::empty({rp,cp},x.options().dtype(at::kHalf));
    if (cp>cols) h.narrow(1,cols,cp-cols).zero_();
    if (rp>rows) h.narrow(0,rows,rp-rows).zero_();
    // Padded rows stay at scale 1 and are never marked unsafe.
    auto maxima=torch::ones({rp},x.options().dtype(at::kFloat));
    auto unsafe=torch::zeros({},x.options().dtype(at::kInt));
    auto marks=return_marks ? torch::zeros({rp},x.options().dtype(at::kInt)) : torch::Tensor();
    if (cols<=2048) {
        int wb=pack_warps();
        if (x.scalar_type()==at::kFloat)
            row_pack_ws<float><<<int((rows+wb-1)/wb),wb*32,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),reinterpret_cast<half*>(h.data_ptr<at::Half>()),maxima.data_ptr<float>(),unsafe.data_ptr<int>(),int(rows),int(cols),int(cp),allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
        else if (x.scalar_type()==at::kFloat8_e4m3fn)
            row_pack_ws<c10::Float8_e4m3fn><<<int((rows+wb-1)/wb),wb*32,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const c10::Float8_e4m3fn*>(x.data_ptr<c10::Float8_e4m3fn>()),reinterpret_cast<half*>(h.data_ptr<at::Half>()),maxima.data_ptr<float>(),unsafe.data_ptr<int>(),int(rows),int(cols),int(cp),allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
        else
            row_pack_ws<__nv_bfloat16><<<int((rows+wb-1)/wb),wb*32,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),reinterpret_cast<half*>(h.data_ptr<at::Half>()),maxima.data_ptr<float>(),unsafe.data_ptr<int>(),int(rows),int(cols),int(cp),allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
    } else {
        if (x.scalar_type()==at::kFloat)
            row_pack<float><<<int(rows),256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),reinterpret_cast<half*>(h.data_ptr<at::Half>()),maxima.data_ptr<float>(),unsafe.data_ptr<int>(),int(rows),int(cols),int(cp),allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
        else if (x.scalar_type()==at::kFloat8_e4m3fn)
            row_pack<c10::Float8_e4m3fn><<<int(rows),256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const c10::Float8_e4m3fn*>(x.data_ptr<c10::Float8_e4m3fn>()),reinterpret_cast<half*>(h.data_ptr<at::Half>()),maxima.data_ptr<float>(),unsafe.data_ptr<int>(),int(rows),int(cols),int(cp),allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
        else
            row_pack<__nv_bfloat16><<<int(rows),256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),reinterpret_cast<half*>(h.data_ptr<at::Half>()),maxima.data_ptr<float>(),unsafe.data_ptr<int>(),int(rows),int(cols),int(cp),allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_marks) return {h,maxima,unsafe,marks};
    return {h,maxima,unsafe};
}

// Batched restore: cf/c are [batch, m, n] row-major, ma is per global row, mb per
// output column (mb_stride = 0 when the weight is broadcast).
__global__ void restore_batched(const float* __restrict__ cf,__nv_bfloat16* __restrict__ c,const float* __restrict__ ma,const float* __restrict__ mb,const float* __restrict__ bias,int64_t total,int m,int n,int mb_stride) {
    int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (idx<total) {
        int64_t r=idx/n;
        int col=int(idx%n);
        double v=double(cf[idx])/(double(ma[r])*double(mb[int64_t(r/m)*mb_stride+col]));
        if (bias) v+=double(bias[col]);
        c[idx]=__float2bfloat16_rn(float(v));
    }
}

// Same math, fp32 store (attention needs fp32 scores).
__global__ void restore_batched_f32(const float* __restrict__ cf,float* __restrict__ c,const float* __restrict__ ma,const float* __restrict__ mb,const float* __restrict__ bias,int64_t total,int m,int n,int mb_stride) {
    int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (idx<total) {
        int64_t r=idx/n;
        int col=int(idx%n);
        double v=double(cf[idx])/(double(ma[r])*double(mb[int64_t(r/m)*mb_stride+col]));
        if (bias) v+=double(bias[col]);
        c[idx]=float(v);
    }
}

// One cuBLAS call for the whole batch; a is [batch*m, k], bt is [batch*n, k].
// stride_b = 0 reuses a single packed weight for every slice.
torch::Tensor gemm_restore_batched(torch::Tensor a,torch::Tensor bt,torch::Tensor ma,torch::Tensor mb,int64_t batch,int64_t m,int64_t n,int64_t mb_stride,c10::optional<torch::Tensor> bias,int64_t scratch_bytes,bool out_fp32) {
    TORCH_CHECK(a.is_cuda() && bt.is_cuda() && a.device()==bt.device() && a.scalar_type()==at::kHalf && bt.scalar_type()==at::kHalf && a.is_contiguous() && bt.is_contiguous(),"expected same-device contiguous half matrices");
    TORCH_CHECK(a.dim()==2 && bt.dim()==2 && a.size(1)==bt.size(1),"invalid shape");
    TORCH_CHECK(batch>0 && m>0 && n>0 && a.size(0)>=batch*m && bt.size(0)>=(mb_stride?batch*n:n),"invalid dimensions");
    TORCH_CHECK(a.size(0)<=INT_MAX && bt.size(0)<=INT_MAX && a.size(1)<=INT_MAX && m<=INT_MAX && n<=INT_MAX,"shape too large");
    const c10::cuda::CUDAGuard guard(a.device());
    int k=int(a.size(1));
    int64_t sa=int64_t(m)*k, sb=int64_t(n)*k, sc=int64_t(m)*n;
    const float* bp=nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->device()==a.device() && bias->scalar_type()==at::kFloat && bias->is_contiguous() && bias->numel()==n,"expected contiguous float32 bias of length n");
        bp=bias->data_ptr<float>();
    }
    auto c=torch::empty({batch,m,n},a.options().dtype(out_fp32?at::kFloat:at::kBFloat16));
    int64_t nb=batch;
    if (scratch_bytes>0) { int64_t cap=scratch_bytes/(sc*4); if (cap<1) cap=1; if (cap<nb) nb=cap; }
    auto cf=torch::empty({nb*sc},a.options().dtype(at::kFloat));
    const float alpha=1.f,beta=0.f;
    auto h=at::cuda::getCurrentCUDABlasHandle();
    const half* ap=reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* bpk=reinterpret_cast<const half*>(bt.data_ptr<at::Half>());
    __nv_bfloat16* cp=out_fp32?nullptr:reinterpret_cast<__nv_bfloat16*>(c.data_ptr<at::BFloat16>());
    for (int64_t b0=0; b0<batch; b0+=nb) {
        int cnt=int(std::min<int64_t>(nb,batch-b0));
        int64_t sbt=mb_stride?sb:0;
        auto stat=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,int(m),k,&alpha,
            bpk+b0*sbt,CUDA_R_16F,k,sbt,
            ap+b0*sa,CUDA_R_16F,k,sa,
            &beta,cf.data_ptr(),CUDA_R_32F,n,sc,cnt,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        TORCH_CHECK(stat==CUBLAS_STATUS_SUCCESS,"cublasGemmStridedBatchedEx failed: ",int(stat));
        int64_t total=int64_t(cnt)*sc;
        if (out_fp32)
            restore_batched_f32<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(cf.data_ptr<float>(),c.data_ptr<float>()+b0*sc,ma.data_ptr<float>()+b0*m,mb.data_ptr<float>()+b0*mb_stride,bp,total,int(m),int(n),int(mb_stride));
        else
            restore_batched<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(cf.data_ptr<float>(),cp+b0*sc,ma.data_ptr<float>()+b0*m,mb.data_ptr<float>()+b0*mb_stride,bp,total,int(m),int(n),int(mb_stride));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return c;
}

// Reads the accumulator at the padded stride, rescales, adds bias, rounds once.
__global__ void restore_rows_cols(const float* __restrict__ cf,__nv_bfloat16* __restrict__ c,const float* __restrict__ ma,const float* __restrict__ mb,const float* __restrict__ bias,int64_t total,int n_out,int n_pad) {
    int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (idx<total) {
        int r=int(idx/n_out),col=int(idx%n_out);
        double v=double(cf[int64_t(r)*n_pad+col])/(double(ma[r])*double(mb[col]));
        if (bias) v+=double(bias[col]);
        c[idx]=__float2bfloat16_rn(float(v));
    }
}

__global__ void restore_rows_cols_f32(const float* __restrict__ cf,float* __restrict__ c,const float* __restrict__ ma,const float* __restrict__ mb,const float* __restrict__ bias,int64_t total,int n_out,int n_pad) {
    int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (idx<total) {
        int r=int(idx/n_out),col=int(idx%n_out);
        double v=double(cf[int64_t(r)*n_pad+col])/(double(ma[r])*double(mb[col]));
        if (bias) v+=double(bias[col]);
        c[idx]=float(v);
    }
}

torch::Tensor gemm_restore(torch::Tensor a,torch::Tensor bt,torch::Tensor ma,torch::Tensor mb,int64_t m_out,int64_t n_out,c10::optional<torch::Tensor> bias,int64_t scratch_bytes,bool out_fp32) {
    TORCH_CHECK(a.is_cuda() && bt.is_cuda() && a.device()==bt.device() && a.scalar_type()==at::kHalf && bt.scalar_type()==at::kHalf && a.is_contiguous() && bt.is_contiguous(),"expected same-device contiguous half matrices");
    TORCH_CHECK(a.dim()==2 && bt.dim()==2 && a.size(1)==bt.size(1) && ma.numel()==a.size(0) && mb.numel()==bt.size(0),"invalid shape");
    TORCH_CHECK(a.size(0)>0 && bt.size(0)>0 && a.size(1)>0 && a.size(0)<=INT_MAX && bt.size(0)<=INT_MAX && a.size(1)<=INT_MAX,"invalid dimension");
    const c10::cuda::CUDAGuard guard(a.device());
    int m=int(a.size(0)), n=int(bt.size(0)), k=int(a.size(1));
    int mo=m_out>0?int(m_out):m, no=n_out>0?int(n_out):n;
    TORCH_CHECK(mo<=m && no<=n,"crop exceeds padded shape");
    const float* bp=nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->device()==a.device() && bias->scalar_type()==at::kFloat && bias->is_contiguous() && bias->numel()==no,"expected contiguous float32 bias of length n_out");
        bp=bias->data_ptr<float>();
    }
    // cuBLAS needs an fp32 buffer before the rescale, so wide outputs are produced in
// row blocks (K, the reduction dim, is untouched).
    int64_t chunk=m;
    if (scratch_bytes>0) { int64_t cap=scratch_bytes/(int64_t(n)*4); if (cap<1) cap=1; if (cap<chunk) chunk=cap; }
    auto cf=torch::empty({chunk,n},a.options().dtype(at::kFloat));
    auto c=torch::empty({mo,no},a.options().dtype(out_fp32?at::kFloat:at::kBFloat16));
    const float alpha=1.f,beta=0.f;
    auto h=at::cuda::getCurrentCUDABlasHandle();
    half* ap=reinterpret_cast<half*>(a.data_ptr<at::Half>());
    __nv_bfloat16* cp=out_fp32?nullptr:reinterpret_cast<__nv_bfloat16*>(c.data_ptr<at::BFloat16>());
    for (int64_t r0=0; r0<mo; r0+=chunk) {
        int rows=int(std::min<int64_t>(chunk,mo-r0));
        // a is column-major k x m with ld k, so logical row r0 is column r0 and
        // starts at r0*k. c is row-major [mo,no], so its block starts at r0*no.
        auto stat=cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,rows,k,&alpha,bt.data_ptr(),CUDA_R_16F,k,ap+r0*k,CUDA_R_16F,k,&beta,cf.data_ptr(),CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        TORCH_CHECK(stat==CUBLAS_STATUS_SUCCESS,"cublasGemmEx failed: ",int(stat));
        int64_t total=int64_t(rows)*no;
        if (out_fp32)
            restore_rows_cols_f32<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(cf.data_ptr<float>(),c.data_ptr<float>()+r0*no,ma.data_ptr<float>()+r0,mb.data_ptr<float>(),bp,total,no,n);
        else
            restore_rows_cols<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(cf.data_ptr<float>(),cp+r0*no,ma.data_ptr<float>()+r0,mb.data_ptr<float>(),bp,total,no,n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return c;
}
// Each warp covers 32 columns, with 8 warps cooperatively scanning a K tile.
template <typename In>
__global__ void col_partial_stats(const In* __restrict__ b, unsigned int* __restrict__ partial, int k, int n) {
    int lane=threadIdx.x & 31, warp=threadIdx.x >> 5;
    int col=int(blockIdx.x)*32+lane, start=int(blockIdx.y)*256;
    unsigned int mn=0x7f800000u, mx=0u, flags=0u;
    if (col<n) for (int r=start+warp; r<k && r<start+256; r+=8) {
        unsigned int u=__float_as_uint(bf2f(b[int64_t(r)*n+col])) & 0x7fffffffu;
        if (u>=0x7f800000u) flags|=1u;
        else if (u) { mn=min(mn,u); mx=max(mx,u); if (u<0x00800000u) flags|=2u; }
    }
    __shared__ unsigned int smn[8][32], smx[8][32], sflags[8][32];
    smn[warp][lane]=mn; smx[warp][lane]=mx; sflags[warp][lane]=flags;
    __syncthreads();
    if (warp==0 && col<n) {
        mn=0x7f800000u; mx=0u; flags=0u;
        #pragma unroll
        for (int w=0;w<8;++w) { mn=min(mn,smn[w][lane]); mx=max(mx,smx[w][lane]); flags|=sflags[w][lane]; }
        int64_t base=int64_t(blockIdx.y)*3*n+col;
        partial[base]=mn; partial[base+n]=mx; partial[base+2LL*n]=flags;
    }
}

__global__ void finish_cols(const unsigned int* __restrict__ partial, float* __restrict__ maxima, int* __restrict__ unsafe, int chunks, int n, bool allow_wide_range, int* __restrict__ bad_cols) {
    int col=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if (col>=n) return;
    unsigned int mn=0x7f800000u,mx=0u,flags=0u;
    for (int i=0;i<chunks;++i) {
        int64_t base=int64_t(i)*3*n+col;
        mn=min(mn,partial[base]); mx=max(mx,partial[base+n]); flags|=partial[base+2LL*n];
    }
    maxima[col]=mx?__uint_as_float(mx):0.f;
    {
        // store the scale itself instead of the max
        maxima[col]=pow2_scale(maxima[col]);
    }
    bool is_bad=flags || (!allow_wide_range && mx && double(__uint_as_float(mx))>double(__uint_as_float(mn))*1073217536.0);
    if (bad_cols) bad_cols[col]=int(is_bad);
    if (is_bad) atomicOr(unsafe,1);
}

// Batched column packing for b [batch, k, n] (the non-transposed operand, i.e.
// torch.bmm and the dW of a batched linear). Same math as the 2D column path;
// the only change is that the column id is global over batch * n and the slice
// offset is added to the base pointer.
template <typename In>
__global__ void col_partial_stats_batched(const In* __restrict__ b, unsigned int* __restrict__ partial, int k, int n, int64_t bn) {
    int lane=threadIdx.x & 31, warp=threadIdx.x >> 5;
    int64_t gb=int64_t(blockIdx.x)*32+lane, start=int64_t(blockIdx.y)*256;
    unsigned int mn=0x7f800000u, mx=0u, flags=0u;
    if (gb<bn) {
        const In* p=b+(gb/n)*int64_t(k)*n+(gb%n);
        for (int64_t r=start+warp; r<k && r<start+256; r+=8) {
            unsigned int u=__float_as_uint(bf2f(p[r*n])) & 0x7fffffffu;
            if (u>=0x7f800000u) flags|=1u;
            else if (u) { mn=min(mn,u); mx=max(mx,u); if (u<0x00800000u) flags|=2u; }
        }
    }
    __shared__ unsigned int smn[8][32], smx[8][32], sflags[8][32];
    smn[warp][lane]=mn; smx[warp][lane]=mx; sflags[warp][lane]=flags;
    __syncthreads();
    if (warp==0 && gb<bn) {
        mn=0x7f800000u; mx=0u; flags=0u;
        #pragma unroll
        for (int w=0;w<8;++w) { mn=min(mn,smn[w][lane]); mx=max(mx,smx[w][lane]); flags|=sflags[w][lane]; }
        int64_t base=int64_t(blockIdx.y)*3*bn+gb;
        partial[base]=mn; partial[base+bn]=mx; partial[base+2LL*bn]=flags;
    }
}

__global__ void finish_cols_batched(const unsigned int* __restrict__ partial, float* __restrict__ maxima, int* __restrict__ unsafe, int chunks, int64_t bn, bool allow_wide_range, int* __restrict__ bad) {
    int64_t gb=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (gb>=bn) return;
    unsigned int mn=0x7f800000u,mx=0u,flags=0u;
    for (int i=0;i<chunks;++i) {
        int64_t base=int64_t(i)*3*bn+gb;
        mn=min(mn,partial[base]); mx=max(mx,partial[base+bn]); flags|=partial[base+2LL*bn];
    }
    float cmax=mx?__uint_as_float(mx):0.f;
    maxima[gb]=pow2_scale(cmax);
    bool is_bad=flags || (!allow_wide_range && mx && double(__uint_as_float(mx))>double(__uint_as_float(mn))*1073217536.0);
    if (bad) bad[gb]=int(is_bad);
    if (is_bad) atomicOr(unsafe,1);
}

template <typename In>
__global__ void pack_cols_transpose_batched(const In* __restrict__ b, half* __restrict__ out, const float* __restrict__ maxima, int k, int n, int ld, int64_t bn) {
    __shared__ half tile[32][33];
    int tx=threadIdx.x, ty=threadIdx.y;
    int64_t gcol=int64_t(blockIdx.x)*32+tx;
    int64_t row=int64_t(blockIdx.y)*32+ty;
    if (gcol<bn) {
        const In* p=b+(gcol/n)*int64_t(k)*n+(gcol%n);
        #pragma unroll
        for (int j=0;j<32;j+=8) {
            if (row+j<k) tile[ty+j][tx]=__float2half_rn(bf2f(p[(row+j)*n])*maxima[gcol]);
        }
    }
    __syncthreads();
    int64_t gout=int64_t(blockIdx.x)*32+ty;
    int outcol=int(blockIdx.y)*32+tx;
    #pragma unroll
    for (int j=0;j<32;j+=8) {
        if (gout+j<bn && outcol<k) {
            int64_t r=gout+j;
            out[(r/n)*int64_t(n)*ld+(r%n)*ld+outcol]=tile[tx][ty+j];
        }
    }
}


// `ld` is the padded tile width, so the transposed tile is written straight into
// instruction-sized storage; bounds still use the real k.
template <typename In>
__global__ void pack_cols_transpose(const In* __restrict__ b, half* __restrict__ out, const float* __restrict__ maxima, int k, int n, int ld) {
    __shared__ half tile[32][33];
    int tx=threadIdx.x, ty=threadIdx.y;
    int col=int(blockIdx.x)*32+tx, row=int(blockIdx.y)*32+ty;
    #pragma unroll
    for (int j=0;j<32;j+=8) {
        if (col<n && row+j<k) tile[ty+j][tx]=__float2half_rn(bf2f(b[int64_t(row+j)*n+col])*maxima[col]);
    }
    __syncthreads();
    int outrow=int(blockIdx.x)*32+ty, outcol=int(blockIdx.y)*32+tx;
    #pragma unroll
    for (int j=0;j<32;j+=8) {
        if (outrow+j<n && outcol<k) out[int64_t(outrow+j)*ld+outcol]=tile[tx][ty+j];
    }
}

std::vector<torch::Tensor> pack_cols(torch::Tensor b, bool allow_wide_range, bool return_marks, int64_t n_pad, int64_t k_pad) {
    TORCH_CHECK(b.is_cuda() && is_w_dtype(b.scalar_type()) && b.is_contiguous() && b.dim()==2 && b.size(0)>0 && b.size(1)>0,"expected nonempty contiguous bf16/fp32/(fp8) CUDA matrix");
    TORCH_CHECK(b.size(0)<=INT_MAX && b.size(1)<=INT_MAX && (b.size(0)+255)/256<=65535 && (b.size(0)+31)/32<=65535,"shape too large");
    int64_t k64=b.size(0),n64=b.size(1);
    int64_t kp=k_pad>0?k_pad:k64, np=n_pad>0?n_pad:n64;
    TORCH_CHECK(kp>=k64 && np>=n64 && kp<=INT_MAX && np<=INT_MAX,"invalid padded shape");
    TORCH_CHECK((kp+255)/256<=65535 && (kp+31)/32<=65535 && (np+31)/32<=65535,"shape too large");
    const c10::cuda::CUDAGuard guard(b.device());
    int k=int(k64),n=int(n64),chunks=(k+255)/256;
    auto out=torch::empty({np,kp},b.options().dtype(at::kHalf));
    if (kp>k64) out.narrow(1,k64,kp-k64).zero_();
    if (np>n64) out.narrow(0,n64,np-n64).zero_();
    // Padded columns stay at scale 1 and are never marked unsafe.
    auto maxima=torch::ones({np},b.options().dtype(at::kFloat));
    auto unsafe=torch::zeros({},b.options().dtype(at::kInt));
    auto marks=return_marks ? torch::zeros({np},b.options().dtype(at::kInt)) : torch::Tensor();
    auto partial=torch::empty({chunks,3,n},b.options().dtype(at::kInt));
    auto stream=at::cuda::getCurrentCUDAStream();
    auto ptr=reinterpret_cast<unsigned int*>(partial.data_ptr<int>());
    auto launch=[&](auto* src){
        using In=typename std::remove_const<typename std::remove_pointer<decltype(src)>::type>::type;
        col_partial_stats<In><<<dim3((n+31)/32,chunks),256,0,stream>>>(src,ptr,k,n);
        finish_cols<<<(n+255)/256,256,0,stream>>>(ptr,maxima.data_ptr<float>(),unsafe.data_ptr<int>(),chunks,n,allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
        pack_cols_transpose<In><<<dim3((n+31)/32,(k+31)/32),dim3(32,8),0,stream>>>(src,reinterpret_cast<half*>(out.data_ptr<at::Half>()),maxima.data_ptr<float>(),k,n,int(kp));
    };
    if (b.scalar_type()==at::kFloat) launch(b.data_ptr<float>());
    else if (b.scalar_type()==at::kFloat8_e4m3fn) launch(reinterpret_cast<const c10::Float8_e4m3fn*>(b.data_ptr<c10::Float8_e4m3fn>()));
    else launch(reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_marks) return {out,maxima,unsafe,marks};
    return {out,maxima,unsafe};
}
// Packed layout is [batch*n, k_pad], identical to what pack_rows produces for a
// transposed weight, so the same batched GEMM consumes it unchanged.
std::vector<torch::Tensor> pack_cols_batched(torch::Tensor b, bool allow_wide_range, bool return_marks, int64_t n_pad, int64_t k_pad) {
    TORCH_CHECK(b.is_cuda() && is_w_dtype(b.scalar_type()) && b.is_contiguous() && b.dim()==3,"expected contiguous bf16/fp32/(fp8) [batch,k,n]");
    int64_t batch=b.size(0),k64=b.size(1),n64=b.size(2);
    TORCH_CHECK(batch>0 && k64>0 && n64>0 && batch*n64<=INT_MAX && k64<=INT_MAX,"shape too large");
    int64_t kp=k_pad>0?k_pad:k64, np=n_pad>0?n_pad:n64;
    TORCH_CHECK(kp>=k64 && np>=n64 && kp<=INT_MAX && np<=INT_MAX,"invalid padded shape");
    int k=int(k64),n=int(n64),chunks=(k+255)/256;
    int64_t bn=batch*n64;
    const c10::cuda::CUDAGuard guard(b.device());
    auto out=torch::empty({bn,kp},b.options().dtype(at::kHalf));
    if (kp>k64) out.narrow(1,k64,kp-k64).zero_();
    auto maxima=torch::ones({bn},b.options().dtype(at::kFloat));
    auto unsafe=torch::zeros({},b.options().dtype(at::kInt));
    auto marks=return_marks ? torch::zeros({bn},b.options().dtype(at::kInt)) : torch::Tensor();
    auto partial=torch::empty({chunks,3,bn},b.options().dtype(at::kInt));
    auto stream=at::cuda::getCurrentCUDAStream();
    auto ptr=reinterpret_cast<unsigned int*>(partial.data_ptr<int>());
    auto launch=[&](auto* src){
        using In=typename std::remove_const<typename std::remove_pointer<decltype(src)>::type>::type;
        col_partial_stats_batched<In><<<dim3(int((bn+31)/32),chunks),256,0,stream>>>(src,ptr,k,n,bn);
        finish_cols_batched<<<int((bn+255)/256),256,0,stream>>>(ptr,maxima.data_ptr<float>(),unsafe.data_ptr<int>(),chunks,bn,allow_wide_range,return_marks?marks.data_ptr<int>():nullptr);
        pack_cols_transpose_batched<In><<<dim3(int((bn+31)/32),int((k+31)/32)),dim3(32,8),0,stream>>>(src,reinterpret_cast<half*>(out.data_ptr<at::Half>()),maxima.data_ptr<float>(),k,n,int(kp),bn);
    };
    if (b.scalar_type()==at::kFloat) launch(b.data_ptr<float>());
    else if (b.scalar_type()==at::kFloat8_e4m3fn) launch(reinterpret_cast<const c10::Float8_e4m3fn*>(b.data_ptr<c10::Float8_e4m3fn>()));
    else launch(reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_marks) return {out,maxima,unsafe,marks};
    return {out,maxima,unsafe};
}

// Invoked only when the first pass found unsafe input. Each result marks a
// complete A row or B column that needs an FP32 dot product.
template <typename In>
__global__ void mark_unsafe_rows(const In* __restrict__ x, int* __restrict__ bad, int cols) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    unsigned int mn=0x7f800000u,mx=0u,flags=0u;
    for (int col=tid;col<cols;col+=256) {
        unsigned int u=__float_as_uint(bf2f(x[int64_t(row)*cols+col])) & 0x7fffffffu;
        if (u>=0x7f800000u) flags|=1u;
        else if (u) { mn=min(mn,u); mx=max(mx,u); if (u<0x00800000u) flags|=2u; }
    }
    for (int d=16;d>0;d>>=1) {
        mn=min(mn,__shfl_down_sync(0xffffffffu,mn,d));
        mx=max(mx,__shfl_down_sync(0xffffffffu,mx,d));
        flags|=__shfl_down_sync(0xffffffffu,flags,d);
    }
    __shared__ unsigned int smn[8],smx[8],sf[8];
    if (lane==0) { smn[warp]=mn; smx[warp]=mx; sf[warp]=flags; }
    __syncthreads();
    if (warp==0) {
        mn=lane<8?smn[lane]:0x7f800000u;
        mx=lane<8?smx[lane]:0u;
        flags=lane<8?sf[lane]:0u;
        for (int d=16;d>0;d>>=1) {
            mn=min(mn,__shfl_down_sync(0xffffffffu,mn,d));
            mx=max(mx,__shfl_down_sync(0xffffffffu,mx,d));
            flags|=__shfl_down_sync(0xffffffffu,flags,d);
        }
        if (lane==0) bad[row]=(flags || (mx && double(__uint_as_float(mx))>double(__uint_as_float(mn))*1073217536.0)) ? 1 : 0;
    }
}

torch::Tensor mark_rows(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type()==at::kBFloat16 || x.scalar_type()==at::kFloat) && x.is_contiguous() && x.dim()==2 && x.size(0)>0 && x.size(1)>0 && x.size(0)<=INT_MAX && x.size(1)<=INT_MAX,"expected nonempty contiguous bfloat16/float32 CUDA matrix");
    const c10::cuda::CUDAGuard guard(x.device());
    auto bad=torch::empty({x.size(0)},x.options().dtype(at::kInt));
    if (x.scalar_type()==at::kFloat)
        mark_unsafe_rows<float><<<int(x.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),bad.data_ptr<int>(),int(x.size(1)));
    else
        mark_unsafe_rows<__nv_bfloat16><<<int(x.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),bad.data_ptr<int>(),int(x.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("pack_rows",&pack_rows,pybind11::arg("x"),pybind11::arg("allow_wide_range")=false,pybind11::arg("return_marks")=false,pybind11::arg("rows_pad")=-1,pybind11::arg("cols_pad")=-1);
    m.def("pack_cols",&pack_cols,pybind11::arg("b"),pybind11::arg("allow_wide_range")=false,pybind11::arg("return_marks")=false,pybind11::arg("n_pad")=-1,pybind11::arg("k_pad")=-1);
    m.def("pack_cols_batched",&pack_cols_batched,pybind11::arg("b"),pybind11::arg("allow_wide_range")=false,pybind11::arg("return_marks")=false,pybind11::arg("n_pad")=-1,pybind11::arg("k_pad")=-1);
    m.def("pack_pair",&pack_pair,pybind11::arg("a"),pybind11::arg("w"),pybind11::arg("allow_wide_range")=false,pybind11::arg("return_marks")=false,pybind11::arg("m_pad")=-1,pybind11::arg("n_pad")=-1,pybind11::arg("k_pad")=-1);
    m.def("mark_rows",&mark_rows);
    m.def("gemm_restore",&gemm_restore,pybind11::arg("a"),pybind11::arg("bt"),pybind11::arg("ma"),pybind11::arg("mb"),pybind11::arg("m_out")=-1,pybind11::arg("n_out")=-1,pybind11::arg("bias")=pybind11::none(),pybind11::arg("scratch_bytes")=0,pybind11::arg("out_fp32")=false);
    m.def("gemm_restore_batched",&gemm_restore_batched,pybind11::arg("a"),pybind11::arg("bt"),pybind11::arg("ma"),pybind11::arg("mb"),pybind11::arg("batch"),pybind11::arg("m"),pybind11::arg("n"),pybind11::arg("mb_stride"),pybind11::arg("bias")=pybind11::none(),pybind11::arg("scratch_bytes")=0,pybind11::arg("out_fp32")=false);
}
