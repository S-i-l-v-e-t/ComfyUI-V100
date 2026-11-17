# Copyright (C) 2026 Silvet.icu
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scaled fp16 tensor-core GEMM for bf16/fp32 activations and bf16/fp32/fp8 weights.

    linear(x, w, bias=None)          # output dtype follows x
    matmul_bt(x, w, bias=None)       # same, explicit weight form [out, in]
    bmm(a, b, out_fp32=False)        # torch.bmm layout, inference only when out_fp32

See docs/gemm-kernel.md and docs/io-dtypes.md.
"""
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

__all__ = ['linear', 'matmul_bt', 'plan']

# Repair policy: unsafe rows/columns (inf/nan, fp32 subnormals, or a within-row
# range wider than ~2^30) are recomputed in fp32 instead of silently packed.
_FAST_COLS = True               # coalesced column reduction + fp16 transpose
_SELECTIVE_FALLBACK = True      # recompute unsafe rows/columns in fp32
_MAX_REPAIR_ROWS = 32
_MAX_REPAIR_COLS = 32
_MAX_REPAIR_FRACTION = 0.05
_USE_TUNED_REPAIR_LIMITS = True
_REPAIR_CALIB = ((4096, 2688), (8192, 3584))   # max_repair(S) = 1792 + 0.21875*S
_TUNED_MAX_FRACTION = 0.875


def _max_repair(size):
    (s0, r0), (s1, r1) = _REPAIR_CALIB
    return r0 + (r1 - r0) * (size - s0) / float(s1 - s0)


def _repair_within_limits(m, k, n, nr, nc):
    if not (nr or nc):
        return False
    if _USE_TUNED_REPAIR_LIMITS:
        max_rows = _max_repair(m)
        max_cols = _max_repair(n)
        max_fraction = _TUNED_MAX_FRACTION
    else:
        max_rows, max_cols, max_fraction = (_MAX_REPAIR_ROWS, _MAX_REPAIR_COLS,
                                            _MAX_REPAIR_FRACTION)
    return nr <= max_rows and nc <= max_cols and nr / m + nc / n <= max_fraction


_ext = None


def _kernel():
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '%d.%d' % (major, minor))
        _ext = load(name='rowcol_scaled_gemm_bf16_sm%d%d' % (major, minor),
                    sources=[str(Path(__file__).parent / 'kernels' /
                                'rowcol_scaled_gemm_bf16.cu')],
                    extra_cuda_cflags=['-O3'], extra_ldflags=['-lcublas'], verbose=False)
    return _ext


def _fp32_mm(a, b, bias=None, out_fp32=False):
    out = torch.mm(a.float(), b.float())
    if bias is not None:
        out = out + bias
    return out if out_fp32 else out.to(a.dtype)


# fp16 tensor cores want multiples of 8, otherwise cuBLAS drops to a slow kernel
_ALIGN = 8
# Above _SCRATCH_LIMIT the output is produced in row blocks of _SCRATCH_BYTES
_SCRATCH_BYTES = 128 << 20
_SCRATCH_LIMIT = 256 << 20
# crossover vs the fp32 bypass (see docs/gemm-kernel.md)
_MIN_VOLUME = 2e9
_MIN_VOLUME_BATCH = 8e8
# above this slice volume, per-slice cuBLAS calls beat the batched one
_BATCH_SLICE_LIMIT = 2e10


def plan(m, k, n):
    """Round each dimension up to a multiple of 8, returning (m_pad, k_pad, n_pad)."""
    return tuple(-(-x // _ALIGN) * _ALIGN for x in (m, k, n))


def _run(a, b, allow_wide_range=False, bias=None, out_fp32=False):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('expected A[M,K] and B[K,N]')
    if (not a.is_cuda or a.device != b.device
            or a.dtype not in (torch.bfloat16, torch.float32)
            or not (b.dtype == a.dtype or b.dtype == torch.float8_e4m3fn)):
        raise ValueError('expected matching bf16/fp32 matrices, or fp8 e4m3 weights '
                         'with bf16/fp32 activations, on one device')
    # fp32 in -> fp32 out
    out_fp32 = out_fp32 or a.dtype == torch.float32
    if min(*a.shape, *b.shape) == 0:
        return torch.mm(a, b)
    bf = None if bias is None else bias.float().contiguous()
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if m * k * n < _MIN_VOLUME:
        return _fp32_mm(a, b, bf, out_fp32)
    m_pad, k_pad, n_pad = plan(m, k, n)
    ac = a.contiguous()
    ext = _kernel()
    sel = _SELECTIVE_FALLBACK
    fast_cols = _FAST_COLS and b.is_contiguous() and (b.shape[0] + 31) // 32 <= 65535
    if fast_cols:
        pa = ext.pack_rows(ac, allow_wide_range, sel, m_pad, k_pad)
        ah, ma, ua = pa[:3]
        marks_a = pa[3] if sel else None
        pb = ext.pack_cols(b, allow_wide_range, sel, n_pad, k_pad)
        bh, mb, ub = pb[:3]
        marks_b = pb[3] if sel else None
    else:
        # Pack A and W in a single launch (same row body, merged grid)
        pg = ext.pack_pair(ac, b.T.contiguous(), allow_wide_range, sel,
                           m_pad, n_pad, k_pad)
        ah, ma, ua, bh, mb, ub = pg[:6]
        marks_a, marks_b = (pg[6], pg[7]) if sel else (None, None)
    if bool((ua | ub).item()):
        if not sel:
            return _fp32_mm(a, b, bf, out_fp32)
        bad_rows = marks_a.nonzero().flatten() if ua.item() else None
        bad_cols = marks_b.nonzero().flatten() if ub.item() else None
        nr = bad_rows.numel() if bad_rows is not None else 0
        nc = bad_cols.numel() if bad_cols is not None else 0
        if not _repair_within_limits(m, k, n, nr, nc):
            return _fp32_mm(a, b, bf, out_fp32)
        # gemm_restore crops to the real shape and adds bias in the same epilogue
        sb = _SCRATCH_BYTES if m * n * 4 > _SCRATCH_LIMIT else 0
        c = ext.gemm_restore(ah, bh, ma, mb, m, n, bf, sb, out_fp32)
        if nr:
            r = torch.mm(a.index_select(0, bad_rows).float(), b.float())
            if bf is not None:
                r = r + bf
            c.index_copy_(0, bad_rows, r.to(c.dtype))
        if nc:
            r = torch.mm(a.float(), b.index_select(1, bad_cols).float())
            if bf is not None:
                r = r + bf[bad_cols]
            c.index_copy_(1, bad_cols, r.to(c.dtype))
        return c
    sb = _SCRATCH_BYTES if m * n * 4 > _SCRATCH_LIMIT else 0
    return ext.gemm_restore(ah, bh, ma, mb, m, n, bf, sb, out_fp32)


class _Gemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, allow_wide_range, bias):
        ctx.save_for_backward(a, b)
        ctx.allow_wide_range = allow_wide_range
        return _run(a, b, allow_wide_range, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        da = _run(g, b.T, ctx.allow_wide_range) if ctx.needs_input_grad[0] else None
        db = _run(a.T, g, ctx.allow_wide_range) if ctx.needs_input_grad[1] else None
        dbias = g.sum(0) if ctx.needs_input_grad[3] else None
        return da, db, None, dbias


def _fp32_bmm(a, b, bias=None, out_fp32=False):
    out = torch.bmm(a.float(), b.float())
    if bias is not None:
        out = out + bias
    return out if out_fp32 else out.to(a.dtype)


def _bmm_pack(a, b, allow_wide_range, bias, col_major, out_fp32=False):
    """Shared core: a [B,M,K] with b either [B,N,K] (weight) or [B,K,N] (bmm)."""
    B, m, k = a.shape
    n = b.shape[2] if col_major else b.shape[1]
    bw = b.shape[0]
    bb = b if col_major else b.transpose(1, 2)
    out_fp32 = out_fp32 or a.dtype == torch.float32
    if m * k * n * B < _MIN_VOLUME_BATCH:
        return _fp32_bmm(a, bb.expand(B, k, n) if bw == 1 else bb, bias, out_fp32)
    # big slices: per-slice calls beat the strided-batched one
    if m * k * n >= _BATCH_SLICE_LIMIT and B <= 32:
        out = torch.empty(B, m, n, device=a.device,
                          dtype=torch.float32 if out_fp32 else a.dtype)
        for i in range(B):
            if col_major:
                bi = b[i]                       # already [k,n]
            else:
                bi = (b[0] if bw == 1 else b[i]).t()    # weight is [n,k] -> [k,n]
            out[i] = _run(a[i], bi, allow_wide_range, bias, out_fp32)
        return out
    k_pad = -(-k // _ALIGN) * _ALIGN
    ext = _kernel()
    sel = _SELECTIVE_FALLBACK
    bf = None if bias is None else bias.float().contiguous()
    pa = ext.pack_rows(a.reshape(B * m, k).contiguous(), allow_wide_range, sel, -1,
                       k_pad)
    if col_major:
        pb = ext.pack_cols_batched(b.contiguous(), allow_wide_range, sel, n, k_pad)
    else:
        pb = ext.pack_rows(b.reshape(bw * n, k).contiguous(), allow_wide_range, sel, -1,
                           k_pad)
    ah, ma, ua = pa[:3]
    bh, mb, ub = pb[:3]
    if bool((ua | ub).item()):
        return _fp32_bmm(a, bb.expand(B, k, n) if bw == 1 else bb, bias, out_fp32)
    mb_stride = 0 if bw == 1 else n
    sb = _SCRATCH_BYTES if B * m * n * 4 > _SCRATCH_LIMIT else 0
    return ext.gemm_restore_batched(ah, bh, ma, mb, B, m, n, mb_stride, bf, sb, out_fp32)


def _run_batched(a, bt, allow_wide_range=False, bias=None):
    """a [B,M,K] @ bt [B or 1,N,K].T -> [B,M,N]."""
    bw = bt.shape[0]
    if bw not in (1, a.shape[0]):
        raise ValueError('weight batch must be 1 or match the input batch')
    if bw == 1:
        return _bmm_pack(a, bt.expand(a.shape[0], bt.shape[1], bt.shape[2]),
                         allow_wide_range, bias, False)
    return _bmm_pack(a, bt, allow_wide_range, bias, False)


def _unwrap(x):
    """[..., M, K] -> (flat [B, M, K], B, leading shape)."""
    if x.dim() < 3:
        raise ValueError('expected at least 3 dimensions')
    lead = x.shape[:-2]
    b = 1
    for v in lead:
        b *= v
    return x.reshape(b, x.shape[-2], x.shape[-1]), b, lead


class _GemmB(torch.autograd.Function):
    """Batched entry. col_major=False means b is [B,N,K] (linear weight form),
    True means b is [B,K,N] (torch.bmm form)."""
    @staticmethod
    def forward(ctx, a, b, allow_wide_range, bias, col_major):
        ctx.save_for_backward(a, b)
        ctx.allow_wide_range = allow_wide_range
        ctx.col_major = col_major
        return _bmm_pack(a, b, allow_wide_range, bias, col_major)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        awr = ctx.allow_wide_range
        g = g.contiguous()
        # Both gradients contract along the batch-invariant axis, so each is one
        # batched GEMM in the col-major layout.
        if ctx.col_major:
            # b is [B,k,n]: dA contracts over b's first axis, so it is b transposed
            # back into the [k,n] operand layout.
            da = (_bmm_pack(g, b.transpose(1, 2).contiguous(), awr, None, True)
                  if ctx.needs_input_grad[0] else None)
            db = (_bmm_pack(a.transpose(1, 2).contiguous(), g, awr, None, True)
                  if ctx.needs_input_grad[1] else None)
        else:
            # b is the weight [B,out,in] = [k_contract, n_out] for dA as written, so
            # it is used as-is.
            da = (_bmm_pack(g, b, awr, None, True)
                  if ctx.needs_input_grad[0] else None)
            db = None
            if ctx.needs_input_grad[1]:
                gt = g.transpose(1, 2).contiguous()
                db = _bmm_pack(gt, a.contiguous(), awr, None, True)
                if b.shape[0] == 1:
                    db = db.sum(0, keepdim=True)
        dbias = g.sum(dim=(0, 1)) if ctx.needs_input_grad[3] else None
        return da, db, None, dbias, None


def matmul_bt(a, w, bias=None, allow_wide_range=False):
    """a[..., K] @ w[N, K].T for an nn.Linear weight [out, in].

    bf16/fp32 activations, bf16/fp32/fp8-e4m3 weights (fp8 is inference only);
    output dtype follows the activation. Bias is added in fp32 in the restore
    epilogue and rounded once, like F.linear.
    """
    if w.dtype == torch.float8_e4m3fn and torch.is_grad_enabled() and (
            a.requires_grad or w.requires_grad):
        raise ValueError('fp8 weights are inference only')
    if a.dim() == 2 and w.dim() == 2:
        if torch.is_grad_enabled() and (a.requires_grad or w.requires_grad):
            out = _Gemm.apply(a, w.t(), allow_wide_range, bias)
        else:
            # inference: the autograd wrapper costs ~15 us a call
            out = _run(a, w.t(), allow_wide_range, bias)
        return out.reshape(*a.shape[:-1], w.shape[0])
    if w.dim() not in (2, 3):
        raise ValueError('weight must be 2D or 3D')
    af, b, lead = _unwrap(a)
    wf = w if w.dim() == 3 else w.unsqueeze(0)
    if wf.shape[0] not in (1, b):
        raise ValueError('weight batch must be 1 or match the input batch')
    out = _GemmB.apply(af.contiguous(), wf.contiguous(), allow_wide_range, bias, False)
    return out.reshape(*lead, af.shape[1], wf.shape[1])


def bmm(a, b, allow_wide_range=False, out_fp32=False):
    """torch.bmm(a[B,M,K], b[B,K,N]) for bf16/fp32 tensors (no transpose copy).

    out_fp32 keeps the result in fp32 for bf16 inputs (inference only); fp32 inputs
    already produce fp32. b may be fp8 e4m3 (W8A16, inference only).
    """
    if b.dtype == torch.float8_e4m3fn and torch.is_grad_enabled() and (
            a.requires_grad or b.requires_grad):
        raise ValueError('fp8 weights are inference only')
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError('bmm expects 3D tensors')
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError('bmm shape mismatch')
    if out_fp32:
        if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
            raise ValueError('out_fp32 is inference-only')
        return _bmm_pack(a.contiguous(), b.contiguous(), allow_wide_range, None,
                         True, out_fp32=True)
    return _GemmB.apply(a.contiguous(), b.contiguous(), allow_wide_range, None, True)


def linear(x, w, bias=None, allow_wide_range=False):
    """F.linear for bf16/fp32 activations; output dtype follows the activation."""
    return matmul_bt(x, w, bias, allow_wide_range)
