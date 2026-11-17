# Copyright (C) 2026 Silvet.icu
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Route attention to the memory-efficient backend (bf16 has no kernel on sm_70).

Usage:
    bf16_attn.install()  /  bf16_attn.uninstall()
    out = bf16_attn.attention(q, k, v, scale=None, is_causal=False)
    bf16_attn.set_mode('fp16_scaled' | 'fp32' | 'fp16')

See docs/attention.md.
"""
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = ['attention', 'install', 'uninstall', 'is_installed', 'set_mode', 'get_mode',
           'fallbacks']

_ORIG = F.scaled_dot_product_attention
_installed = False

# How bf16 enters EFFICIENT: 'fp16_scaled' (default) | 'fp32' | 'fp16'.
# Only module attributes are patched, so early-bound imports are not covered.
_MODES = ('fp16_scaled', 'fp32', 'fp16')
_mode = 'fp16_scaled'


def set_mode(mode):
    """Pick how bf16 attention enters EFFICIENT: 'fp16_scaled', 'fp32' or 'fp16'."""
    global _mode
    if mode not in _MODES:
        raise ValueError('mode must be one of %s' % (_MODES,))
    _mode = mode


def get_mode():
    return _mode


def _pow2_scale(mx):
    """Power of two that maps |max| into [1024, 2048): the safe band for fp16."""
    if not (mx > 0.0) or not math.isfinite(mx):
        return 1.0
    return 2.0 ** math.floor(math.log2(2048.0 / mx))


# Fallback threshold: ratio beyond which entries vanish in fp16 (see docs/attention.md).
_MAX_SAFE_RATIO = 2.0 ** 38
_fallbacks = 0


def fallbacks():
    """How many calls fell back to fp32 because fp16 could not hold them."""
    return _fallbacks


def _range(t):
    """(max|.|, smallest |.|) as device scalars - no host sync yet."""
    a = t.abs()
    mn, mx = torch.aminmax(a)
    return mx, mn


def _min_nonzero(t):
    """Smallest non-zero |.|; only needed when the tensor holds exact zeros."""
    a = t.abs()
    return torch.where(a > 0, a, torch.full_like(a, float('inf'))).amin().item()


def _unsafe(mx, mn):
    """True if the fp16 round trip would lose entries this tensor carries."""
    if not math.isfinite(mx):
        return True
    if mx <= 0.0 or not math.isfinite(mn):
        return False
    return mx > mn * _MAX_SAFE_RATIO


def attention(query, key, value, *args, **kwargs):
    """Same dtype in and out; only bf16 is detoured, and by default to fp32.

    sm_70's memory-efficient backend has no bf16 kernel - feeding bf16 straight in
    raises "No available kernel. Aborting execution." So bf16 is the one case that
    is upcast: q/k/v go to fp32 (or fp16, per `set_mode`), run on the EFFICIENT
    kernel, and the result is rounded back to bf16 on the way out.

    fp16 input is the backend's native dtype and is measured several times faster
    than fp32 (S=4096, H=10, D=64: 1.54 ms vs 5.09 ms), so it is passed through
    as-is instead of being upcast for no reason. fp32 likewise runs as-is.
    """
    dt = query.dtype
    if dt != torch.bfloat16:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = _ORIG(query, key, value, *args, **kwargs)
        return out

    if _mode == 'fp32':
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = _ORIG(query.float(), key.float(), value.float(), *args, **kwargs)
        return out.to(dt)

    if _mode == 'fp16':
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = _ORIG(query.half(), key.half(), value.half(), *args, **kwargs)
        return out.to(dt)

    # 'fp16_scaled': per-tensor power-of-two scale, folded into the softmax scale
    # for q/k and undone on the output for v. Falls back to fp32 when a tensor's
    # range does not survive fp16 (see docs/attention.md).
    global _fallbacks
    pairs = [_range(t) for t in (query, key, value)]
    extremes = torch.stack([v for pair in pairs for v in pair]).tolist()   # one sync
    pairs = [(extremes[2 * i], extremes[2 * i + 1]) for i in range(3)]
    # aminmax reports exact zeros as the minimum; those are harmless, so recover the
    # smallest non-zero entry for such tensors only.
    for i, t in enumerate((query, key, value)):
        if pairs[i][1] == 0.0:
            pairs[i] = (pairs[i][0], _min_nonzero(t))
    (mxq, mnq), (mxk, mnk), (mxv, mnv) = pairs
    if (_unsafe(mxq, mnq) or _unsafe(mxk, mnk) or _unsafe(mxv, mnv)
            or not (math.isfinite(mxq) and math.isfinite(mxk)
                    and math.isfinite(mxv))):
        _fallbacks += 1
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = _ORIG(query.float(), key.float(), value.float(), *args, **kwargs)
        return out.to(dt)

    sq, sk, sv = _pow2_scale(mxq), _pow2_scale(mxk), _pow2_scale(mxv)
    scale = kwargs.pop('scale', None)
    if scale is None:
        scale = float(query.shape[-1]) ** -0.5
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = _ORIG((query * sq).half(), (key * sk).half(), (value * sv).half(),
                    *args, scale=scale / (sq * sk), **kwargs)
    return (out.float() / sv).to(dt)


def install():
    """Patch F.scaled_dot_product_attention. Idempotent."""
    global _installed
    if not _installed:
        F.scaled_dot_product_attention = attention
        _installed = True


def uninstall():
    global _installed
    F.scaled_dot_product_attention = _ORIG
    _installed = False


def is_installed():
    return _installed
