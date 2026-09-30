#!/usr/bin/env python
"""Check a model for NaN/Inf/overflow, then optionally convert bf16 -> fp16.

Usage:
  python convert_bf16_to_fp16.py check  <model.safetensors> [more files...]
  python convert_bf16_to_fp16.py convert <model.safetensors> <output.safetensors>
"""

import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP16_MAX = 65504.0


def scan(path):
    """One pass over the file, tensor by tensor. Returns dtype counts and any danger list."""
    dtype_counts = {}
    danger = []
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            dtype_counts[t.dtype] = dtype_counts.get(t.dtype, 0) + 1
            if t.is_floating_point():
                n_nan = int(t.isnan().sum())
                n_inf = int(t.isinf().sum())
                max_abs = t.abs().max().item() if t.numel() else 0.0
                if n_nan or n_inf or max_abs >= FP16_MAX:
                    danger.append((name, str(t.dtype), n_nan, n_inf, max_abs))
    return dtype_counts, danger


def report(path):
    dtype_counts, danger = scan(path)
    print(f"== {path}")
    for dt, n in dtype_counts.items():
        print(f"   {dt}: {n} tensors")
    if not danger:
        print("   OK: no NaN, no Inf, no |x| >= 65504")
    else:
        for name, dt, n_nan, n_inf, max_abs in danger:
            print(f"   DANGER {name}: dtype={dt} nan={n_nan} inf={n_inf} max_abs={max_abs}")


def convert(src, dst):
    dtype_counts, danger = scan(src)
    if danger:
        print("Refusing to convert, dangerous values found:")
        for name, dt, n_nan, n_inf, max_abs in danger:
            print(f"  {name}: dtype={dt} nan={n_nan} inf={n_inf} max_abs={max_abs}")
        sys.exit(1)

    out = {}
    with safe_open(src, framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if t.is_floating_point() and t.dtype != torch.float16:
                t = t.to(torch.float16)
            out[name] = t
    save_file(out, dst)
    print(f"Wrote {dst} ({len(out)} tensors, {dtype_counts})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    mode = sys.argv[1]
    paths = sys.argv[2:]
    if mode == "check":
        for p in paths:
            report(p)
    elif mode == "convert":
        if len(paths) != 2:
            print("convert takes <input> <output>")
            sys.exit(2)
        convert(paths[0], paths[1])
    else:
        print(__doc__)
        sys.exit(2)
