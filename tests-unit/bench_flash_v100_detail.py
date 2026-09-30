"""Detailed benchmark: TORCH_EFFICIENT vs flash_attn_v100 on V100.

Sweeps head_dim and seq length at FLUX-like dense fp16 shapes to check whether
flash_attn_v100 actually wins, and where.
"""
import torch
from yunchang.kernels import select_flash_attn_impl, AttnType
from flash_attn_v100.flash_attn_interface import flash_attn_bhmd_func


def bench(fn, *args, warmup=10, iters=60, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


torch_efficient_fn = select_flash_attn_impl(AttnType.TORCH_EFFICIENT, stage="fwd-only")

torch.manual_seed(0)
print(f"{'B':>1} {'T':>6} {'H':>3} {'D':>4} | {'TORCH_EFFICIENT':>15} {'flash_v100':>12} {'speedup':>8}")
print("-" * 62)
for D in (64, 128):
    for T in (2048, 4096, 8192, 16384):
        B, H = 1, 24
        qb = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")
        kb = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")
        vb = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")
        qn, kn, vn = qb.transpose(1, 2).contiguous(), kb.transpose(1, 2).contiguous(), vb.transpose(1, 2).contiguous()

        t_eff = bench(torch_efficient_fn, qn, kn, vn, causal=False)
        t_v100 = bench(flash_attn_bhmd_func, qb, kb, vb)
        print(f"{B:>1} {T:>6} {H:>3} {D:>4} | {t_eff:>15.2f} {t_v100:>12.2f} {t_eff / t_v100:>7.2f}x")
