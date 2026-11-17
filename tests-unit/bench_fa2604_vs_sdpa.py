"""flash_attn_v100 26.04 (SM70 prefill build) vs torch memory-efficient on V100.

26.04 exposes the standard flash_attn_func with (B, T, H, D) layout.
Sweeps the shapes that matter here: MiniMax H3 is 56 heads x 128 dim, FLUX 24.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash_attn_v100.flash_attn_interface import flash_attn_func


def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def sdpa(q, k, v):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)


torch.manual_seed(0)
print(f"{'model':<10} {'T':>6} {'H':>3} {'D':>4} | {'SDPA-eff':>9} {'FA26.04':>9} | {'FA/SDPA':>8}")
print("-" * 68)

for label, H, D, seqs in (
    ("MiniMaxH3", 56, 128, (2048, 4096, 8192, 16384)),
    ("FLUX", 24, 128, (4096, 8192, 16384)),
    ("FLUX-D64", 24, 64, (4096, 8192, 16384)),
    ("H3-long", 56, 128, (32768,)),
):
    for T in seqs:
        iters = 20 if T >= 8192 else 50
        qf = torch.randn(1, T, H, D, dtype=torch.float16, device="cuda")
        kf = torch.randn_like(qf)
        vf = torch.randn_like(qf)

        t_fa = bench(lambda: flash_attn_func(qf, kf, vf), iters)

        qs = qf.transpose(1, 2).contiguous()
        ks = kf.transpose(1, 2).contiguous()
        vs = vf.transpose(1, 2).contiguous()
        t_sdpa = bench(lambda: sdpa(qs, ks, vs), iters)

        print(
            f"{label:<10} {T:>6} {H:>3} {D:>4} | {t_sdpa:>9.2f} {t_fa:>9.2f} | "
            f"{t_sdpa / t_fa:>7.2f}x"
        )

qf = torch.randn(1, 2048, 24, 128, dtype=torch.float16, device="cuda")
qs = qf.transpose(1, 2).contiguous()
ref = sdpa(qs, qs, qs).transpose(1, 2).float()
got = flash_attn_func(qf, qf, qf).float()
print(f"\n数值一致性: max|FA - SDPA| = {(got - ref).abs().max().item():.4f}")
