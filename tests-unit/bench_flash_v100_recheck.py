"""Re-check flash_attn_v100 against torch memory-efficient attention on V100.

1. Covers the shapes that actually matter on this box: MiniMax H3 is 56 heads x
   128 dim, FLUX is 24 heads. The earlier bench only swept H=24.
2. Isolates the call-site layout: raylight hands FA a tensor that came from
   (B,T,H,D).transpose(1,2), i.e. non-contiguous, which makes FA copy inside
   maybe_contiguous(). Compares it against a contiguous input and against an
   explicit out= buffer.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash_attn_v100.flash_attn_interface import flash_attn_bhmd_func


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
print(f"{'model':<10} {'T':>6} {'H':>3} {'D':>4} | {'SDPA-eff':>9} {'FA':>9} {'FA(noncont)':>12} {'FA(out=)':>9} | {'vs-eff':>7}")
print("-" * 92)

for label, H, D, seqs in (
    ("MiniMaxH3", 56, 128, (2048, 4096, 8192, 16384)),
    ("FLUX", 24, 128, (4096, 8192, 16384)),
    ("FLUX-D64", 24, 64, (4096, 8192, 16384)),
):
    for T in seqs:
        iters = 20 if T >= 8192 else 50
        q = torch.randn(1, H, T, D, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        t_sdpa = bench(lambda: sdpa(q, k, v), iters)
        t_fa = bench(lambda: flash_attn_bhmd_func(q, k, v), iters)

        # raylight 的调用方式：(B,T,H,D) -> transpose -> 非连续的 (B,H,T,D)
        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).contiguous()
        vt = v.transpose(1, 2).contiguous()
        qr, kr, vr = qt.transpose(1, 2), kt.transpose(1, 2), vt.transpose(1, 2)
        t_fa_nc = bench(lambda: flash_attn_bhmd_func(qr, kr, vr), iters)

        out = torch.empty_like(q)
        t_fa_out = bench(lambda: flash_attn_bhmd_func(q, k, v, out=out), iters)

        print(
            f"{label:<10} {T:>6} {H:>3} {D:>4} | {t_sdpa:>9.2f} {t_fa:>9.2f} "
            f"{t_fa_nc:>12.2f} {t_fa_out:>9.2f} | {t_sdpa / t_fa:>6.2f}x"
        )

# 数值一致性
q = torch.randn(1, 24, 2048, 128, dtype=torch.float16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)
ref = sdpa(q, k, v).float()
got = flash_attn_bhmd_func(q, k, v).float()
print(f"\n数值一致性: max|FA - SDPA| = {(got - ref).abs().max().item():.4f}")
