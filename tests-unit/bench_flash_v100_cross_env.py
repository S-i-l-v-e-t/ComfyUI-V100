"""Cross-check: run the same flash_attn_v100 dense shape under two envs.

Intent: confirm whether the ~2x slowdown vs mem-efficient is intrinsic to the
flash_attn_v100 kernel (both in the venv build and the 1cat env build) or an
artifact of the venv compilation.
"""
import torch

try:
    from yunchang.kernels import select_flash_attn_impl, AttnType
    torch_efficient_fn = select_flash_attn_impl(AttnType.TORCH_EFFICIENT, stage="fwd-only")
    HAS_EFF = True
except Exception:
    HAS_EFF = False

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


torch.manual_seed(0)
B, H, T, D = 1, 24, 8192, 128
qb = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")
kb = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")
vb = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")

t_v100 = bench(flash_attn_bhmd_func, qb, kb, vb)
line = f"py={torch.__version__:>10} flash_v100: {t_v100:7.2f}ms"
if HAS_EFF:
    qn, kn, vn = qb.transpose(1, 2).contiguous(), kb.transpose(1, 2).contiguous(), vb.transpose(1, 2).contiguous()
    t_eff = bench(torch_efficient_fn, qn, kn, vn, causal=False)
    line += f"  TORCH_EFFICIENT: {t_eff:7.2f}ms  ratio(flash/eff): {t_v100 / t_eff:.2f}x"
print(line)
