"""Smoke test: load the cp312 flash_attn_v100 .so inside the ComfyUI venv (torch 2.9.1).

Run with the venv python; the flash_attn_v100 package dir is injected from the
1cat-vllm-sm70 env without pulling in that env's torch.
"""
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")

from flash_attn_v100.flash_attn_interface import flash_attn_bhmd_func
print("import flash_attn_v100 OK")

torch.manual_seed(0)
B, T, H, D = 2, 256, 8, 64
q = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")

out = flash_attn_bhmd_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
).transpose(1, 2)
print("finite:", torch.isfinite(out).all().item())
print("max_abs_err: %.5f" % (out.float() - ref.float()).abs().max().item())
print("SMOKE_OK")
