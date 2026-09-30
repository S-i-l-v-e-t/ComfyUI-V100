"""End-to-end test of FearL0rd/ComfyUI-Flash-Attention_v100 on the current ComfyUI.

Loads the node's __init__.py directly, patches comfy.ldm.modules.attention, and
calls optimized_attention the same way the current ComfyUI does.
"""
import sys
import traceback
import importlib.util

sys.path.insert(0, "/run/media/silvet/新加卷/ComfyUI")

import torch
from comfy.ldm.modules import attention as attn

orig = attn.optimized_attention
print("原始 optimized_attention:", orig.__name__)

spec = importlib.util.spec_from_file_location(
    "flash_v100_node", "/tmp/ComfyUI-Flash-Attention_v100/__init__.py"
)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)

ok = node.patcher.patch()
print("patch() ->", ok, "| patched:", node.patcher.patched, "| gpu:", node.patcher.gpu_arch)
print("patch 后 optimized_attention:", attn.optimized_attention.__name__)

torch.manual_seed(0)
B, S, H, D = 1, 4096, 24, 128
q = torch.randn(B, S, H * D, dtype=torch.float16, device="cuda")
k = torch.randn(B, S, H * D, dtype=torch.float16, device="cuda")
v = torch.randn(B, S, H * D, dtype=torch.float16, device="cuda")

print("\n--- Case 1: 当前 ComfyUI 真实调用方式（带 transformer_options kwarg）---")
try:
    out1 = attn.optimized_attention(q, k, v, H, attn_precision=None, transformer_options={})
    print("调用成功, out shape:", out1.shape)
except Exception as e:
    print(f"调用失败: {type(e).__name__}: {e}")

print("\n--- Case 2: 节点假设的旧布局 (batch*heads, seq, dim) + 计时 ---")
qb, kb, vb = torch.randn(B * H, S, D, dtype=torch.float16, device="cuda"), torch.randn(B * H, S, D, dtype=torch.float16, device="cuda"), torch.randn(B * H, S, D, dtype=torch.float16, device="cuda")
# 语义一致的基准：qb 即 (B, H, S, D) 展平 head 维，用 BHMD 布局 SDPA
def ref_attn(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.reshape(B, H, S, D), k.reshape(B, H, S, D), v.reshape(B, H, S, D)
    ).reshape(B * H, S, D)
try:
    for _ in range(5):
        out2 = attn.optimized_attention(qb, kb, vb, H)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(30):
        out2 = attn.optimized_attention(qb, kb, vb, H)
    e.record(); torch.cuda.synchronize()
    t_patched = s.elapsed_time(e) / 30
    err = (out2.float() - ref_attn(qb, kb, vb).float()).abs().max().item()
    print(f"out shape: {out2.shape} | max_abs_err: {err:.5f}")

    # baseline: 原始 attention (mem-efficient)，等价 FLOPs
    for _ in range(5):
        orig(qb.reshape(B, S, H * D), kb.reshape(B, S, H * D), vb.reshape(B, S, H * D), H)
    torch.cuda.synchronize()
    s.record()
    for _ in range(30):
        orig(qb.reshape(B, S, H * D), kb.reshape(B, S, H * D), vb.reshape(B, S, H * D), H)
    e.record(); torch.cuda.synchronize()
    t_orig = s.elapsed_time(e) / 30
    print(f"patched(v100_attention): {t_patched:7.2f}ms | original(mem-eff): {t_orig:7.2f}ms | ratio: {t_patched / t_orig:.2f}x")
except Exception as ex:
    traceback.print_exc()
    print(f"Case2 失败: {type(ex).__name__}: {ex}")

print("\n--- restore 测试 ---")
node.patcher.restore()
print("restore 后 optimized_attention:", attn.optimized_attention.__name__)
print("DONE")
