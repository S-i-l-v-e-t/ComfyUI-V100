"""Functional test for the raylight V100 flash ring-attn wrapper (single process).

Verifies that a dense fp16 call routes to flash_attn_v100 and that a call with
a condition we cannot satisfy (e.g. causal=True) falls back to the original.
"""
import sys

sys.path.insert(0, "/run/media/silvet/新加卷/ComfyUI/custom_nodes/raylight/src")

import torch
import torch.distributed as dist

from raylight.distributed_modules.attention import _flash_v100_ring_attn_factory

dist.init_process_group("gloo", init_method="tcp://127.0.0.1:23458", rank=0, world_size=1)
group = dist.group.WORLD

calls = {"orig": 0}

def orig_ring_attn(q, k, v, *args, **kwargs):
    calls["orig"] += 1
    return torch.zeros_like(q)

wrapped = _flash_v100_ring_attn_factory(orig_ring_attn)

torch.manual_seed(0)
B, T, H, D = 1, 64, 4, 64
q = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")

# Case 1: dense fp16, ring size 1 -> flash path (orig not called)
out1 = wrapped(q, k, v, group=group, softmax_scale=None, causal=False, window_size=(-1, -1))
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
).transpose(1, 2)
print("case1 orig_calls:", calls["orig"], "finite:", torch.isfinite(out1).all().item(),
      "err: %.5f" % (out1.float() - ref.float()).abs().max().item())

# Case 2: causal=True -> must fall back to original
out2 = wrapped(q, k, v, group=group, causal=True)
print("case2 orig_calls:", calls["orig"], "shape_ok:", out2.shape == q.shape)

# Case 3: fp32 -> must fall back to original
qf = q.float()
out3 = wrapped(qf, qf, qf, group=group)
print("case3 orig_calls:", calls["orig"])

print("RAYLIGHT_WRAPPER_OK" if calls["orig"] == 2 else "RAYLIGHT_WRAPPER_FAIL")
