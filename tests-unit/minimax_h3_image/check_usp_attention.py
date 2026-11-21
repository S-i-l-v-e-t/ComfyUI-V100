"""Compare raylight's attention replacement with ComfyUI's Attention.forward.

Same module, same input, single rank: any difference here is a semantic bug in the
USP attention path (it is what the image model runs under raylight).
"""
import sys

import torch

import comfy.model_management as mm
import comfy.ops
from comfy.ldm.minimax.model import Attention, rope_rotation_table

sys.path.insert(0, "custom_nodes/raylight/src")
import types

from raylight.distributed_modules import attention as raylight_attention

raylight_attention.set_attn_type("TORCH_EFFICIENT")
raylight_attention.set_sync_ulysses(False)
from raylight.diffusion_models.minimax import xdit_context_parallel as xcp

HIDDEN, HEADS, HEAD_DIM, S = 256, 4, 64, 96
ROPE_DIM = 32


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    attn = Attention(HIDDEN, HEADS, HEAD_DIM, 1e-6, gate_compress=False,
                     dtype=torch.bfloat16, device=dev, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0.0, 0.05)
            p.requires_grad_(False)
    x = (torch.randn(S, HIDDEN, device=dev) * 0.5).to(torch.bfloat16)
    angles = torch.randn(S, ROPE_DIM, device=dev, dtype=torch.float32)
    rope = rope_rotation_table(angles, torch.bfloat16)

    with torch.inference_mode():
        ref = attn(x, rope_freqs=rope)
        attn.forward = types.MethodType(xcp.usp_attn_forward, attn)
        out = attn(x, rope_freqs=rope)

    diff = (out.float() - ref.float()).abs()
    scale = ref.float().abs().mean()
    print(f"comfy Attention.forward  absmean={float(ref.float().abs().mean()):.5f} "
          f"absmax={float(ref.float().abs().max()):.3f}")
    print(f"raylight usp_attn_forward absmean={float(out.float().abs().mean()):.5f} "
          f"absmax={float(out.float().abs().max()):.3f}")
    print(f"相对误差 {(float(diff.mean()) / float(scale)):.3e}  max|diff|={float(diff.max()):.4g}")
    print("finite:", bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    main()
