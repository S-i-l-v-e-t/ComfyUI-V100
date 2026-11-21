"""Block-level drift: fp32 reference vs bf16, and bf16 with/without the gated-MLP
wide-range packing.

Runs the first N blocks of MiniMax-H3-Image on real checkpoint weights and
compares the residual stream after each block.
"""
import contextlib
import os

import torch
from safetensors import safe_open

import comfy.ops
import comfy.quant_ops
from comfy.ldm.minimax.model import DiTBlock, rope_rotation_table

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
HIDDEN, HEADS, HEAD_DIM, FFN, T_DIM = 5376, 56, 128, 14336, 8
N_BLOCKS = int(os.environ.get("BLOCKS", "4"))
T = int(os.environ.get("BENCH_TOKENS", "2048"))


def build(count, dtype, ops, device):
    blocks = []
    with safe_open(CKPT, "pt") as f:
        keys = set(f.keys())
        for index in range(count):
            block = DiTBlock(HIDDEN, HEADS, HEAD_DIM, FFN, T_DIM, 1e-6, 1e-6,
                             gate_compress=False, dtype=dtype, device=device, operations=ops)
            prefix = f"blocks.{index}."
            for key in keys:
                if not key.startswith(prefix) or key.endswith(("comfy_quant", ".weight_scale")):
                    continue
                name = key[len(prefix):]
                tensor = f.get_tensor(key).to(device=device, dtype=dtype)
                scale_key = prefix + name[:-len(".weight")] + ".weight_scale"
                if name.endswith(".weight") and scale_key in keys:
                    tensor = tensor * f.get_tensor(scale_key).item()
                block.get_parameter(name).data.copy_(tensor)
            for parameter in block.parameters():
                parameter.requires_grad_(False)
            blocks.append(block.eval())
    return blocks


def run(blocks, x, rope, dtype):
    t_emb = (torch.randn(1, T_DIM, device=x.device) * 0.1).to(dtype)
    seg = [(0, T, 0)]
    for block in blocks:
        x = block(x, t_emb, seg, rope)
    return x


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    angles = torch.randn(8000, 32, device=device, dtype=torch.float32).cumsum(0)
    x0 = (torch.randn(T, HIDDEN, device=device) * 0.1)

    print(f"{torch.cuda.get_device_name()} | {N_BLOCKS} blocks, T={T}")

    bf16_wide = build(N_BLOCKS, torch.bfloat16, comfy.ops.softbf16_ops, device)
    with torch.inference_mode():
        x = run(bf16_wide, x0.to(torch.bfloat16), rope_rotation_table(angles, torch.bfloat16)[:, :T], torch.bfloat16)
    out_wide = x.float()

    # 关掉宽范围打包 = 改动前的行为（逐行 fp32 修复）
    comfy.quant_ops.softbf16_allow_wide_range = contextlib.nullcontext
    with torch.inference_mode():
        x = run(bf16_wide, x0.to(torch.bfloat16), rope_rotation_table(angles, torch.bfloat16)[:, :T], torch.bfloat16)
    out_repair = x.float()
    del bf16_wide
    torch.cuda.empty_cache()

    fp32 = build(N_BLOCKS, torch.float32, comfy.ops.disable_weight_init, device)
    with torch.inference_mode():
        out_fp32 = run(fp32, x0, rope_rotation_table(angles, torch.float32)[:, :T], torch.float32)
    del fp32
    torch.cuda.empty_cache()

    rel = lambda a, b: float((a - b).abs().mean() / b.abs().mean())
    print(f"输出量级 mean|x|={float(out_fp32.abs().mean()):.4g}")
    print(f"bf16(宽范围) vs fp32 : relerr={rel(out_wide, out_fp32):.3e}  max|diff|={float((out_wide-out_fp32).abs().max()):.4g}")
    print(f"bf16(逐行修复) vs fp32: relerr={rel(out_repair, out_fp32):.3e}  max|diff|={float((out_repair-out_fp32).abs().max()):.4g}")
    print(f"宽范围 vs 逐行修复    : relerr={rel(out_wide, out_repair):.3e}  max|diff|={float((out_wide-out_repair).abs().max()):.4g}")


if __name__ == "__main__":
    main()
