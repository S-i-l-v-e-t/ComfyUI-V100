"""Operator-level profile of one MiniMax-H3-Image block (bf16 softbf16)."""
import torch
from pathlib import Path
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

import comfy.ops
from comfy.ldm.minimax.model import DiTBlock, rope_rotation_table

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
HIDDEN, HEADS, HEAD_DIM, FFN, T_DIM = 5376, 56, 128, 14336, 8
T = int(__import__("os").environ.get("BENCH_TOKENS", "9409"))


def load_block(device, dtype, ops):
    block = DiTBlock(HIDDEN, HEADS, HEAD_DIM, FFN, T_DIM, 1e-6, 1e-6,
                     gate_compress=False, dtype=dtype, device=device, operations=ops)
    prefix = "blocks.0."
    with safe_open(CKPT, "pt") as f:
        keys = set(f.keys())
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
    return block.eval()


def main():
    device = torch.device("cuda")
    block = load_block(device, torch.bfloat16, comfy.ops.softbf16_ops)
    angles = torch.randn(40000, 32, device=device, dtype=torch.float32).cumsum(0)
    rope = rope_rotation_table(angles, torch.bfloat16)
    t_emb = (torch.randn(1, T_DIM, device=device) * 0.1).to(torch.bfloat16)
    x = (torch.randn(T, HIDDEN, device=device) * 0.1).to(torch.bfloat16)
    rf, seg = rope[:, :T], [(0, T, 0)]

    with torch.inference_mode():
        for _ in range(3):
            block(x, t_emb, seg, rf)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            block(x, t_emb, seg, rf)
        torch.cuda.synchronize()
    for event in prof.events():
        if event.name not in ("aten::mm", "aten::linear") or event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        stack = [frame for frame in (event.stack or []) if "site-packages" not in frame.filename]
        print(f"\n{event.name} cuda={event.device_time / 1000:.1f} ms shapes={event.input_shapes}")
        for frame in stack[:8]:
            print(f"    {Path(frame.filename).name}:{frame.lineno} {frame.name}")
    print()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))


if __name__ == "__main__":
    main()
