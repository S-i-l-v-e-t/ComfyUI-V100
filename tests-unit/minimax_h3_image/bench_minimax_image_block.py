"""Stable per-block cost, bf16 (softbf16) vs fp16, at the image token count."""
import statistics

import torch
from safetensors import safe_open

import comfy.ops
from comfy.ldm.minimax.model import DiTBlock, rope_rotation_table

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
HIDDEN, HEADS, HEAD_DIM, FFN, T_DIM = 5376, 56, 128, 14336, 8
TOKEN_COUNTS = (4080, 9409)


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


def rounds(fn, repeats=4, iters=6):
    samples = []
    for _ in range(repeats):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return min(samples), statistics.median(samples)


def main():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(), "| free %.1f GiB" % (torch.cuda.mem_get_info()[0] / 2**30))
    angles = torch.randn(40000, 32, device=device, dtype=torch.float32).cumsum(0)
    print(f"{'dtype':>9} {'tokens':>7} {'block min':>10} {'med':>7} {'attn min':>9} {'mlp min':>8} {'50 blocks s':>12}")
    for dtype, ops in ((torch.bfloat16, comfy.ops.softbf16_ops), (torch.float16, comfy.ops.manual_cast)):
        block = load_block(device, dtype, ops)
        rope = rope_rotation_table(angles, dtype)
        t_emb = (torch.randn(1, T_DIM, device=device) * 0.1).to(dtype)
        for T in TOKEN_COUNTS:
            x = (torch.randn(T, HIDDEN, device=device) * 0.1).to(dtype)
            rf, seg = rope[:, :T], [(0, T, 0)]
            with torch.inference_mode():
                full = rounds(lambda: block(x, t_emb, seg, rf))
                attn = rounds(lambda: block.attn(x, rope_freqs=rf))
                mlp = rounds(lambda: block.mlp(block.norm2(x)))
            print(f"{str(dtype).split('.')[-1]:>9} {T:>7} {full[0]:10.1f} {full[1]:7.1f} "
                  f"{attn[0]:9.1f} {mlp[0]:8.1f} {full[0] * 50 / 1000:12.2f}")
        del block, rope
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
