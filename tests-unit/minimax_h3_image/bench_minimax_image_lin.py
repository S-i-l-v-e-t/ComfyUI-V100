"""Per-linear cost of MiniMax-H3-Image block shapes on this machine.

Compares the dtype paths the two MiniMax families take today: bf16 (softbf16,
what the image model loads with) against fp16 (the video model's W8A16 path).
"""
import torch
from safetensors import safe_open

import comfy.ops  # installs the sm70 bf16 -> softbf16 F.linear patch

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
LAYERS = {
    "qkv": ("blocks.0.attn.qkv_proj", 9409),
    "out_proj": ("blocks.0.attn.out_proj", 9409),
    "fc1": ("blocks.0.mlp.fc1", 9409),
    "fc2": ("blocks.0.mlp.fc2", 9409),
}
ITERS = 6


def load(name, device):
    with safe_open(CKPT, "pt") as f:
        q = f.get_tensor(name + ".weight")
        scale = f.get_tensor(name + ".weight_scale").float().to(device)
    return q.to(device), scale


def timeit(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main():
    device = torch.device("cuda")
    print(f"{torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}")
    print(f"{'layer':9} {'shape':>18} {'bf16':>8} {'fp16':>8} {'fp16scaled':>11} {'fp8->bf16':>10}")
    totals = {"bf16": 0.0, "fp16": 0.0, "fp16scaled": 0.0, "fp8": 0.0}
    for name, (key, T) in LAYERS.items():
        q, scale = load(key, device)
        out_f, in_f = q.shape
        w_bf16 = (q.to(torch.bfloat16) * scale.to(torch.bfloat16))
        w_fp16 = (q.to(torch.float16) * scale.to(torch.float16))
        x_bf16 = (torch.randn(T, in_f, device=device) * 0.05).to(torch.bfloat16)
        x_fp16 = x_bf16.to(torch.float16)
        x_fp32 = x_bf16.float()

        t_bf16 = timeit(lambda: torch.nn.functional.linear(x_bf16, w_bf16))
        t_fp16 = timeit(lambda: torch.nn.functional.linear(x_fp16, w_fp16))
        t_scaled = timeit(lambda: comfy.ops.fp16_scaled_linear(x_fp32, w_fp16))
        t_fp8 = timeit(lambda: torch.nn.functional.linear(x_bf16, q))
        flops = 2 * T * in_f * out_f
        print(f"{name:9} {str((T, in_f, out_f)):>18} "
              f"{t_bf16:8.2f} {t_fp16:8.2f} {t_scaled:11.2f} {t_fp8:10.2f}")
        for k, t in (("bf16", t_bf16), ("fp16", t_fp16), ("fp16scaled", t_scaled), ("fp8", t_fp8)):
            totals[k] += t
        print(f"{'':9} {'':>18} tf/s: bf16 {flops / t_bf16 / 1e9:.1f}  fp16 {flops / t_fp16 / 1e9:.1f}"
              f"  fp16scaled {flops / t_scaled / 1e9:.1f}  fp8->bf16 {flops / t_fp8 / 1e9:.1f}")

    print()
    print("one block (4 linears) =", {k: round(v, 1) for k, v in totals.items()}, "ms")
    print("50 blocks            =", {k: round(v * 50 / 1000, 2) for k, v in totals.items()}, "s per step"
          " (linear only, 1 GPU, single-rank T=9409)")


if __name__ == "__main__":
    main()
