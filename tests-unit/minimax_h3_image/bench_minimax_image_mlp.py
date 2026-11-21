"""Where the bf16 MLP time goes: full module path vs its pieces at T=9409."""
import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.ops
from comfy.ldm.minimax.model import MLP

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
HIDDEN, FFN = 5376, 14336
T = 9409


def load_mlp(dtype, ops, device):
    mlp = MLP(HIDDEN, FFN, dtype=dtype, device=device, operations=ops)
    with safe_open(CKPT, "pt") as f:
        for name in ("fc1", "fc2"):
            key = f"blocks.0.mlp.{name}.weight"
            scale = f.get_tensor(f"blocks.0.mlp.{name}.weight_scale")
            weight = f.get_tensor(key).to(device=device, dtype=dtype) * scale.item()
            getattr(mlp, name).weight.data.copy_(weight)
    for parameter in mlp.parameters():
        parameter.requires_grad_(False)
    return mlp.eval()


def timeit(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def manual(mlp, x):
    gate, value = mlp.fc1(x).chunk(2, dim=-1)
    return mlp.fc2(F.silu(gate) * value)


def main():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name())
    print(f"{'dtype':>9} {'module':>8} {'manual':>8} {'fc1':>8} {'swiglu only':>12}")
    for dtype, ops in ((torch.bfloat16, comfy.ops.softbf16_ops), (torch.float16, comfy.ops.manual_cast)):
        mlp = load_mlp(dtype, ops, device)
        x = (torch.randn(T, HIDDEN, device=device) * 0.1).to(dtype)
        with torch.inference_mode():
            t_module = timeit(lambda: mlp(x))
            t_manual = timeit(lambda: manual(mlp, x))
            t_fc1 = timeit(lambda: mlp.fc1(x))
            hidden = mlp.fc1(x)
            gate, value = hidden.chunk(2, dim=-1)
            act = F.silu(gate) * value
            print(f"{str(dtype).split('.')[-1]:>9} {t_module:8.2f} {t_manual:8.2f} {t_fc1:8.2f}   {act.dtype} {act.shape}")
            t_fc2 = timeit(lambda: mlp.fc2(act))
            print(f"{'':>9} fc2 on swiglu out: {t_fc2:.2f} ms -> fc1+fc2 = {t_fc1 + t_fc2:.2f} ms")
        del mlp, x, hidden, act
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
