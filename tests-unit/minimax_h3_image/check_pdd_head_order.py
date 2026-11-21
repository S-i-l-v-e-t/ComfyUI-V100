"""Check that the source pdd_proj_out.{i} heads match the official fused head order."""
import torch
from safetensors import safe_open

with safe_open("/tmp/pdd_fused.safetensors", "pt") as f:
    official = f.get_tensor("video_out.weight").float()   # [6, 96, 5376]
    official_b = f.get_tensor("video_out.bias").float()   # [6, 96]

src = []
src_b = []
with safe_open("models/diffusion_models/MiniMax-H3-Image-FP8.safetensors", "pt") as f:
    for i in range(6):
        src.append(f.get_tensor(f"pdd_proj_out.{i}.weight").float())
        src_b.append(f.get_tensor(f"pdd_proj_out.{i}.bias").float())

print("源 pdd_proj_out.i  vs  官方 fused head j  的相对 L2 误差（行=i，列=j）")
print("        " + "".join(f"   head{j}" for j in range(6)))
for i in range(6):
    row = "".join(f" {float((src[i] - official[j]).norm() / official[j].norm()):7.3f}" for j in range(6))
    print(f"src {i}: {row}")

print("\nbias 的 L2 距离")
for i in range(6):
    row = "".join(f" {float((src_b[i] - official_b[j]).norm()):7.2f}" for j in range(6))
    print(f"src {i}: {row}")
