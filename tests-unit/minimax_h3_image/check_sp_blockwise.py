"""Find the first block where the SP forward diverges from the full-sequence one.

Records each block's input on both paths at real weights and compares per block,
so the split/pad/rope/modulation step that breaks shows up directly.

Run: PYTHONPATH=custom_nodes/raylight/src:. .venv/bin/torchrun --nproc_per_node=2 \
        tests-unit/check_sp_blockwise.py
"""
import os
import types

import numpy as np
import torch
from PIL import Image

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
VAE = "models/vae/minimax_h3_image_vae_fp16.safetensors"
IMAGE = "input/vintage_yellow_taxi.png"
TEXT_LEN, LAT = 64, 32


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    import raylight.distributed_modules.attention as xfuser_attn
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    xfuser_attn.set_attn_type("TORCH_EFFICIENT")
    xfuser_attn.set_sync_ulysses(False)
    init_distributed_environment(rank=local_rank, world_size=world)
    initialize_model_parallel(data_parallel_degree=1, sequence_parallel_degree=world,
                              classifier_free_guidance_degree=1, ring_degree=1,
                              ulysses_degree=world, pipeline_parallel_degree=1)

    import comfy.sd, comfy.utils
    import raylight.diffusion_models.minimax.xdit_context_parallel as xcp

    patcher = comfy.sd.load_diffusion_model(CKPT)
    base = patcher.model
    model = base.diffusion_model
    model.raylight_fp32_residual = False

    vae_sd, vae_md = comfy.utils.load_torch_file(VAE, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_md)
    img = Image.open(IMAGE).convert("RGB").resize((LAT * 16, LAT * 16), Image.LANCZOS)
    px = torch.from_numpy(np.asarray(img).copy()).float().div_(255.0).unsqueeze(0)

    torch.manual_seed(0)
    x = torch.randn(1, 24, 1, LAT, LAT, device=dev).to(base.get_dtype())
    ctx = (torch.randn(1, TEXT_LEN, 5120, device=dev) * 0.1).to(base.get_dtype())
    t = torch.tensor([500.0], device=dev)
    opts = {"sample_sigmas": torch.tensor(model.pdd_sigmas, device=dev)}
    with torch.inference_mode():
        ref_latent = vae.encode(px).to(base.get_dtype())
    payload = {"refs": [{"kind": "image", "latent_h": LAT, "latent_w": LAT}],
               "cond_video_latents": [ref_latent]}

    recorded = {}
    tag = ["full"]
    for i, block in enumerate(model.blocks):
        original = block.forward

        def wrapped(self, x_, *args, _i=i, _orig=original, **kwargs):
            recorded.setdefault(_i, {})[tag[0]] = x_.detach().float().cpu()
            return _orig(x_, *args, **kwargs)

        block.forward = types.MethodType(wrapped, block)

    with torch.inference_mode():
        model._forward(x.clone(), t, ctx, transformer_options=dict(opts), minimax_payload=dict(payload))
    tag[0] = "sp"
    # the SP run must use the SP attention (cross-rank K/V); the reference must not
    for block in model.blocks:
        block.attn.forward = types.MethodType(xcp.usp_attn_forward, block.attn)
    with torch.inference_mode():
        xcp.usp_image_forward.__get__(model)(x.clone(), t, ctx, transformer_options=dict(opts), minimax_payload=dict(payload))

    from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size
    seq_len = TEXT_LEN + 2 * (LAT // 2) * (LAT // 2)
    local = -(-seq_len // get_sequence_parallel_world_size())
    start = get_sequence_parallel_rank() * local
    for i in sorted(recorded):
        full = recorded[i]["full"]
        sp = recorded[i]["sp"]
        ref = full[start:start + sp.shape[0]]
        rel = float((sp - ref).norm() / ref.norm())
        print(f"[rank{local_rank}] block {i:2d}: full{tuple(full.shape)} local{tuple(sp.shape)} "
              f"rel={rel:.3e} max|diff|={float((sp - ref).abs().max()):.4g}", flush=True)
        if rel > 1e-2:
            print(f"[rank{local_rank}]   -> 第一处分叉是 block {i}", flush=True)
            break


if __name__ == "__main__":
    main()
