"""Compare raylight's image forward with ComfyUI's at real weights and shapes.

No FSDP, no text encoder: each rank loads a full copy of the real fp8 checkpoint,
the reference latent comes from the real VAE, and the text context is random but
correctly shaped. Only the forward differs between the two paths.

Run: PYTHONPATH=custom_nodes/raylight/src:. .venv/bin/torchrun --nproc_per_node=2 \
        tests-unit/check_real_forward_usp.py
"""
import os
import types

import numpy as np
import torch
from PIL import Image

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
VAE = "models/vae/minimax_h3_image_vae_fp16.safetensors"
IMAGE = "input/vintage_yellow_taxi.png"
TEXT_LEN = 64
LAT = 32


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

    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    import raylight.diffusion_models.minimax.xdit_context_parallel as xcp

    if local_rank == 0:
        print("loading checkpoint on every rank (no FSDP)", flush=True)
    patcher = comfy.sd.load_diffusion_model(CKPT)
    base = patcher.model
    model = base.diffusion_model
    model.eval()
    if local_rank == 0:
        print("model dtype:", base.get_dtype(), "| manual cast:", getattr(base, "manual_cast_dtype", None), flush=True)

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

    for fp32_residual in (False, True):
        model.raylight_fp32_residual = fp32_residual
        with torch.inference_mode():
            ref = model._forward(x.clone(), t, ctx, transformer_options=dict(opts), minimax_payload=dict(payload))
        for block in model.blocks:
            block.attn.forward = types.MethodType(xcp.usp_attn_forward, block.attn)
        with torch.inference_mode():
            out = xcp.usp_image_forward.__get__(model)(
                x.clone(), t, ctx, transformer_options=dict(opts), minimax_payload=dict(payload))
        gathered = [torch.empty_like(out) for _ in range(world)]
        torch.distributed.all_gather(gathered, out.contiguous())
        symmetric = all(torch.equal(gathered[0], g) for g in gathered)
        rf, of = ref.float(), out.float()
        rel = float((of - rf).norm() / rf.norm())
        print(f"[rank{local_rank}] fp32_residual={fp32_residual} out{tuple(of.shape)} "
              f"finite={bool(torch.isfinite(of).all())} symmetric={symmetric} "
              f"rel={rel:.3e} max|diff|={float((of - rf).abs().max()):.4g} "
              f"mean|x| ref={float(rf.abs().mean()):.4f} ray={float(of.abs().mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
