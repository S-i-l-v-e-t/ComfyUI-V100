"""Reference run through the plain ComfyUI path (no raylight), for comparison.

Same checkpoint, same prompt, same reference image, official node classes,
6 steps at cfg 1, then decode. The 20 GB fp8 UNET and the 32B text encoder cannot
co-reside on one V100, so the prompt is encoded first and the encoder unloaded
before the UNET loads.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pass  # 不强制任何 vram 模式，按 comfy 默认行为流式加载

import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

UNET = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
CLIP = "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "models/vae/minimax_h3_image_vae_fp16.safetensors"
IMAGE = "input/vintage_yellow_taxi.png"
PROMPT = ("Replace the vintage yellow taxi in <Picture 1> with a pastel pink ice-cream truck "
          "with a scalloped awning and a glowing service window, sitting in the same side profile at the curb.")
WIDTH, HEIGHT = 2048, 1152
STEPS, CFG, SEED = 6, 1.0, 42


def load_image(path, width, height):
    img = Image.open(path).convert("RGB")
    scale = (width * height / (img.width * img.height)) ** 0.5
    size = (round(img.width * scale / 16) * 16, round(img.height * scale / 16) * 16)
    img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def main():
    image = load_image(IMAGE, WIDTH, HEIGHT)
    print("== loading VAE", flush=True)
    vae_sd, vae_md = comfy.utils.load_torch_file(VAE, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_md)

    print("== loading text encoder", flush=True)
    clip = comfy.sd.load_clip(ckpt_paths=[CLIP], embedding_directory=[], clip_type=comfy.sd.CLIPType.MINIMAX)
    print("== encoding prompt + reference image", flush=True)
    result = MiniMaxH3ImageConditioning.execute(clip=clip, prompt=PROMPT, width=WIDTH, height=HEIGHT,
                                                vae=vae, images={"image_1": image})
    cond, latent = getattr(result, "result", result)
    negative = [[torch.zeros_like(tensor), meta] for tensor, meta in cond]
    del clip
    mm.unload_all_models()
    mm.soft_empty_cache()

    print("== loading UNET", flush=True)
    model = comfy.sd.load_diffusion_model(UNET)

    print(f"== sampling {STEPS} steps at {WIDTH}x{HEIGHT}", flush=True)
    noise = comfy.sample.prepare_noise(latent["samples"], SEED)
    samples = comfy.sample.sample(model, noise, STEPS, CFG, "euler", "simple", cond, negative,
                                  latent["samples"], denoise=1.0)
    print("== decoding", flush=True)
    out = vae.decode(samples)
    arr = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save("/tmp/official_path.png")
    print("wrote /tmp/official_path.png", arr.shape, flush=True)


if __name__ == "__main__":
    main()
