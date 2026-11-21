"""Spot-check the quantized text encoder: dequantize a layer with comfy's own
nvfp4 kernel and compare it against the official bf16 weights (a few rows only,
fetched by HTTP range so nothing large is materialized)."""
import json
import struct
import subprocess

import torch
import comfy_kitchen
from safetensors import safe_open

SRC = "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
LAYER = "model.layers.0.mlp.down_proj"
INDEX = "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model.safetensors.index.json"
PROXY = "http://127.0.0.1:7897"
ROWS = 8


def curl(url, path, rng=None):
    base = ["curl", "-sfL"]
    for cmd in (base + ["--proxy", PROXY], base):
        args = cmd + (["-r", rng] if rng else []) + [url, "-o", path]
        if subprocess.call(args) == 0:
            return
    raise SystemExit(f"fetch failed: {url}")


def main():
    with safe_open(SRC, "pt") as f:
        q = f.get_tensor(LAYER + ".weight").to("cuda")
        block = f.get_tensor(LAYER + ".weight_scale").to("cuda")
        gscale = f.get_tensor(LAYER + ".weight_scale_2").float().to("cuda")
        pre = f.get_tensor(LAYER + ".pre_quant_scale").float().to("cuda")
    print(f"本地: qdata{tuple(q.shape)} block_scale{tuple(block.shape)} global={gscale.tolist()} "
          f"pre_quant_scale{tuple(pre.shape)}")

    with torch.no_grad():
        w = comfy_kitchen.dequantize_nvfp4(q, gscale, block, output_type=torch.float32)
    print(f"反量化后 {tuple(w.shape)} finite={bool(torch.isfinite(w).all())} "
          f"std={float(w.std()):.5f} absmax={float(w.abs().max()):.4f}")

    curl(INDEX, "/tmp/te_index.json")
    weight_map = json.load(open("/tmp/te_index.json"))["weight_map"]
    shard = weight_map[LAYER + ".weight"]
    url = f"https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/{shard}"
    curl(url, "/tmp/te_shard_head.bin", "0-4000000")
    raw = open("/tmp/te_shard_head.bin", "rb").read()
    size = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + size])
    info = hdr[LAYER + ".weight"]
    base = 8 + size
    cols = info["shape"][1]
    row_bytes = cols * 2
    curl(url, "/tmp/te_rows.bin", f"{base + info['data_offsets'][0]}-{base + info['data_offsets'][0] + ROWS * row_bytes - 1}")
    off = torch.frombuffer(bytearray(open("/tmp/te_rows.bin", "rb").read()), dtype=torch.bfloat16)
    official = off.view(ROWS, cols).float()
    print(f"官方 bf16 {tuple(official.shape)} std={float(official.std()):.5f} "
          f"absmax={float(official.abs().max()):.4f}")

    for label, cand in (("直接反量化", w[:ROWS]), ("× pre_quant_scale", w[:ROWS] * pre[None, :])):
        rel = float((cand - official).norm() / official.norm())
        print(f"{label:>20}: 相对 L2 误差 {rel:.4f}")


if __name__ == "__main__":
    main()
