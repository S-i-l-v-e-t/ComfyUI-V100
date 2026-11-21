"""Probe the native qkv_proj layout (block vs per-head interleaved) against the
source to_q/to_k/to_v, so the ComfyUI split order can be confirmed."""
import json
import struct
import subprocess
import torch
from safetensors import safe_open

REPO = "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/transformer/model-00001-of-00012.safetensors"
PROXY = "http://127.0.0.1:7897"
SRC = "models/diffusion_models/MiniMax-H3-Image-FP8.safetensors"
NAME = "blocks.0.attn.qkv_proj.weight"


def curl_range(start, end, path):
    for cmd in (
        ["curl", "-sfL", "--proxy", PROXY, "-r", f"{start}-{end}", REPO, "-o", path],
        ["curl", "-sfL", "-r", f"{start}-{end}", REPO, "-o", path],
    ):
        if subprocess.call(cmd) == 0:
            return
    raise SystemExit("range fetch failed")


def main():
    curl_range(0, 4_000_000, "/tmp/shard_head.bin")
    raw = open("/tmp/shard_head.bin", "rb").read()
    size = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + size])
    base = 8 + size
    info = hdr[NAME]
    rows, cols = info["shape"]
    print(f"{NAME}: {info['dtype']} shape={info['shape']}")
    item = 2 if info["dtype"] == "BF16" else 4
    row_bytes = cols * item

    def read(first, count=8):
        off = base + info["data_offsets"][0] + first * row_bytes
        curl_range(off, off + count * row_bytes - 1, "/tmp/rows.bin")
        data = open("/tmp/rows.bin", "rb").read()
        dt = torch.bfloat16 if item == 2 else torch.float32
        return torch.frombuffer(bytearray(data), dtype=dt).view(count, cols).float()

    with safe_open(SRC, "pt") as f:
        def dequant(n):
            return (f.get_tensor(n + "._weight_qdata").float() * f.get_tensor(n + "._weight_scale")).float()
        q, k, v = (dequant(f"transformer_blocks.0.attn.to_{n}") for n in ("q", "k", "v"))

    def rel(a, b):
        return float((a - b).norm() / b.norm())

    head_dim = 128
    heads = cols // head_dim
    starts = {
        "0 (block q / head0 q)": 0,
        "128 (head0 k if 交织)": head_dim,
        "256 (head0 v if 交织)": 2 * head_dim,
        f"{heads * head_dim} (block k)": heads * head_dim,
        f"{2 * heads * head_dim} (block v)": 2 * heads * head_dim,
    }
    print("\n原生行起点 vs 源 to_q/to_k/to_v 前 8 行")
    for label, first in starts.items():
        rows_ = read(first)
        print(f"{label:>28}: q={rel(rows_, q[:8]):6.3f}  k={rel(rows_, k[:8]):6.3f}  v={rel(rows_, v[:8]):6.3f}")


if __name__ == "__main__":
    main()
