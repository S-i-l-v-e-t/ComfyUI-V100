"""Verify the diffusers -> ComfyUI remap (qkv order, fc1 half order) against the
official native MiniMax-H3-Image shard, fetching only the few rows needed."""
import json
import struct
import subprocess
import torch

REPO = "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/transformer/model-00001-of-00012.safetensors"
PROXY = "http://127.0.0.1:7897"
SRC = "models/diffusion_models/MiniMax-H3-Image-FP8.safetensors"


def curl_range(start, end, path):
    for cmd in (
        ["curl", "-sfL", "--proxy", PROXY, "-r", f"{start}-{end}", REPO, "-o", path],
        ["curl", "-sfL", "-r", f"{start}-{end}", REPO, "-o", path],
    ):
        if subprocess.call(cmd) == 0:
            return
    raise SystemExit(f"range fetch failed: {start}-{end}")


def header():
    curl_range(0, 4_000_000, "/tmp/shard_head.bin")
    raw = open("/tmp/shard_head.bin", "rb").read()
    size = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + size]), 8 + size


def read_rows(hdr, base, name, first, count):
    info = hdr[name]
    off = base + info["data_offsets"][0]
    row_bytes = info["shape"][1] * 2          # bf16
    curl_range(off + first * row_bytes, off + (first + count) * row_bytes - 1, "/tmp/rows.bin")
    data = open("/tmp/rows.bin", "rb").read()
    shape = (count, info["shape"][1])
    return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(shape).float()


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    hdr, base = header()
    from safetensors import safe_open
    with safe_open(SRC, "pt") as f:
        def dequant(name):
            q = f.get_tensor(name + "._weight_qdata").float()
            s = f.get_tensor(name + "._weight_scale").float()
            return (q * s).float()

        toq = dequant("transformer_blocks.0.attn.to_q")
        tok = dequant("transformer_blocks.0.attn.to_k")
        tov = dequant("transformer_blocks.0.attn.to_v")
        ff = dequant("transformer_blocks.0.ff.net.0.proj")
        native_ff = hdr["blocks.0.mlp.fc1.weight"]["shape"][0] // 2

    print("=== qkv_proj 行块归属（native 前三段 vs 源 to_q/to_k/to_v 前 8 行）===")
    for label, first in (("q (0)", 0), ("k (7168)", 7168), ("v (14336)", 14336)):
        rows = read_rows(hdr, base, "blocks.0.attn.qkv_proj.weight", first, 8)
        print(f"{label:>11}: q={rel(rows, toq[:8]):6.3f}  k={rel(rows, tok[:8]):6.3f}  v={rel(rows, tov[:8]):6.3f}")

    print("\n=== mlp.fc1 半区（native 上半 vs 源上半/下半）===")
    top = read_rows(hdr, base, "blocks.0.mlp.fc1.weight", 0, 8)
    bottom = read_rows(hdr, base, "blocks.0.mlp.fc1.weight", native_ff, 8)
    print(f"native 上: 源上={rel(top, ff[:8]):6.3f}  源下={rel(top, ff[native_ff:native_ff + 8]):6.3f}")
    print(f"native 下: 源上={rel(bottom, ff[:8]):6.3f}  源下={rel(bottom, ff[native_ff:native_ff + 8]):6.3f}")


if __name__ == "__main__":
    main()
