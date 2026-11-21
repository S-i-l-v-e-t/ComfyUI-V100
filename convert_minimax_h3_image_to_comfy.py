#!/usr/bin/env python3
"""Repack the diffusers (unsloth) export of MiniMax-H3-Image into ComfyUI's
MiniMaxH3Image checkpoint layout.

The source is a `MiniMaxH3ImageTransformer2DModel` state dict whose fp8 weights
are stored torchao style: `<module>._weight_qdata` (float8_e4m3fn [out, in]) plus
`<module>._weight_scale` (float32 [out, 1], one scale per output row).

ComfyUI stores the same layers as `<module>.weight` (float8_e4m3fn),
`<module>.weight_scale` (one scalar per tensor) and a `<module>.comfy_quant`
metadata blob, under its own module names. Three families need more than a
rename:

  * attn.to_q / to_k / to_v -> one fused attn.qkv_proj ([q; k; v] rows)
  * ff.net.0.proj           -> mlp.fc1 with the SwiGLU halves swapped
                               (diffusers fuses [value; gate], ComfyUI [gate; value])
  * pdd_proj_out.{i}        -> final_layer.video_out, heads stacked in step order,
                               with heads 1..N stored as offsets from head 0
                               (ComfyUI adds head 0 back, so raw full heads double it)

`rope.inv_freq` is not in the source (diffusers rebuilds it at load time) but
ComfyUI reads it from the checkpoint, so it is regenerated from rope_theta.

Usage:
    python convert_minimax_h3_image_to_comfy.py SRC.safetensors DST.safetensors
    python convert_minimax_h3_image_to_comfy.py SRC.safetensors DST.safetensors --dry-run
"""

import argparse
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
QUANT_CONF = json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")

ROPE_THETA = 10000.0
ROPE_FREQ_DIM = 16

# per-block tensors that keep their bytes
PLAIN_SUFFIX = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "attn.norm_q.weight": "attn.q_norm.weight",
    "attn.norm_k.weight": "attn.k_norm.weight",
}
ADALN_SUFFIX = {
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
}
# quantized linears that only need a rename
QUANT_SUFFIX = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.2": "mlp.fc2",
}


class Source:
    def __init__(self, path):
        self.handle = safe_open(path, framework="pt")
        self.keys = set(self.handle.keys())
        self.used = set()
        self.meta = json.loads(self.handle.metadata().get("unsloth_metadata", "{}"))

    def take(self, name):
        self.used.add(name)
        return self.handle.get_tensor(name)

    def take_weight(self, module):
        """Per-row fp8 (qdata, scale[out, 1]) -> fp32 weight."""
        self.used.update((module + "._weight_qdata", module + "._weight_scale"))
        qdata = self.handle.get_tensor(module + "._weight_qdata")
        scale = self.handle.get_tensor(module + "._weight_scale")
        return qdata.to(torch.float32) * scale


def emit_fp8(out, name, weight, errors):
    """Store an fp32 weight the way ComfyUI's float8_e4m3fn layers expect it."""
    scale = weight.abs().amax().to(torch.float32) / FP8_MAX
    qdata = (weight / scale).to(FP8_DTYPE)
    errors.append((name, float((weight - qdata.to(torch.float32) * scale).norm() / weight.norm())))
    if out is not None:
        out[name + ".weight"] = qdata
        out[name + ".weight_scale"] = scale
        out[name + ".comfy_quant"] = torch.tensor(list(QUANT_CONF), dtype=torch.uint8)


def emit_block(src, out, errors, src_prefix, dst_prefix, adaln):
    for source, target in PLAIN_SUFFIX.items():
        tensor = src.take(f"{src_prefix}.{source}")
        if out is not None:
            out[f"{dst_prefix}.{target}"] = tensor
    if adaln:
        for source, target in ADALN_SUFFIX.items():
            tensor = src.take(f"{src_prefix}.{source}")
            if out is not None:
                out[f"{dst_prefix}.{target}"] = tensor

    q = src.take_weight(f"{src_prefix}.attn.to_q")
    k = src.take_weight(f"{src_prefix}.attn.to_k")
    v = src.take_weight(f"{src_prefix}.attn.to_v")
    emit_fp8(out, f"{dst_prefix}.attn.qkv_proj", torch.cat([q, k, v], dim=0), errors)

    for source, target in QUANT_SUFFIX.items():
        emit_fp8(out, f"{dst_prefix}.{target}", src.take_weight(f"{src_prefix}.{source}"), errors)

    value, gate = src.take_weight(f"{src_prefix}.ff.net.0.proj").chunk(2, dim=0)
    emit_fp8(out, f"{dst_prefix}.mlp.fc1", torch.cat([gate, value], dim=0), errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="diffusers/unsloth MiniMax-H3-Image transformer .safetensors")
    parser.add_argument("dst", help="output ComfyUI checkpoint (single .safetensors)")
    parser.add_argument("--dry-run", action="store_true", help="report the mapping and requant error without writing")
    args = parser.parse_args()

    src = Source(args.src)
    out = None if args.dry_run else {}
    errors = []
    pdd_steps = int(src.meta["pdd_num_heads"])
    pdd_sigmas = [float(s) for s in src.meta["pdd_sigmas"]]

    def take(name, target):
        tensor = src.take(name)
        if out is not None:
            out[target] = tensor

    take("proj_in.weight", "video_patch_proj.weight")
    take("proj_in.bias", "video_patch_proj.bias")
    take("context_embedder.bias", "condition_proj.bias")
    take("time_embedder.table", "adaln_t_table")
    take("token_refiner.final_norm.weight", "token_refiner.final_norm.weight")
    take("norm_out.norm.weight", "final_layer.norm.weight")
    take("norm_out.linear.weight", "final_layer.adaln_proj.linear.weight")
    take("norm_out.linear.bias", "final_layer.adaln_proj.linear.bias")
    emit_fp8(out, "condition_proj", src.take_weight("context_embedder"), errors)

    heads = [src.take(f"pdd_proj_out.{i}.weight").float() for i in range(pdd_steps)]
    biases = [src.take(f"pdd_proj_out.{i}.bias").float() for i in range(pdd_steps)]
    if out is not None:
        # unsloth fuses the published head pack into the base weights, so every
        # pdd_proj_out.{i} is a complete head. ComfyUI's PDD head bank keeps the
        # later blocks as offsets from block 0 and adds block 0 back itself.
        out["final_layer.video_out.weight"] = torch.cat(
            [heads[0]] + [h - heads[0] for h in heads[1:]], dim=0)
        out["final_layer.video_out.bias"] = torch.cat(
            [biases[0]] + [b - biases[0] for b in biases[1:]], dim=0)
    del heads, biases

    num_layers = 1 + max(int(k.split(".")[1]) for k in src.keys if k.startswith("transformer_blocks."))
    num_refiner = 1 + max(int(k.split(".")[2]) for k in src.keys if k.startswith("token_refiner.refiner_blocks."))
    for i in range(num_layers):
        emit_block(src, out, errors, f"transformer_blocks.{i}", f"blocks.{i}", adaln=True)
    for i in range(num_refiner):
        emit_block(src, out, errors, f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}", adaln=False)

    exponent = torch.arange(0, 2 * ROPE_FREQ_DIM, 2, dtype=torch.float32) / (2 * ROPE_FREQ_DIM)
    if out is not None:
        out["rope.inv_freq"] = 1.0 / (ROPE_THETA ** exponent)

    unused = src.keys - src.used
    if unused:
        raise KeyError(f"{len(unused)} source tensor(s) not converted, e.g. {sorted(unused)[:5]}")

    print(f"{num_layers} blocks, {num_refiner} refiner layers, {pdd_steps} PDD heads")
    worst = max(errors, key=lambda item: item[1])
    print(f"{len(errors)} quantized tensors, worst per-tensor fp8 requant rel. L2 error: {worst[1]:.5f} ({worst[0]})")
    if out is None:
        print("dry run: nothing written")
        return

    metadata = {"config": json.dumps({"transformer": {"pdd_steps": pdd_steps, "pdd_sigmas": pdd_sigmas}})}
    save_file(out, args.dst, metadata=metadata)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
