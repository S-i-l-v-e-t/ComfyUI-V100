"""Compare comfy's Llama2_.forward against t5_split50's hand-written replacement
on the real minimax text encoder, same weights, same tokens."""
import importlib.util

import torch
import comfy.sd
import folder_paths
from comfy.text_encoders import llama as llamamod

TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
PROMPT = "make it night time, cinematic lighting"


def load_split_module():
    spec = importlib.util.spec_from_file_location("t5split50", "custom_nodes/t5_split50/__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flatten(cond):
    out = []

    def walk(v):
        if isinstance(v, torch.Tensor):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(cond)
    return out


def main():
    dev = "cuda:0"
    torch.set_grad_enabled(False)
    path = folder_paths.get_full_path_or_raise("text_encoders", TE)
    clip = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.MINIMAX, model_options={}, disable_dynamic=True)
    print(f"weights loaded: {clip.patcher.model.model_loaded_weight_memory / 1024**3:.2f} GB, "
          f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    clip.patcher.model.to(dev)

    tokens = clip.tokenize(PROMPT)
    orig_forward = llamamod.Llama2_.forward
    patched_forward = load_split_module()._patch_llama_forward  # installs it

    ref = flatten(clip.encode_from_tokens_scheduled(tokens))
    ref = [t.clone() for t in ref]

    patched_forward()
    assert llamamod.Llama2_.forward is not orig_forward, "patch did not apply"
    got = flatten(clip.encode_from_tokens_scheduled(tokens))
    llamamod.Llama2_.forward = orig_forward

    print(f"tensors: {len(ref)} vs {len(got)}")
    for i, (a, b) in enumerate(zip(ref, got)):
        rel = float((a.float() - b.float()).norm() / (a.float().norm() + 1e-12))
        mx = float((a.float() - b.float()).abs().max())
        print(f"  [{i}] {tuple(a.shape)} {a.dtype} rel={rel:.3e} max|diff|={mx:.4g}")


main()
