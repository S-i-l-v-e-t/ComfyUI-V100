"""Is raylight's attention replacement numerically equal to ComfyUI's Attention.forward?

The USP test patches block.attn.forward on both sides before comparing, so this
piece was never checked on its own. Single rank, bf16 like the real run.

Run: PYTHONPATH=custom_nodes/raylight/src:. .venv/bin/torchrun --nproc_per_node=1 \
        tests-unit/check_usp_attn_equiv.py
"""
import os
import types

import torch

HIDDEN, HEADS, HEAD_DIM, S, ROPE_DIM = 512, 8, 64, 512, 32


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    import comfy.ops
    from comfy.ldm.minimax.model import Attention, rope_rotation_table
    import raylight.distributed_modules.attention as xfuser_attn
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

    xfuser_attn.set_attn_type("TORCH_EFFICIENT")
    xfuser_attn.set_sync_ulysses(False)
    init_distributed_environment(rank=local_rank, world_size=world)
    initialize_model_parallel(data_parallel_degree=1, sequence_parallel_degree=world,
                              classifier_free_guidance_degree=1, ring_degree=1,
                              ulysses_degree=world, pipeline_parallel_degree=1)
    import raylight.diffusion_models.minimax.xdit_context_parallel as xcp

    torch.manual_seed(0)
    attn = Attention(HIDDEN, HEADS, HEAD_DIM, 1e-6, gate_compress=False,
                     dtype=torch.bfloat16, device=dev, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0.0, 0.05)
            p.requires_grad_(False)
        # norms want ~1 so the RMSNorm output is realistic
        attn.q_norm.weight.fill_(1.0)
        attn.k_norm.weight.fill_(1.0)

    x = (torch.randn(S, HIDDEN, device=dev) * 0.5).to(torch.bfloat16)
    angles = torch.randn(S, ROPE_DIM, device=dev, dtype=torch.float32).cumsum(0)
    rope = rope_rotation_table(angles, torch.bfloat16)

    with torch.inference_mode():
        ref = attn(x, rope_freqs=rope)
        comfy_attn = attn.forward
        attn.forward = types.MethodType(xcp.usp_attn_forward, attn)
        out = attn(x, rope_freqs=rope)

    ref_f, out_f = ref.float(), out.float()
    rel = float((out_f - ref_f).norm() / ref_f.norm())
    print(f"comfy Attention.forward : mean|x|={float(ref_f.abs().mean()):.6f} max|x|={float(ref_f.abs().max()):.4f}")
    print(f"raylight usp_attn_forward: mean|x|={float(out_f.abs().mean()):.6f} max|x|={float(out_f.abs().max()):.4f}")
    print(f"相对 L2 误差 {rel:.3e} | max|diff|={float((out_f - ref_f).abs().max()):.4g} | finite={bool(torch.isfinite(out).all())}")


if __name__ == "__main__":
    main()
