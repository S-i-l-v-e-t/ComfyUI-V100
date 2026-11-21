"""Two-process USP correctness test for MiniMax-H3-Image.

Runs the sequence-parallel forward (`raylight.diffusion_models.minimax.
xdit_context_parallel.usp_image_forward`) under ulysses=2 and compares it with
the plain ComfyUI forward of the same weights. Covers the three things a
single-rank run cannot:

  * every rank produces the same gathered output (collective symmetry),
  * the row split/pad/gather path is right for sequence lengths that do not
    divide by the world size,
  * the packed layout with reference images (ref_img segment) splits correctly.

Run:
    PYTHONPATH=custom_nodes/raylight/src:. .venv/bin/torchrun --nproc_per_node=2 \
        tests-unit/test_minimax_h3_image_usp.py
"""

import os
import types

import torch
import torch.distributed as dist

TEXT_DIM = 32
LATENT_CH = 24


def build_model(dev):
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3ImageModel

    torch.manual_seed(0)
    model = MiniMaxH3ImageModel(
        pdd_sigmas=[1.0, 0.5, 0.0], pdd_steps=2,
        hidden_size=64, num_layers=2, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=32, ffn_hidden_size=128,
        latents_dim=LATENT_CH, patch_size=(1, 2, 2), text_dim=TEXT_DIM,
        timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
        rope_inv_freq_len=4, dtype=torch.float32, device=dev,
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.raylight_fp32_residual = False
    with torch.no_grad():
        # disable_weight_init leaves the params uninitialized; both ranks must start equal
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, float(os.environ.get("INIT_STD", "0.05")))
            else:
                p.fill_(1.0)
        # the checkpoint supplies these, there is nothing to load here
        model.rope.inv_freq.normal_()
    return model


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
    initialize_model_parallel(
        data_parallel_degree=1,
        sequence_parallel_degree=world,
        classifier_free_guidance_degree=1,
        ring_degree=1,
        ulysses_degree=world,
        pipeline_parallel_degree=1,
    )

    import comfy.model_management as model_management
    model_management.in_training = True  # out-of-place rope, identical math
    import comfy.ldm.minimax.model as minimax_model
    from comfy.ldm.modules.attention import attention_pytorch

    # both paths then share the attention kernel, so the comparison isolates the split
    minimax_model.optimized_attention = attention_pytorch

    import raylight.diffusion_models.minimax.xdit_context_parallel as xcp

    model = build_model(dev)
    for block in model.blocks:
        block.attn.forward = types.MethodType(xcp.usp_attn_forward, block.attn)

    def check(name, payload, x, text_len):
        ctx = torch.randn(1, text_len, TEXT_DIM, device=dev)
        t = torch.tensor([500.0], device=dev)
        opts = {"sample_sigmas": torch.tensor(model.pdd_sigmas, device=dev)}
        with torch.no_grad():
            ref = model._forward(x.clone(), t, ctx, transformer_options=dict(opts), minimax_payload=dict(payload))
            out = xcp.usp_image_forward.__get__(model)(
                x.clone(), t, ctx, transformer_options=dict(opts), minimax_payload=dict(payload))
        gathered = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(gathered, out.contiguous())
        symmetric = all(torch.equal(gathered[0], g) for g in gathered)
        rel = float((out - ref).norm() / ref.norm())
        print(f"[rank{local_rank}] {name:34s} shape={tuple(out.shape)} "
              f"rank_symmetric={symmetric} rel_err_vs_full_seq={rel:.2e}", flush=True)
        assert torch.isfinite(out).all() and torch.isfinite(ref).all(), f"{name}: non-finite output"
        assert symmetric, f"{name}: ranks disagree"
        assert rel < 1e-2, f"{name}: rel err {rel:.3e}"

    x = torch.randn(1, LATENT_CH, 1, 12, 12, device=dev)  # 12 不是 patch 2 的倍数
    check("t2i, 44 rows (divides evenly)", {}, x, text_len=8)
    check("t2i, 43 rows (odd -> padded)", {}, x, text_len=7)
    check("i2i, 77 rows with ref_img segment",
          {"refs": [{"kind": "image", "latent_h": 12, "latent_w": 12}],
           "cond_video_latents": [torch.randn(1, LATENT_CH, 1, 12, 12, device=dev)]},
          x, text_len=5)
    check("t2i, 1 row target only", {}, torch.randn(1, LATENT_CH, 1, 2, 2, device=dev), text_len=3)

    x_big = torch.randn(1, LATENT_CH, 1, 32, 32, device=dev)  # 256 patch 行 + 7 text = 263 行
    check("t2i, 263 rows (1 pad of 264)", {}, x_big, text_len=7)

    if local_rank == 0:
        print("OK: usp_image_forward matches the full-sequence forward under 2 ranks")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
