"""Does xfuser's ulysses attention agree with plain full-sequence attention here?

Real geometry (56 heads x 128), sequence split across ranks the way raylight splits
it. Compares against an fp32 full-sequence reference computed on every rank.

Run: PYTHONPATH=custom_nodes/raylight/src:. .venv/bin/torchrun --nproc_per_node=2 \
        tests-unit/check_attn_sp.py
"""
import os

import torch


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    import raylight.distributed_modules.attention as xfuser_attn
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel, get_sp_group

    xfuser_attn.set_attn_type("TORCH_EFFICIENT")
    xfuser_attn.set_sync_ulysses(False)
    init_distributed_environment(rank=local_rank, world_size=world)
    initialize_model_parallel(data_parallel_degree=1, sequence_parallel_degree=world,
                              classifier_free_guidance_degree=1, ring_degree=1,
                              ulysses_degree=world, pipeline_parallel_degree=1)
    import raylight.diffusion_models.minimax.xdit_context_parallel as xcp

    heads, dim, total = 56, 128, 576
    local = -(-total // world)
    torch.manual_seed(0)
    q = torch.randn(1, heads, total, dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, heads, total, dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, heads, total, dim, device=dev, dtype=torch.bfloat16)

    # fp16 full-sequence reference (same precision as the SP call, so only the split is tested)
    with torch.inference_mode():
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)).float()

    start = local_rank * local
    with torch.inference_mode():
        out = xcp.xfuser_optimized_attention(
            q[:, :, start:start + local].to(torch.float16),
            k[:, :, start:start + local].to(torch.float16),
            v[:, :, start:start + local].to(torch.float16),
            heads, skip_reshape=True,
        )
        # xfuser returns [B, S, H*D] (the caller reshapes after out_proj)
        assert out.ndim == 3, tuple(out.shape)
        gathered = get_sp_group().all_gather(out.float().contiguous(), dim=1)[:, :total]
        ref = ref.transpose(1, 2).reshape(gathered.shape)

    rel = float((gathered - ref).norm() / ref.norm())
    print(f"[rank{local_rank}] heads={heads} dim={dim} total={total} local={local} "
          f"out{tuple(out.shape)} rel_vs_full={rel:.3e} "
          f"max|diff|={float((gathered - ref).abs().max()):.4g} "
          f"mean|x| ref={float(ref.abs().mean()):.4f} sp={float(gathered.abs().mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
