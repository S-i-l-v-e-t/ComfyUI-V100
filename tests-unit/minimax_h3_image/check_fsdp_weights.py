"""Do the FSDP-gathered quantized weights match the checkpoint?

Every raylight run goes through this, so a bad gather would degrade all of them
equally. Loads the real checkpoint under FSDP on 2 ranks and compares a few
layers against the file.

Run: PYTHONPATH=custom_nodes/raylight/src:. .venv/bin/torchrun --nproc_per_node=2 \
        tests-unit/check_fsdp_weights.py
"""
import os

import torch
from safetensors import safe_open

CKPT = "models/diffusion_models/minimax_h3_image_pruned_fp8_scaled.safetensors"
LAYERS = ("blocks.0.mlp.fc1", "blocks.0.attn.qkv_proj", "blocks.25.mlp.fc2", "final_layer.video_out")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    import torch.distributed as dist
    import raylight.distributed_modules.attention as xfuser_attn
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

    xfuser_attn.set_attn_type("TORCH_EFFICIENT")
    xfuser_attn.set_sync_ulysses(False)
    init_distributed_environment(rank=local_rank, world_size=world)
    initialize_model_parallel(data_parallel_degree=1, sequence_parallel_degree=world,
                              classifier_free_guidance_degree=1, ring_degree=1,
                              ulysses_degree=world, pipeline_parallel_degree=1)

    from torch.distributed.device_mesh import init_device_mesh
    from raylight.comfy_dist.sd import fsdp_load_diffusion_model

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
    model, _ = fsdp_load_diffusion_model(CKPT, local_rank, mesh, False, model_options={})
    if local_rank == 0:
        print("loaded:", type(model.model).__name__, "| fsdp wrapped:", hasattr(model.model, "blocks"), flush=True)

    target = model.model
    for name in LAYERS:
        module = target
        for part in name.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
        if module is None:
            if local_rank == 0:
                print(f"{name}: 模型里找不到")
            continue
        weight = getattr(module, "weight", module)
        if hasattr(weight, "dequantize"):
            weight = weight.dequantize()
        weight = weight.detach().float().cpu()
        with safe_open(CKPT, "pt") as f:
            q = f.get_tensor(name + ".weight").float()
            scale = float(f.get_tensor(name + ".weight_scale").float())
        ref = q * scale
        if tuple(ref.shape) != tuple(weight.shape):
            if local_rank == 0:
                print(f"{name}: 形状不符 ref{tuple(ref.shape)} vs got{tuple(weight.shape)}")
            continue
        rel = float((weight - ref).norm() / ref.norm())
        extra = ""
        if local_rank == 0:
            print(f"{name:26} ref{tuple(ref.shape)} 相对误差 {rel:.2e} "
                  f"std ref={float(ref.std()):.5f} got={float(weight.std()):.5f}{extra}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
