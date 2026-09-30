"""Two-process FSDP2 repro for NATIVE (plain, no comfy_quant/Qtensor) FP8
weights on V100 (sm70).

Native FP8 checkpoints (e.g. QwenImage *fp8_e4m3fn.safetensors) load as raw
``float8_e4m3fn`` nn.Parameters.  Under raylight FSDP those raw fp8 params are
sharded and all-gathered by torch FSDP2, and the full state dict is broadcast
from rank 0 with ``set_model_state_dict(..., broadcast_from_rank0=True)``.
NCCL 2.27 rejects every FP8 collective below sm90:

    FP8 reduction support begins with sm90 capable devices.

Unlike comfy_kitchen QuantizedTensor FP8 (handled by
raylight/comfy_dist/kitchen_patches/fp8.py pre/post_all_gather -> uint8), plain
FP8 params have no Qtensor hooks, so the bytes hit NCCL as FP8.

Run with (no patch -> NCCL FP8 error; USE_FP8_UINT8_COLLECTIVE_PATCH=1 -> OK):
    torchrun --nproc_per_node=2 tests-unit/test_fp8_native_fsdp_repro.py
"""
import os
import sys
import torch

import torch.distributed as dist


def make_module(device):
    torch.manual_seed(0)
    in_f, out_f = 256, 512
    # native plain-fp8 semantics: values are fp8-rounded, implicit scale 1.0
    w32 = torch.randn(out_f, in_f, device=device) * 0.02
    w8 = w32.to(torch.float8_e4m3fn)
    w16_ref = w8.float().to(torch.float16)  # dequant (scale=1) reference

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(w8, requires_grad=False)

        def forward(self, x):
            # emulate a native-fp8 linear: dequant to fp16 then matmul
            return torch.nn.functional.linear(x, self.weight.to(torch.float16))

    return M(), w16_ref


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    if os.environ.get("USE_FP8_UINT8_COLLECTIVE_PATCH", "1") == "1":
        from raylight.comfy_dist.native_fp8 import install_native_fp8_collective_patch
        install_native_fp8_collective_patch()
        print(f"[rank{local_rank}] native fp8 uint8 collective patch installed")

    from torch.distributed.fsdp import fully_shard

    m, w16_ref = make_module(device)
    fully_shard(m, reshard_after_forward=True)

    # 1) exercise the same broadcast a full state-dict FSDP init performs.
    dummy = torch.rand(64, device=device).to(torch.float8_e4m3fn)
    dist.broadcast(dummy, src=0)
    torch.cuda.synchronize()

    # 2) exercise the FSDP unshard (all_gather) path with a forward.
    x = torch.randn(4, 256, dtype=torch.float16, device=device)
    try:
        out = m(x)
        torch.cuda.synchronize()
        ref = torch.nn.functional.linear(x, w16_ref)
        rel = ((out.float() - ref.float()).abs() / (ref.float().abs().mean() + 1e-3)).mean().item()
        print(f"[rank{local_rank}] FORWARD OK shape={tuple(out.shape)} dtype={out.dtype} "
              f"mean_rel_err_vs_fp8ref={rel:.4%}")
    except Exception as e:
        print(f"[rank{local_rank}] FORWARD FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
