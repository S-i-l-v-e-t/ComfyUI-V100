"""Two-process FSDP2 repro for raylight FP8 all_gather on V100 (sm70).

Run with:
    torchrun --nproc_per_node=2 tests-unit/test_fp8_fsdp_repro.py

Each rank uses its own GPU (cuda:0 / cuda:1); wraps a tiny module whose weight
is an FP8 QuantizedTensor (with raylight's fsdp_pre/post_all_gather hooks) and
runs one forward to exercise the FSDP unshard (all_gather) path.
"""
import os
import sys
import torch

import torch.distributed as dist


def make_module(device):
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    torch.manual_seed(0)
    in_f, out_f = 128, 256
    scale = 0.05
    w16 = torch.randn(out_f, in_f, dtype=torch.float16, device=device)
    w8 = (w16 * (1 / scale)).clamp(-448, 448).round().to(torch.float8_e4m3fn)
    params = TensorCoreFP8Layout.Params(
        scale=torch.tensor([scale], dtype=torch.float32, device=device),
        orig_dtype=torch.float16,
        orig_shape=(out_f, in_f),
    )
    qt = QuantizedTensor(w8, "TensorCoreFP8Layout", params)
    w16 = (w8.float() * scale).to(torch.float16)  # fp8-quantized reference

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(qt, requires_grad=False)
            self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    return M(), w16


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    if os.environ.get("USE_RAY_PATCH", "1") == "1":
        from raylight.comfy_dist.kitchen_patches.fp8 import install_fp8_patches
        install_fp8_patches()
        print(f"[rank{local_rank}] raylight fp8 patches installed")

    from torch.distributed.fsdp import fully_shard

    m, w16 = make_module(device)
    m = m.to(device)
    fully_shard(m, reshard_after_forward=True, ignored_params={m.scale})

    x = torch.randn(4, 128, dtype=torch.float16, device=device)
    try:
        out = m(x)
        torch.cuda.synchronize()
        ref = torch.nn.functional.linear(x, w16)
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
