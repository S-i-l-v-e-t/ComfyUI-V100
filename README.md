# ComfyUI on V100

V100 (SM70 / Volta) has no bf16 hardware and no fp8 or int8 compute units, so upstream ComfyUI
either errors out on this card or falls back to a very slow path. This repository is a copy of
upstream v0.45.4 with a set of patches that make it work on V100. Nodes, interfaces and workflow
format are the same as upstream, so the ComfyUI documentation applies as usual.

Tested on 2× Tesla V100 32GB, torch 2.9.1+cu128, CUDA 12.8, NCCL 2.27.5. A single card runs
normal workflows; two cards are used for distributed sampling through raylight.

bf16 on this card comes from a kernel written for this fork
([softbf16](https://github.com/S-i-l-v-e-t/Soft-BF16-V100)) that maps bf16 matmuls onto fp16
tensor cores, so bf16 models run as they are and do not need converting to fp16 first.
`Using softbf16 ops` in the log means that path is active.

Distributed sampling needs the matching raylight fork,
[raylight-V100](https://github.com/S-i-l-v-e-t/raylight-V100).

## Branches

Only use the files on the `v100` branch, to avoid running into unknown issues.

## Installing

Install softbf16 first ([Soft-BF16-V100](https://github.com/S-i-l-v-e-t/Soft-BF16-V100)):

```bash
uv pip install --python .venv/bin/python git+https://github.com/S-i-l-v-e-t/Soft-BF16-V100
```

It provides the `bf16_online` module, which `comfy/ops.py` imports at the top of the file.
Without it ComfyUI fails to start with `ModuleNotFoundError: No module named 'bf16_online'`.
It is not listed in `requirements.txt`, so it has to be installed separately.

Do not install `requirements.txt` wholesale. That file does not pin torch, and installing it
replaces the torch 2.9.1+cu128 build this card needs. Install only these packages instead (the
environment is managed by uv and has no pip):

```bash
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple \
  comfyui-frontend-package==1.58.5 comfyui-workflow-templates==0.11.109 \
  comfyui-embedded-docs==0.5.16 comfy-kitchen==0.2.41 comfy-aimdo==0.6.0
```

If uv reports `not found in the package registry` for a package, the mirror is usually behind.
Add `--index-url https://pypi.org/simple` and try again.

Keep torch at 2.9.1+cu128. Changing it means rebuilding xformers.

xformers has to be built from source as well: official wheels have not shipped V100 kernels since
0.0.2x. Use `v0.0.33` (built against torch 2.9.0 and compiles fine against 2.9.1; 0.0.34 and newer
require torch 2.10):

```bash
export TORCH_CUDA_ARCH_LIST="7.0" CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH MAX_JOBS=12
git clone --recursive --branch v0.0.33 https://github.com/facebookresearch/xformers.git
cd xformers && python setup.py bdist_wheel
uv pip install --python <ComfyUI directory>/.venv/bin/python --no-deps dist/xformers-*.whl
```

`--no-deps` is required, otherwise installing the wheel replaces torch with the version it
declares. Build it in a separate venv with a matching torch, then install the wheel into the main
environment.

## Running

```bash
python main.py --listen 127.0.0.1 --enable-manager
```

Add `--use-xformers` to use xformers; without it ComfyUI picks another attention implementation.
Add `--fp32-unet` when you need to compare against full fp32 output. Do not enable
`--supports-fp8-compute`: this card has no fp8 compute units and the flag sends ComfyUI down the
wrong path.

## Known limitations

Startup prints a cu130 warning. V100 is not supported on CUDA 13, so ComfyUI falls back to a
compatible path. This is expected and can be ignored.

## Self-check

`tests-unit/` has two two-GPU self-check scripts. Run them after changing the environment to
confirm the multi-GPU fp8 path still works:

```bash
PYTHONPATH=custom_nodes/raylight/src:$PYTHONPATH \
  .venv/bin/torchrun --nproc_per_node=2 tests-unit/test_fp8_fsdp_repro.py
PYTHONPATH=custom_nodes/raylight/src:$PYTHONPATH \
  .venv/bin/torchrun --nproc_per_node=2 tests-unit/test_fp8_native_fsdp_repro.py
```

Both printing `FORWARD OK` means the path is fine.

## References

- Documentation: https://docs.comfy.org
- Startup flags: https://docs.comfy.org/development/comfyui-server/startup-flags
- Troubleshooting: https://docs.comfy.org/troubleshooting/overview
- Upstream repository: https://github.com/Comfy-Org/ComfyUI

## License

Based on [ComfyUI](https://github.com/Comfy-Org/ComfyUI); see `LICENSE` (GPL-3.0). The companion
[softbf16](https://github.com/S-i-l-v-e-t/Soft-BF16-V100) is released separately under AGPL-3.0.
