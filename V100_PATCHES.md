# V100 手动补丁记录（ComfyUI + raylight）

> 本机环境：2× Tesla V100 32GB（SM70 / Volta），torch 2.9.1+cu128，NCCL 2.27.5。
> **升级 ComfyUI 或 raylight 后，以下两处修改可能被覆盖，需要手动重新应用。**
> 验证方式见文末。

---

## 补丁 1：ComfyUI 支持 `int8_rowwise` 量化格式

### 背景
`flux2-dev-int8-convrot.safetensors`（AX1Y2JP 社区版）的 `comfy_quant` 元数据声明格式为
`int8_rowwise`（每行一个 scale，`weight_scale` 形状 `[out_features,1]`，支持 convrot），
但 ComfyUI 的 `QUANT_ALGOS` 只注册了 `int8_tensorwise`，导致加载报：

```
KeyError: 'int8_rowwise'   (comfy/ops.py _load_quantized_module)
```

`int8_rowwise` 与 `int8_tensorwise` 功能完全等价（都走 `TensorWiseINT8Layout` +
`comfy_kitchen.int8_linear`，后者原生支持 per-row scale）。

### 修改文件与位置

**`comfy/quant_ops.py`** — 在 `QUANT_ALGOS["int8_tensorwise"]` 定义之后加一行：

```python
# int8_rowwise is the per-row-scale variant of int8_tensorwise; both share
# TensorWiseINT8Layout + int8_linear, whose weight_scale may be a scalar or an
# [N,1] per-row tensor.
QUANT_ALGOS["int8_rowwise"] = QUANT_ALGOS["int8_tensorwise"]
```

**`comfy/ops.py`** — `_load_quantized_module` 中：

```python
# 原：elif module.quant_format == "int8_tensorwise":
elif module.quant_format in ("int8_tensorwise", "int8_rowwise"):
```

**`comfy/ops.py`** — `_quantized_weight_state_dict`（保存/round-trip 用）：

```python
# 原：if module.quant_format == "int8_tensorwise" and getattr(params, "convrot", False):
if module.quant_format in ("int8_tensorwise", "int8_rowwise") and getattr(params, "convrot", False):
```

### 注意
- 只影响**加载**（还有保存 round-trip），不改任何计算内核。
- 官方 ComfyUI 的 fp8 模型（`float8_e4m3fn`/`float8_e5m2`）本就在 `QUANT_ALGOS` 里，无需此补丁。

---

## 补丁 2：raylight fp8 模型 + FSDP 在 V100 上 NCCL 报错

### 背景
raylight 采样时 FSDP 反分片（unshard → all-gather）把 **fp8 权重原始数据**直接送进 NCCL，
而 NCCL 2.27 的 FP8 归约只支持 sm90+（Hopper），V100（sm70）直接报：

```
torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:3690, invalid argument
Last error:
FP8 reduction support begins with sm90 capable devices.
```

**根因**：torch 2.9.1 的 FSDP2（`fully_shard`）all-gather 走
`dist.all_gather_into_tensor` → `group._allgather_base`（ProcessGroup 直连路径），
**不经过** `torch.ops._c10d_functional.all_gather_into_tensor`；而 raylight 原 fp8 补丁
只拦了 `_c10d_functional` 路径，所以没拦到，裸 fp8 进了 NCCL。

### 修复原理
`pre_all_gather` 把 fp8 qdata 以 **uint8 视角**返回（fp8 与 uint8 同为 1 字节，字节级无损）；
torch 的 `foreach_all_gather` 检测到 uint8 会自动走 uint8 all-gather（NCCL 全架构支持）；
`post_all_gather` 再把聚合数据 view 回 fp8。通信仍走 NCCL，未降级到 Gloo。

### 修改文件与位置

**`custom_nodes/raylight/src/raylight/comfy_dist/kitchen_patches/fp8.py`**
（`install_fp8_patches` 内）：

```python
def pre_all_gather(qtensor, mesh):
    qdata = qtensor._qdata
    if not qdata.is_contiguous():
        qdata = qdata.contiguous()
    scale = qtensor._params.scale
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=qdata.device)
    # NCCL has no FP8 collectives below sm90; ship the raw bytes as uint8
    return (qdata.view(torch.uint8),), (scale,)
```

```python
def post_all_gather(qtensor, all_gather_outputs, metadata, param_dtype, *, out=None):
    (data,) = all_gather_outputs
    (scale,) = metadata
    data = data.view(qtensor._qdata.dtype)   # ← 加这一行：uint8 还原回 fp8
    orig_shape = tuple(qtensor._params.orig_shape)
    ...
```

**`custom_nodes/raylight/src/raylight/comfy_dist/kitchen_patches/fp8_eager.py`**
（`fp8_eager` 变体，同样的两处，注意这里是 `@classmethod`）：

```python
# pre_all_gather 内：
return (qdata.view(torch.uint8),), (scale,)

# post_all_gather 内：
(data,) = all_gather_outputs
(scale,) = metadata
data = data.view(qtensor._qdata.dtype)
```

### 注意
- 通信后端仍是 **NCCL**，只是 all-gather 的 dtype 从 fp8 换成 uint8。
- int8 的 NCCL all-gather 全架构支持，**无需**此补丁；nvfp4 存储本就是 uint8，天然走 uint8 路径。
- raylight 从 `/tmp/raylight-ray/ray/session_*/runtime_resources/py_modules_files/_ray_pkg_*/raylight/`
  打包加载代码 → **改完必须重启整个 raylight / Ray worker 才生效**。

---

## 验证方法（V100，2 卡）

复现脚本：`tests-unit/test_fp8_fsdp_repro.py`（torchrun 2 进程 2 卡，模拟 raylight 的
fp8 QuantizedTensor + FSDP2 all-gather）：

```bash
cd /run/media/silvet/新加卷/ComfyUI
PYTHONPATH="custom_nodes/raylight/src:$PYTHONPATH" \
  .venv/bin/torchrun --nproc_per_node=2 --master_port=29580 \
  tests-unit/test_fp8_fsdp_repro.py
```

- 修复前：双 rank 报 `FP8 reduction support begins with sm90 capable devices`。
- 修复后：双 rank `FORWARD OK`，与 fp8 参考误差 ≈ 0.04%（字节级无损）。

---

## 补丁 3：raylight「原生(plain) fp8」模型 + FSDP 在 V100 上的 NCCL broadcast/all_gather

### 背景
QwenImage 等 `*_fp8_e4m3fn.safetensors`（全部权重 `float8_e4m3fn`，无 `comfy_quant`/scale 元数据，
1933 键全 fp8）属于**原生 fp8**，区别于带 `comfy_quant` 元数据的 comfy_kitchen Qtensor fp8。

raylight `fsdp_load_diffusion_model_stat_dict` 用 `assign=True` 加载 → 参数保持 fp8 存储
（RayUNETLoader 选 `fp16` 也只影响计算 dtype，不改存储）。FSDP 下两处把裸 fp8 送进 NCCL：

- `patch_fsdp` 的 `set_model_state_dict(broadcast_from_rank0=True)` → `dist.broadcast(fp8)`
- 前向 unshard → `dist.all_gather_into_tensor(fp8)`

NCCL 2.27 在 sm90 以下拒绝一切 FP8 collective → `FP8 reduction support begins with sm90 capable devices.`
（报错点在 `distributed_c10d.py ... in broadcast / group.broadcast`）。
Qtensor fp8 的 pre/post_all_gather→uint8 补丁（补丁 2）对原生 fp8 参数不生效，因为它们不是 Qtensor。

### 修复
新文件 `custom_nodes/raylight/src/raylight/comfy_dist/native_fp8.py`：
`install/restore_native_fp8_collective_patch()` 把上面两个 eager collective 的 fp8 输入在通信前
`.view(torch.uint8)`、通信完回填同一 storage（fp8/uint8 同为 1 字节 → 字节级无损）。
仅在 compute capability < sm90 时安装；挂在
`raylight/comfy_dist/kitchen_distributed.py` 的 `patch_enable_comfy_kitchen_fsdp`
FSDP 分支上，随每次 FSDP 采样安装/恢复（common_ksampler/custom_sampler/custom_sampler_advanced 均被装饰）。

要点：
- **不改模型表示、不做 fp16 全量转换、保持 fp8 存储**（20B fp8 FSDP 分片后每卡 ~10GB，而非 40GB fp16）。
- 只处理 eager 路径；torch.compile 不作用于 FSDP，无需管 `_c10d_functional`。
- sm90+ 不安装，原生 fp8 NCCL 正常；非 FSDP（ulysses 权重全副本）不涉及 fp8 权重通信，无需补丁。
- 改完必须**重启 raylight / Ray worker** 才生效（代码打包进 Ray runtime）。

### 验证
```bash
cd /run/media/silvet/新加卷/ComfyUI
PYTHONPATH="custom_nodes/raylight/src:$PYTHONPATH" \
  .venv/bin/torchrun --nproc_per_node=2 --master_port=29585 \
  tests-unit/test_fp8_native_fsdp_repro.py
```
- `USE_FP8_UINT8_COLLECTIVE_PATCH=0`：broadcast 直接报 FP8 sm90 错误（复现原 bug）。
- 默认（补丁开）：双 rank `FORWARD OK`，与 fp8 参考误差 0.0000%（broadcast + FSDP unshard 均过）。

---

## V100 性能备忘（量化在本机只有省显存的价值）

| 方案 | 每层耗时（Flux2 层 36864×6144, M=2048） | 说明 |
|---|---|---|
| 纯 fp16 | 10.3 ms | V100 原生 FP16 tensor core，最快 |
| fp8（降级→fp16） | 19.0 ms | 无 FP8 tensor core，反量化后 fp16 计算 |
| int8+convrot | 26.8 ms | 无 INT8 tensor core（DP4A 模拟）+ ConvRot 旋转开销 |

- V100 缺 FP8/INT8/BF16 硬件支持，量化模型一律走模拟/反量化路径，**计算上不可能比 fp16 快**。
- 量化的价值是**省一半显存**（1 字节/权重），让 ~35GB 的 Flux2 在 2×32GB 上能跑。
- 在"装得下"的量化方案里，**fp8 优于 int8+convrot**（约快 30%）。

---

## 升级流程（2026-09-30 起用 Git 管理）

源码已纳入 Git：`origin` = `S-i-l-v-e-t/ComfyUI-V100`（可用分支 `v100`），
`upstream` = `Comfy-Org/ComfyUI`。模型、输出、插件、`user/`、`.venv/` 全在
`.gitignore` 里，不会进仓库。

### 测试新版（在 `update` 分支上做，随时可回滚）

```bash
git switch v100
git switch -c update            # 首次；之后直接 git switch update
git fetch upstream --tags
git merge v0.XX.0               # 目标稳定 tag
```

冲突固定只有这三类：

```bash
git rm -r --ignore-unmatch .github README.md      # 上游 CI/README，本机不要
git checkout --theirs requirements.txt            # 依赖跟上游新版
git add -A && git commit
```

### 回滚

```bash
git switch v100
git reset --hard                # 工作区立刻回到当前能跑的版本
git branch -D update            # 想彻底丢掉这次测试时
```

### 升级后必做

1. **不要**直接 `pip install -r requirements.txt`：V100 用的是特制的 torch 2.9.1+cu128，
   而该文件里 `torch` 没有版本约束，会被 pip 换成默认版本而破坏环境。只装变化的包，例如
   `.venv/bin/pip install comfyui-frontend-package==X comfyui-workflow-templates==X
   comfyui-embedded-docs==X av==X comfy-kitchen==X comfy-aimdo==X`
   （注意新版已移除 `torchaudio`）。
2. 逐个确认补丁 1~3 是否还在，尤其是 `comfy/quant_ops.py`、`comfy/ops.py`、
   `comfy/ldm/modules/attention.py`；上游若重写了对应函数就得重新打。
3. 重启 ComfyUI，跑一张图确认。

---

*最后更新：2026-09-30。升级后若改动被覆盖，按上文重新应用即可。*
