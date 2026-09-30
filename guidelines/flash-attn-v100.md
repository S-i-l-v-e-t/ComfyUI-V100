# FlashAttention-V100 编译与 ComfyUI Attention Patch

> 目标：V100（SM70 / Volta）上用 flash-attention-v100（1Cat-vLLM 魔改版）替代慢速 attention，吃满 fp16 tensor core。
> 环境：2× Tesla V100 32GB，CUDA 12.8，torch 2.9.1+cu128。
> 源码：`~/LLMs/vLLM/1Cat-vLLM/flash-attention-v100`（1Cat 魔改，BHMD layout，针对 sm70 优化）。

---

## 一、编译前：CUDA 12.8 头文件 C23 noexcept 冲突（一次性修复，本机已应用）

### 现象
编译 sm70 CUDA 扩展报错：`/usr/local/cuda-12.8/include/crt/math_functions.h` / `.hpp` 中
`cospi / sinpi / rsqrt / rsqrtf / sincospi` 等 C23 函数声明**没有 `noexcept`**，
与 glibc 2.39 `mathcalls.h` 的 `noexcept(true)` 声明冲突，nvcc 直接报错。

### 修复（sudo，永久生效；已有备份 `math_functions.h.bak`）

```bash
# 1) .h 去掉重复的 noexcept（之前误加成双份的）
sudo perl -0777 -i -pe 's/noexcept\(true\) noexcept\(true\)/noexcept(true)/g' /usr/local/cuda-12.8/include/crt/math_functions.h

# 2) .hpp 给 8 个 C23 函数声明补 noexcept(true)
sudo perl -0777 -i -pe 's/(__func__\((?:double|float|void)\s+(?:rsqrt|rsqrtf|sinpi|sinpif|cospi|cospif|sincospi|sincospif)\s*\([^;]*?\))\)/$1 noexcept(true))/g' /usr/local/cuda-12.8/include/crt/math_functions.hpp

# 3) 验证：第一个应为 0，第二个应为 8
grep -c "noexcept(true) noexcept(true)" /usr/local/cuda-12.8/include/crt/math_functions.h
grep -c "noexcept(true)" /usr/local/cuda-12.8/include/crt/math_functions.hpp
```

> 装 ninja 进目标 venv（编译自动多核并行）：
> `uv pip install --python <目标venv>/bin/python ninja`

---

## 二、编译 flash-attention-v100

```bash
cd ~/LLMs/vLLM/1Cat-vLLM/flash-attention-v100
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASH_ATTN_V100_CUDA_ARCH_LIST="7.0"
uv pip install --python <目标venv>/bin/python --no-build-isolation .
```

> ⚠️ `.so` 是 **cp311 ABI**（绑定 Python 3.11）。目标环境是别的 Python 版本，
> 必须换 `--python` 重新编译，否则 import 直接 `ImportError` / undefined symbol。
> 成功标志：site-packages 里出现约 **29MB** 的 `flash_attn_v100_cuda.cpython-3xx-*.so`。

---

## 三、验证（GPU 实测对拍 SDPA）

```python
import torch, warnings
warnings.filterwarnings('ignore')
from flash_attn_v100.flash_attn_interface import flash_attn_func, flash_attn_bhmd_func
import torch.nn.functional as F

torch.manual_seed(0)
B, T, H, D = 2, 512, 8, 64
q = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
k = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
v = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')

# BMHD（标准接口）
out = flash_attn_func(q, k, v, causal=True)
print('BMHD finite =', torch.isfinite(out).all().item())

# BHMD（ComfyUI patch 用这个）
qb = q.transpose(1, 2).contiguous(); kb = k.transpose(1, 2).contiguous(); vb = v.transpose(1, 2).contiguous()
outb = flash_attn_bhmd_func(qb, kb, vb, causal=True)
ref  = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
print('BHMD finite =', torch.isfinite(outb).all().item(),
      'max_abs_err = %.5f' % (outb.float() - ref.float()).abs().max().item())
```

本机实测：两项均通过，误差 **0.00049**（fp16 正常精度水平）。

---

## 四、Patch ComfyUI attention（已验证于 horde 内置 ComfyUI）

文件：`comfy/ldm/modules/attention.py`

### 步骤 1 — 顶部加 import 检测（放在 `sageattention` import 之后）

```python
# FlashAttention for V100 (SM70) via flash-attention-v100 (1Cat cu128 build)
try:
    from flash_attn_v100 import flash_attn_bhmd_func as _flash_v100_bhmd_func
    _HAS_FLASH_V100 = True
except Exception:
    _HAS_FLASH_V100 = False
```

### 步骤 2 — 加 `attention_flash_v100` 函数（放在 `attention_pytorch` 之后）

```python
def attention_flash_v100(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    # Flash-V100 dense path only supports fp16 and has no arbitrary-mask support.
    if mask is not None or q.dtype != torch.float16:
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape)
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),  # -> (B, H, T, D)
            (q, k, v),
        )
    out = _flash_v100_bhmd_func(q, k, v)
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out
```

### 步骤 3 — 在 `optimized_attention` 选择链插入分支（xformers 之后、pytorch 之前）

```python
elif (
    _HAS_FLASH_V100
    and model_management.pytorch_attention_enabled()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] == 7
):
    logging.info("Using flash_attn_v100 attention (V100)")
    optimized_attention = attention_flash_v100
```

### 最终优先级
```
sage_attention > xformers > flash_attn_v100 (V100) > pytorch SDPA > split > sub_quad
```

### 关键设计点
- **BHMD layout**：ComfyUI 的 `optimized_attention` 输入是 `[B, T, H*D]`；
  flash-attention-v100 的 `flash_attn_bhmd_func` 用 `[B, H, T, D]`。
  所以进内核前 `view + transpose` 成 BHMD，出来再 `transpose + reshape` 还原成 `[B, T, H*D]`。
- **只走 fp16 + 无 mask**：V100 版 dense 内核不支持任意 mask、非 fp16，
  命中这两种情况直接 fallback 到 `attention_pytorch`（安全兜底，绝不崩）。
- **capability 判据**：仅 `capability[0] == 7`（SM70）启用，其他卡完全不受影响。
- 返回值与 `attention_basic` 同构，上层调用方（`OptimizedAttention` 等）零改动。

---

## 五、与 raylight 的关系（实现完成，但当前已禁用，见第六节实测结论）

- 本指南的 patch 作用于 **ComfyUI 标准 attention 路径**（`comfy/ldm/modules/attention.py`），
  适用于直接跑 ComfyUI，也覆盖 raylight 的非 xdit 回退路径（如 `Flux.forward_orig`）。
- **raylight 的 xdit/USP 路径**（`usp_dit_forward`）走 `RAY_ACTORS` 自定义 attention，
  不经过 `optimized_attention`，已另做集成（见第六节）。
- 本机 ComfyUI venv 是 **Python 3.12**（`.venv`，torch 2.9.1+cu128）；1cat-vllm-sm70
  里的 cp312 构建是按 torch 2.10 编译的，**ABI 不兼容**（`c10_cuda_check_implementation`
  符号缺失），必须按第二节命令为 venv 重编。

---

## 六、raylight 集成（V100 flash attention，2026-08-25 已实现）

### 原理
- raylight xdit 路径的 attention 一律走 `xFuserLongContextAttention`（Ulysses all-to-all
  → ring attention → all-to-all 还原）。即使 ring_degree=1（如用户 2 卡 ulysses=2），
  本地 attention 也是 ring 循环里的**单次 dense attention**（全序列 × 分片头数）。
- 因此把 `xFuserLongContextAttention.ring_attn_fn`（实例属性，默认
  `xdit_ring_flash_attn_func`）换成一个包装：**dense + fp16 + ring 世界大小 ≤ 1 +
  无特殊参数**时直接用 `flash_attn_v100.flash_attn_bhmd_func`（输入输出 `[B,H,T,D]`，
  需 `transpose(1,2)`），其余情况（ring>1、kv cache、joint、causal、mask 类参数、
  非 fp16）原样回退原 ring 实现。

### 改动文件
- `custom_nodes/raylight/src/raylight/distributed_modules/attention.py`：
  顶部加 `_HAS_FLASH_V100` 检测
  （`from flash_attn_v100.flash_attn_interface import flash_attn_bhmd_func`）；
  新增 `_flash_v100_ring_attn_factory(orig_ring_attn)`；
  `make_xfuser_attention` 里
  `if _HAS_FLASH_V100: xfuser_attn.ring_attn_fn = _flash_v100_ring_attn_factory(xfuser_attn.ring_attn_fn)`。
- FLUX 的 `attention()`（`diffusion_models/flux/xdit_context_parallel.py`）不用改，
  它 `skip_reshape=True` 传 `[B,H,T,D]`，包装内部转成 BHMD 进内核再转回。

### 生效与验证
- **改完必须重启 raylight / Ray worker** 才生效（代码打包进 Ray runtime）。
- 单进程验证：`tests-unit/test_raylight_flash_v100_wrapper.py`（dense fp16 走 flash、
  causal/fp32 回退原实现，V100 对拍 SDPA 误差 ~3e-5）。
- 通用 smoke：`tests-unit/test_flash_v100_import.py`（venv 内 import + GPU 对拍）。

### ⚠️ 实测结论（2026-08-25，重要）：dense prefill 下是负优化，已禁用

V100 实测（venv torch 2.9.1 自编译 .so 与 1cat torch 2.10 官方 .so **性能完全一致**，
非编译问题）：flash_attn_v100 在 dense prefill（fp16，head_dim 64/128）上
**比 torch mem-efficient（TORCH_EFFICIENT）慢 ~2 倍**：

| T | D | TORCH_EFFICIENT | flash_v100 | 比值 |
|---|---|---|---|---|
| 4096 | 128 | 5.7ms | 11.9ms | 0.47x |
| 8192 | 128 | 23.1ms | 48.2ms | 0.48x |
| 16384 | 128 | 99.3ms | 194.5ms | 0.51x |
| 32768 | 128 | 492.9ms | 773.0ms | 0.64x |
| 65536 | 128 | 2701.2ms | 3076.9ms | 0.88x |

- 显存峰值与 mem-efficient **完全相同**（同为 tiled/fused，0.66/1.32/2.63GB），
  没有“省显存 / 能跑更长”的补偿优势。
- 原因：1Cat 魔改版是为 **vLLM decode/paged** 场景优化的，dense prefill 不是强项，
  Volta 上还不如 PyTorch 内置 mem-efficient kernel。扩散采样全是 prefill，用不上。
- **因此当前已禁用两处启用点**（代码与函数保留备用）：
  - `comfy/ldm/modules/attention.py` 选择链不再引用 `attention_flash_v100`。
  - `raylight/distributed_modules/attention.py` 的 `make_xfuser_attention`
    不再覆盖 `ring_attn_fn`（注释里给了重新启用的一行）。
- raylight 里 `XFuser_attention` 保持 **`TORCH_EFFICIENT`**（V100 上 `TORCH_FLASH`
  会直接崩：flash backend 仅 Ampere+；`TORCH_MATH` 在长序列 OOM）。
- 复测方法：`tests-unit/bench_flash_v100_detail.py`（多 shape 扫秒）与
  `tests-unit/bench_flash_v100_cross_env.py`（venv vs 1cat 交叉验证）。
- 若未来拿到更适合 sm70 prefill 的 flash 内核，改回启用点前务必先用上述脚本复测
  ≥1x 再上。

### 第三方节点 `FearL0rd/ComfyUI-Flash-Attention_v100` 实测（2026-08-25，已拉下验证）

clone 到 /tmp 实测，结论：**在当前 ComfyUI 上直接崩溃，即使修好也是负优化，别装。**

- **Case 1（当前 ComfyUI 真实调用）**：`TypeError: v100_attention() got an
  unexpected keyword argument 'transformer_options'`。当前 ComfyUI 的
  `CrossAttention.forward` 调 `optimized_attention(..., transformer_options=...)`，
  该节点的 `v100_attention(q,k,v,heads,mask,attn_precision)` 签名不接受此 kwarg，
  TypeError 在函数体外抛出，连它的 fallback 都救不了 → 每次 attention 直接崩。
- 它假设的输入布局 `(batch*heads, seq, dim)` 与当前 ComfyUI 的 `(batch, seq, heads*dim)`
  也不匹配（`batch_size = q.shape[0] // heads` 会算错）。写于 6 个月前，基于旧版 ComfyUI。
- **Case 2（在其假设的旧布局下，单独调它内部路径）**：计算正确（对拍 err 6e-5），
  但性能 **12.95ms vs 原始 mem-efficient 5.50ms = 慢 2.35 倍**（T=4096, H=24, D=128）。
  它内部就是调同一个 `flash_attn_v100` 内核，没有任何黑科技。
- 它还把假的 `flash_attn` 模块塞进 `sys.modules`（兼容性 hack），非必要。
- 测试脚本：`tests-unit/test_flash_attn_v100_node_e2e.py`。
