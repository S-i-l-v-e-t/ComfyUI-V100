"""
DualCLIPLoaderSplit50 - DualCLIPLoader that splits the T5 encoder 50/50 across two GPUs.

Rationale (raylight setup):
  * UNet/DiT and LoRA run inside Ray workers, so the ComfyUI main process only
    manages the text encoders (CLIP-L + T5xxl) and the VAE.  Nothing in the main
    process ever triggers an unpatch/reload of the CLIP, so the T5 stays resident
    on both GPUs once placed.
  * With comfy-aimdo enabled, text encoders are initially allocated on CPU
    (`text_encoder_initial_device` returns the offload device), so we can fan the
    T5 layers straight from CPU into the two GPUs without a 9.3GB peak on one card.

The split is reapplied on every model load via a hook, so even an unexpected
reload (VRAM pressure, manual refresh) restores the two-card layout instead of
collapsing back onto device0.
"""

import logging

import torch
import folder_paths
import comfy.sd
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders import t5 as t5mod
from comfy.text_encoders import llama as llamamod
from comfy.patcher_extension import CallbacksMP

logger = logging.getLogger("t5_split50")

_PATCH_MARKER = "_t5_split_forward_patched"
_SPLIT_ATTR = "_t5_split_cfg"


# ---------------------------------------------------------------------------
# Cross-device T5 forward: move hidden states to each block's device, and move
# the encoder output back to the primary device before returning.
# ---------------------------------------------------------------------------
def _patch_t5_stack_forward(device0="cuda:0", device1="cuda:1"):
    if getattr(t5mod.T5Stack, _PATCH_MARKER, False):
        return
    _orig = t5mod.T5Stack.forward

    def split_forward(self, x, attention_mask=None, intermediate_output=None,
                      final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        intermediate = None
        past_bias = None
        if intermediate_output is not None and intermediate_output < 0:
            intermediate_output = len(self.block) + intermediate_output

        for i, l in enumerate(self.block):
            ldev = next(l.parameters()).device
            if x.device != ldev:
                x = x.to(ldev)
            # mask and T5's relative-attention bias must ride along with x to the
            # layer's device, or attention mixes cuda:0 bias with cuda:1 q/k/v.
            if mask is not None and mask.device != ldev:
                mask = mask.to(ldev)
            if past_bias is not None and past_bias.device != ldev:
                past_bias = past_bias.to(ldev)
            opt = optimized_attention_for_device(ldev, mask=attention_mask is not None, small_input=True)
            x, past_bias = l(x, mask, past_bias, opt)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)

        x = x.to(device0)
        if intermediate is not None:
            intermediate = intermediate.to(device0)
        return x, intermediate

    t5mod.T5Stack.forward = split_forward
    setattr(t5mod.T5Stack, _PATCH_MARKER, True)


# ---------------------------------------------------------------------------
# Split/re-split the T5 encoder across device0/device1. Idempotent.
# ---------------------------------------------------------------------------
def _apply_split(patcher, device0, device1, split_blocks):
    model = patcher.model
    te = model
    t5 = getattr(te, "t5xxl", None)
    if t5 is None or not hasattr(t5, "transformer"):
        logger.warning("DualCLIPLoaderSplit50: unsupported clip type, no t5xxl.transformer found; skipping split")
        return False
    enc = t5.transformer.encoder
    if not hasattr(enc, "block"):
        logger.warning("DualCLIPLoaderSplit50: no T5 encoder block found; skipping split")
        return False

    blocks = enc.block
    n = len(blocks)
    sb = min(max(1, split_blocks), n - 1)

    for b in blocks[:sb]:
        b.to(device0)
    for b in blocks[sb:]:
        b.to(device1)
    enc.final_layer_norm.to(device1)
    if hasattr(t5, "shared"):
        t5.shared.to(device0)
    if hasattr(te, "clip_l"):
        te.clip_l.to(device0)

    # Bookkeeping: tell Comfy this model is resident so it is not reloaded.
    cuda0_bytes = 0
    for b in blocks[:sb]:
        cuda0_bytes += sum(p.numel() * p.element_size() for p in b.parameters())
    if hasattr(t5, "shared"):
        cuda0_bytes += sum(p.numel() * p.element_size() for p in t5.shared.parameters())
    if hasattr(te, "clip_l"):
        cuda0_bytes += sum(p.numel() * p.element_size() for p in te.clip_l.parameters())
    model.model_loaded_weight_memory = cuda0_bytes
    model.device = torch.device(device0)
    logger.info("DualCLIPLoaderSplit50: T5 split -> %d layers on %s, %d layers on %s (cuda0 %.2fGB)",
                sb, device0, n - sb, device1, cuda0_bytes / (1024**3))
    return True


def _split_reload_hook(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
    cfg = getattr(patcher.model, _SPLIT_ATTR, None)
    if cfg is None:
        return
    if cfg.get("kind") == "llm":
        _apply_llm_split(patcher, cfg["device0"], cfg["device1"], cfg["split_blocks"])
    else:
        _apply_split(patcher, cfg["device0"], cfg["device1"], cfg["split_blocks"])


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class DualCLIPLoaderSplit50:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
            "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
            "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image", "hunyuan_video_15", "kandinsky5", "kandinsky5_image", "ltxv", "newbie", "ace"],),
        },
            "optional": {
                "device0": (["cuda:0", "cuda:1"], {"default": "cuda:0"}),
                "device1": (["cuda:0", "cuda:1"], {"default": "cuda:1"}),
                "split_blocks": ("INT", {"default": 12, "min": 1, "max": 23}),
            }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "multigpu"

    def load_clip(self, clip_name1, clip_name2, type, device0="cuda:0", device1="cuda:1", split_blocks=12):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        # disable_dynamic=True -> base ModelPatcher (not ModelPatcherDynamic), so our
        # weight placement is not fought by the DynamicVRAM loader.
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2],
                                  embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=clip_type,
                                  model_options={},
                                  disable_dynamic=True)

        patcher = clip.patcher

        # Remember split config for the reload hook.
        cfg = {"kind": "t5", "device0": device0, "device1": device1, "split_blocks": split_blocks}
        setattr(patcher.model, _SPLIT_ATTR, cfg)

        _patch_t5_stack_forward(device0, device1)

        # Hook reloads so an unexpected reload re-applies the split instead of
        # collapsing onto device0.  (Base ModelPatcher.load fires ON_LOAD.)
        try:
            patcher.add_callback_with_key(CallbacksMP.ON_LOAD, "t5_split50", _split_reload_hook)
        except Exception as e:
            logger.warning("DualCLIPLoaderSplit50: could not register reload hook: %s", e)

        # Place layers now (T5 is on CPU at this point under aimdo).
        # We deliberately do NOT register the model in current_loaded_models:
        # Comfy's own load_models_gpu -> model_load path sets up LoadedModel
        # correctly, and because model_loaded_weight_memory is > 0,
        # partially_load returns early without reloading, so the split holds.
        if not _apply_split(patcher, device0, device1, split_blocks):
            return (clip,)

        return (clip,)


# ---------------------------------------------------------------------------
# Single-CLIP split: decoder-only LLM text encoders (OmniGen2 / Qwen2.5-VL-3B
# and friends).  The LLM lives at clip.cond_stage_model.<sub>.transformer.model
# (a Llama2_ instance with .layers).  We fan the decoder layers across the two
# GPUs and patch Llama2_.forward so hidden states (plus freqs_cis / mask) are
# moved to each layer's device, and the output is moved back to the input
# device.  The patch is idempotent: on an unsplit model every layer is on the
# same device and nothing moves.
# ---------------------------------------------------------------------------
def _tensors_to(value, device):
    # freqs_cis is a tuple (cos, sin, nsin) - or a list of them for multi-theta
    # configs - so move every tensor inside it instead of calling .to() on the
    # container itself.
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(_tensors_to(v, device) for v in value)
    return value


def _patch_llama_forward():
    if getattr(llamamod.Llama2_, _PATCH_MARKER, False):
        return

    def split_forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                      final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[],
                      past_key_values=None, input_ids=None, deepstack_embeds=None, visual_pos_masks=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        out_dev = x.device
        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = self.get_past_len(past_key_values)
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)
        freqs_cis = self.compute_freqs_cis(position_ids, x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)
        if seq_len > 1:
            causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask

        intermediate = None
        all_intermediate = None
        only_layers = None
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        next_key_values = []
        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone().to(out_dev))

            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[i] if len(past_key_values) > 0 else []

            ldev = next(layer.parameters()).device
            if x.device != ldev:
                x = x.to(ldev)
            # Keep every aux tensor on the layer's device each iteration, so the
            # attention kernel never sees a mask/rope on a different GPU than q/k/v.
            if freqs_cis is not None:
                freqs_cis = _tensors_to(freqs_cis, ldev)
            if mask is not None and mask.device != ldev:
                mask = mask.to(ldev)
            opt = optimized_attention_for_device(ldev, mask=mask is not None, small_input=True)

            x, current_kv = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis,
                                  optimized_attention=opt, past_key_value=past_kv)
            if current_kv is not None:
                next_key_values.append(current_kv)

            if deepstack_embeds is not None and i < len(deepstack_embeds):
                vpm = visual_pos_masks
                if isinstance(vpm, torch.Tensor):
                    vpm = vpm.to(x.device)
                x[vpm] = x[vpm] + deepstack_embeds[i].to(x)
            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)
        x = x.to(out_dev)

        if all_intermediate is not None:
            if only_layers is None or ((i + 1) in only_layers):
                all_intermediate.append(x.unsqueeze(1).clone())
        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)
        if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
            # final_layer_norm sits on the second GPU (with the back half of the
            # layers); intermediate may have been collected back on out_dev, so
            # move it to the norm's device before applying.
            intermediate = self.norm(intermediate.to(next(self.norm.parameters()).device))
        if intermediate is not None:
            intermediate = intermediate.to(out_dev)

        if len(next_key_values) > 0:
            return x, intermediate, [kv.to(out_dev) for kv in next_key_values]
        else:
            return x, intermediate

    llamamod.Llama2_.forward = split_forward
    setattr(llamamod.Llama2_, _PATCH_MARKER, True)


def _find_llms(model):
    return [m for _n, m in model.named_modules() if isinstance(m, llamamod.Llama2_)]


def _apply_llm_split(patcher, device0, device1, split_blocks):
    model = patcher.model
    llms = _find_llms(model)
    if not llms:
        logger.warning("CLIPLoaderSplit: no Llama2_ LLM found in clip; skipping split")
        return False

    cuda0_bytes = 0
    for llm in llms:
        layers = llm.layers
        n = len(layers)
        sb = min(max(1, split_blocks), n - 1)
        for l in layers[:sb]:
            l.to(device0)
        for l in layers[sb:]:
            l.to(device1)
        if getattr(llm, "norm", None) is not None:
            llm.norm.to(device1)
        for attr in ("embed_tokens", "lm_head"):
            sub = getattr(llm, attr, None)
            if sub is not None:
                sub.to(device0)
        for l in layers[:sb]:
            cuda0_bytes += sum(p.numel() * p.element_size() for p in l.parameters())
        for attr in ("embed_tokens", "lm_head"):
            sub = getattr(llm, attr, None)
            if sub is not None:
                cuda0_bytes += sum(p.numel() * p.element_size() for p in sub.parameters())
        logger.info("CLIPLoaderSplit: %s -> %d/%d layers on %s/%s (cuda0 %.2fGB)",
                    type(llm).__name__, sb, n - sb, device0, device1, cuda0_bytes / (1024**3))

    model.model_loaded_weight_memory = cuda0_bytes
    model.device = torch.device(device0)
    return True


class CLIPLoaderSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("text_encoders"),),
            "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu", "krea2", "joyimage", "mage", "minimax"],),
        },
            "optional": {
                "device0": (["cuda:0", "cuda:1"], {"default": "cuda:0"}),
                "device1": (["cuda:0", "cuda:1"], {"default": "cuda:1"}),
                "split_blocks": ("INT", {"default": 0, "min": 0, "max": 99}),
            }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "multigpu"

    def load_clip(self, clip_name, type="stable_diffusion", device0="cuda:0", device1="cuda:1", split_blocks=0):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                                  embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=clip_type,
                                  model_options={},
                                  disable_dynamic=True)

        patcher = clip.patcher

        llms = _find_llms(patcher.model)
        if not llms:
            logger.warning("CLIPLoaderSplit: clip has no Llama2_ LLM to split (%s); returning unchanged", type)
            return (clip,)
        if split_blocks == 0:
            split_blocks = max(1, len(llms[0].layers) // 2)

        cfg = {"kind": "llm", "device0": device0, "device1": device1, "split_blocks": split_blocks}
        setattr(patcher.model, _SPLIT_ATTR, cfg)

        _patch_llama_forward()

        try:
            patcher.add_callback_with_key(CallbacksMP.ON_LOAD, "t5_split50_llm", _split_reload_hook)
        except Exception as e:
            logger.warning("CLIPLoaderSplit: could not register reload hook: %s", e)

        if not _apply_llm_split(patcher, device0, device1, split_blocks):
            return (clip,)

        # Same as DualCLIPLoaderSplit50: let Comfy manage LoadedModel through
        # load_models_gpu; model_loaded_weight_memory > 0 keeps the split intact.
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "DualCLIPLoaderSplit50": DualCLIPLoaderSplit50,
    "CLIPLoaderSplit": CLIPLoaderSplit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DualCLIPLoaderSplit50": "DualCLIP Loader Split 50/50",
    "CLIPLoaderSplit": "CLIP Loader Split 50/50",
}
