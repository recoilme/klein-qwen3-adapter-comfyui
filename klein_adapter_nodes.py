"""ComfyUI nodes: FLUX.2-klein-4B driven by Qwen3-0.6B + adapter.

Replaces klein's native 4B text encoder (Qwen3-4B) with Qwen3-0.6B + a small
adapter that maps its hidden states into the DiT's 7680-dim conditioning space.
Everything else (DiT, VAE, scheduler) stays stock klein — the adapter only
changes how the text condition is computed, matching `example.py` from
https://huggingface.co/AiArtLab/qwen3-0.6b-4b-adapter exactly.

Two nodes:

  KleinQwen3AdapterLoader  -> loads Qwen3-0.6B + adapter + tokenizer (cached)
  KleinQwen3AdapterEncode  -> text -> CONDITIONING (B, L, 7680), drop_first applied

Distilled klein wants 4 steps / guidance 1.0; the base model 50 steps / CFG 4.0.
"""
import os

import torch
from transformers import AutoModel, AutoTokenizer

import comfy.model_management

# Import the vendored adapter schema/loader. Works both when this module is a
# package member (relative import) and when the package dir is on sys.path.
try:
    from . import klein_adapter_lib as adapter_lib
except ImportError:
    import klein_adapter_lib as adapter_lib

# Encoder hidden-state layers that feed the adapter (student_layers from the
# safetensors metadata; the fallback matches the released adapter).
_FALLBACK_LAYERS = [2, 9, 14, 18, 23, 27]
_FALLBACK_DROP_FIRST = 5

_CACHE = {}


def _torch_device():
    try:
        return comfy.model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_adapter(adapter: str) -> str:
    """Adapter field accepts a file path, a name in models/klein_adapter, or a HF repo id."""
    if adapter and os.path.isfile(adapter):
        return adapter

    # name in ComfyUI/models/klein_adapter/
    try:
        import folder_paths
        names = folder_paths.get_filename_list("klein_adapter")
        if adapter in names:
            return folder_paths.get_full_path("klein_adapter", adapter)
        if any(os.path.basename(n) == adapter for n in names):
            return folder_paths.get_full_path(
                "klein_adapter", next(n for n in names if os.path.basename(n) == adapter))
    except Exception:
        pass

    # HF repo id -> download adapter_v14_bal.safetensors
    if adapter and not os.path.isabs(adapter):
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(adapter, "adapter_v14_bal.safetensors")
        except Exception as e:
            raise FileNotFoundError(
                f"cannot resolve adapter {adapter!r}: not a file, not in "
                f"models/klein_adapter, and HF download failed ({e})") from e

    raise FileNotFoundError(f"cannot resolve adapter {adapter!r}")


def _load(encoder_id, adapter_ref, dtype_str):
    key = (encoder_id, adapter_ref, dtype_str)
    if key in _CACHE:
        return _CACHE[key]

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    device = _torch_device()

    enc = AutoModel.from_pretrained(encoder_id, dtype=torch_dtype).eval().to(device)
    tok = AutoTokenizer.from_pretrained(encoder_id)

    adapter_path = _resolve_adapter(adapter_ref)
    # The adapter runs in fp32 (same as example.py); input bf16 -> fp32 -> out bf16.
    adapter, meta = adapter_lib.load_adapter(adapter_path, device=device, dtype=torch.float32)
    layers = [int(x) for x in str(meta.get("student_layers",
                                           ",".join(map(str, _FALLBACK_LAYERS)))).split(",")]
    drop_first = int(meta.get("drop_first", _FALLBACK_DROP_FIRST))

    obj = {
        "enc": enc,
        "tok": tok,
        "adapter": adapter,
        "layers": layers,
        "drop_first": drop_first,
    }
    _CACHE[key] = obj
    return obj


def _get_qwen3_prompt_embeds(enc, tok, prompt, device, max_sequence_length, hidden_states_layers):
    """Verbatim port of diffusers Flux2KleinPipeline._get_qwen3_prompt_embeds."""
    dtype = enc.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt

    all_input_ids, all_attention_masks = [], []
    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt", padding="max_length",
                     truncation=True, max_length=max_sequence_length)
        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    with torch.no_grad():
        output = enc(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)

    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)

    batch_size, num_channels, seq_len, hidden_dim = out.shape
    return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)


class KleinQwen3AdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder": ("STRING", {"default": "Qwen/Qwen3-0.6B",
                                       "tooltip": "Qwen3-0.6B HF model id or local path"}),
                "adapter": ("STRING", {"default": "adapter_v14_bal.safetensors",
                                       "tooltip": "adapter safetensors: file path, "
                                                 "models/klein_adapter name, or HF repo id"}),
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"}),
            },
        }

    RETURN_TYPES = ("KLEIN_ADAPTER",)
    RETURN_NAMES = ("adapter",)
    FUNCTION = "load"
    CATEGORY = "KleinAdapter"
    OUTPUT_TOOLTIPS = ("Qwen3-0.6B + adapter wrapper for KleinQwen3AdapterEncode",)

    def load(self, encoder, adapter, dtype):
        obj = _load(encoder, adapter, dtype)
        device = _torch_device()
        obj["enc"] = obj["enc"].to(device)
        obj["adapter"] = obj["adapter"].to(device)
        return (obj,)


class KleinQwen3AdapterEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                    "tooltip": "The text to encode."}),
                "adapter": ("KLEIN_ADAPTER", {}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "KleinAdapter"
    OUTPUT_TOOLTIPS = ("A conditioning built by Qwen3-0.6B + adapter (drops the first "
                       "N attention-sink tokens).",)

    def encode(self, text, adapter):
        enc = adapter["enc"]
        tok = adapter["tok"]
        model = adapter["adapter"]
        layers = adapter["layers"]
        drop_first = adapter["drop_first"]
        device = _torch_device()

        embeds = _get_qwen3_prompt_embeds(enc, tok, text, device,
                                          max_sequence_length=256,
                                          hidden_states_layers=layers)
        # Adapter runs in fp32; output cast back to bf16 (same as example.py).
        dt = next(model.parameters()).dtype
        embeds = model(embeds.to(dt)).to(torch.bfloat16)

        # Template attention sinks (norm ~6000 vs ~170) are dropped at the front.
        if drop_first:
            embeds = embeds[:, drop_first:].contiguous()

        embeds = embeds.to(comfy.model_management.intermediate_device())
        return ([(embeds, {})],)


NODE_CLASS_MAPPINGS = {
    "KleinQwen3AdapterLoader": KleinQwen3AdapterLoader,
    "KleinQwen3AdapterEncode": KleinQwen3AdapterEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KleinQwen3AdapterLoader": "Load Qwen3-0.6B Klein Adapter",
    "KleinQwen3AdapterEncode": "Encode Qwen3-0.6B Klein Adapter",
}
