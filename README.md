# klein-qwen3-adapter for ComfyUI

ComfyUI nodes that drive **FLUX.2-klein-4B** with **Qwen3-0.6B + a small adapter**
instead of its native Qwen3-4B text encoder. The DiT, VAE and scheduler stay
stock klein — only the text conditioning is computed differently, exactly like
`example.py` from [AiArtLab/qwen3-0.6b-4b-adapter](https://huggingface.co/AiArtLab/qwen3-0.6b-4b-adapter).

- ~7.5 GB less VRAM than the native Qwen3-4B encoder, ~0.3 s/step faster.
- Adapter: point-wise MLP `6144 -> 8192 -> 8192 -> 7680` (180M) + a 2-block
  residual attention branch (39.6M), fp32.
- Works with the distilled model (4 steps, guidance 1.0) and the base model
  (50 steps, guidance 4.0 — CFG works there).

## Install

Clone into ComfyUI's `custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/klein-qwen3-adapter-comfyui
```

Dependencies (`transformers`, `safetensors`) are already part of a standard
ComfyUI install; if the node reports them missing run:

```bash
pip install -r custom_nodes/klein-qwen3-adapter-comfyui/requirements.txt
```

## Models

| File | Where | Notes |
|---|---|---|
| `flux-2-klein-4b.safetensors` (distilled) or `flux-2-klein-base-4b.safetensors` | `ComfyUI/models/diffusion_models/` | the klein DiT (stock) |
| `flux2-vae.safetensors` | `ComfyUI/models/vae/` | the stock Flux2 VAE (untouched by the adapter) |
| `adapter_v14_bal.safetensors` | `ComfyUI/models/klein_adapter/` | this adapter |
| Qwen3-0.6B | HF cache (`Qwen/Qwen3-0.6B`) | downloaded automatically on first use |

The node registers a `models/klein_adapter/` folder automatically. The
`adapter` field also accepts an absolute path or an HF repo id
(`AiArtLab/qwen3-0.6b-4b-adapter`).

## Usage

Two nodes (category `KleinAdapter`):

1. **Load Qwen3-0.6B Klein Adapter** — `encoder`, `adapter`, `dtype`.
2. **Encode Qwen3-0.6B Klein Adapter** — `text` + the adapter → `CONDITIONING`.

Plug the `CONDITIONING` into the rest of a standard Flux2 Klein workflow:
`UNETLoader` → klein DiT, `VAELoader` → Flux2 VAE, `Flux2Scheduler` (4 steps for
distilled), `KSamplerSelect` (euler), `CFGGuider` (cfg 1.0), `SamplerCustomAdvanced`.

A ready workflow is in [`workflows/flux2_klein_qwen3_06b_adapter.json`](workflows/flux2_klein_qwen3_06b_adapter.json):
drag it into the ComfyUI canvas or load it via the menu.

## Settings

| Model | Steps | Guidance | Sampler |
|---|---|---|---|
| `flux-2-klein-4b.safetensors` (distilled) | 4 | 1.0 | euler |
| `flux-2-klein-base-4b.safetensors` (base) | 50 | 4.0 | euler |

## How it matches the reference

The node ports `Flux2KleinPipeline._get_qwen3_prompt_embeds` verbatim and runs
the adapter in fp32, dropping the first `drop_first` (5) template attention-sink
tokens — so the conditioning tensor is byte-identical to `example.py`. The final
image differs from `diffusers` only by framework sampling (noise RNG, scheduler),
which is the same difference you'd get with the *native* klein encoder.

## License

Apache-2.0 (the adapter itself is Apache-2.0).
