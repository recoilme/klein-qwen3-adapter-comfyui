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

## Quickstart (from scratch)

Copy-paste path from an empty machine to a first image (verified end to end on
ComfyUI 0.36 / RTX 5090):

```bash
# 1. ComfyUI itself — skip if you already have one
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv .venv && . .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt

# 2. this custom node
cd custom_nodes
git clone https://github.com/recoilme/klein-qwen3-adapter-comfyui
cd ..

# 3. model files (paths are relative to ComfyUI/, 8.4 GiB of downloads)
mkdir -p models/klein_adapter
curl -L -o models/diffusion_models/flux-2-klein-4b.safetensors \
  https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors
curl -L -o models/vae/flux2-vae.safetensors \
  https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors
curl -L -o models/klein_adapter/adapter_v14_bal.safetensors \
  https://huggingface.co/AiArtLab/qwen3-0.6b-4b-adapter/resolve/main/adapter_v14_bal.safetensors
# Qwen3-0.6B (1.4 GiB) needs no manual step: the node pulls it from the Hub on the
# first run. Offline machine: fetch it beforehand with `hf download Qwen/Qwen3-0.6B`
# (or point the node's `encoder` field at a local directory).

# 4. start ComfyUI
python main.py                                     # then open the URL it prints (:8188)

# 5. in the UI: Workflow -> Open ->
#    custom_nodes/klein-qwen3-adapter-comfyui/workflows/flux2_klein_qwen3_06b_adapter.json
#    (dragging the file onto the canvas works too). Type into "Positive prompt", press Run.
```

| File | Size | Destination |
|---|---|---|
| `flux-2-klein-4b.safetensors` | 7.2 GiB | `models/diffusion_models/` |
| `flux2-vae.safetensors` | 321 MiB | `models/vae/` |
| `adapter_v14_bal.safetensors` | 840 MiB | `models/klein_adapter/` |
| Qwen3-0.6B (auto-download) | 1.4 GiB | Hugging Face cache (`$HF_HOME`) |

Text-to-image at 768×1280 with the distilled model takes ~2 s per image on an
RTX 5090 and peaks at ~12.6 GiB of VRAM (most of it is the klein DiT itself). The
encoder side is 2.2 GiB (Qwen3-0.6B bf16 + adapter in fp32) instead of the 7.5 GiB
of klein's native Qwen3-4B — that is where the saving comes from.

The two nodes live under the `KleinAdapter` category; `adapter` accepts a name from
`models/klein_adapter/`, an absolute path, or an HF repo id (`AiArtLab/qwen3-0.6b-4b-adapter`).

## Install

Clone into ComfyUI's `custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/recoilme/klein-qwen3-adapter-comfyui
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
