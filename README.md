# Anime Party Generator

Local, offline image generation in a semi-realistic **2.5D anime style**. You type an
idea, a small LLM turns it into a proper SDXL prompt, and DreamShaper XL v2 Turbo
renders it — all on your own machine, tuned for an **8 GB RTX 3070 Laptop**.

```
"American teenagers having fun at a party"
        |
        v   llama3.2:3b via Ollama, pinned to the CPU (num_gpu: 0)
"six american teenagers dancing in a living room, string lights, warm rim
 lighting, joyful mood, low angle, casual streetwear"
        |
        v   + style contract (owned by the app, not the LLM)
"..., semi-realistic 2.5D anime style, 3D-shaded characters, volumetric
 lighting, glossy rendering, detailed faces, cinematic composition, high detail"
        |
        v   DreamShaper XL v2 Turbo, fp16, 6 steps, guidance 2.0, 832x1216
outputs/20260716_143012_seed1234.png  +  20260716_143012_seed1234.json
```

The LLM never touches the GPU, so the 8 GB stay reserved for SDXL. If Ollama is down,
generation still works — the raw prompt is used instead.

## Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| GPU | NVIDIA, 8 GB VRAM (RTX 3070 Laptop is the reference target) |
| Disk | ~8 GB for the checkpoint and the VAE |
| Ollama | optional but recommended — the tool falls back without it |

## Install

```bash
git clone <your-fork-url> anime-party-generator
cd anime-party-generator

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 1. CUDA-enabled torch FIRST (the default PyPI wheel is CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Everything else
pip install -e .

# 3. The prompt-enhancement model (CPU-bound, ~2 GB)
ollama pull llama3.2:3b
```

Check the CUDA wheel landed:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.x.x+cu121 True
```

`False` means you installed the CPU build — reinstall with the `--index-url` above.

## Download the models

The demo must never download weights live. Fetch them once:

```bash
# VAE (fp16 fix) — pulled straight from Hugging Face
python scripts/download_models.py --skip-checkpoint

# Checkpoint — no URL is hardcoded, because Civitai links rotate and may need a token
python scripts/download_models.py --url "<direct-download-url>"
python scripts/download_models.py --url "https://civitai.com/api/download/models/XXXXX" --token "$CIVITAI_TOKEN"
```

If the download fails, the script prints manual instructions. The short version: get
the DreamShaper XL **v2 Turbo** `.safetensors` (from
[Civitai](https://civitai.com/models/112902) or the
[Hugging Face mirror](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo)) and save it as
`models/DreamShaperXL_v2_Turbo.safetensors`. The filename is configurable in
`config/settings.yaml` (`model.checkpoint_filename`).

The script is idempotent: existing files are skipped unless you pass `--force`.

Verify everything is in place:

```bash
animegen info
# checkpoint : ...\models\DreamShaperXL_v2_Turbo.safetensors (found)
# ollama     : http://localhost:11434 (llama3.2:3b) up
```

## Usage

### CLI

```bash
# Simplest form
animegen generate "American teenagers having fun at a party"

# Four images, reproducible, pipeline warmed up first
animegen generate "rooftop party at sunset" --images 4 --seed 1234 --warmup

# Skip Ollama entirely; square format, more steps
animegen generate "beach party bonfire" --no-llm --size 1024x1024 --steps 8
```

| Flag | Default | Notes |
|---|---|---|
| `--images`, `-n` | 1 | Rendered one at a time; capped at 4 |
| `--seed` | random | Image *i* uses `seed + i`; always recorded |
| `--size` | `832x1216` | Or `1024x1024`; must be multiples of 8 |
| `--steps` | 6 | Clamped to the Turbo range 4–8 |
| `--guidance` | 2.0 | Turbo models want low guidance |
| `--no-llm` | off | Bypass Ollama, render the raw prompt |
| `--warmup` | off | Throwaway render first, so the real one is fast |
| `--config`, `--verbose`, `--version` | | Global options, before the subcommand |

Every image is written to `outputs/YYYYmmdd_HHMMSS_seed{N}.png` with a JSON sidecar
holding the original prompt, the enhanced prompt, the fallback flag, the full settings
and per-stage timings — enough to reproduce any shot later.

### Web UI

```bash
animegen ui                       # http://127.0.0.1:7860
animegen ui --port 8080 --warmup  # warm up at startup
set ANIMEGEN_UI_WARMUP=1          # same thing via the environment (Windows)
```

Prompt box, image-count slider, seed field (blank = random), size dropdown and an
LLM-enhancement checkbox. The gallery captions each image with its seed, and the
"Prompt details and seeds" panel shows exactly what SDXL received. The queue runs a
single worker: one generation at a time, by design.

## Configuration

Everything lives in [`config/settings.yaml`](config/settings.yaml): model paths,
scheduler, steps, guidance, resolution, Ollama host/model/timeout, the style suffix and
the negative prompt.

Any value can be overridden by an environment variable — `ANIMEGEN_` prefix, `__` for
nesting — which is handy mid-demo:

```bash
ANIMEGEN_GENERATION__STEPS=8 animegen generate "..."
ANIMEGEN_OLLAMA__HOST=http://192.168.1.5:11434 animegen generate "..."
ANIMEGEN_CONFIG_FILE=/path/to/other.yaml animegen info
```

Precedence: CLI flags → environment → `settings.yaml` → built-in defaults.

## How the 8 GB budget is spent

| Decision | Why |
|---|---|
| `torch.float16` | fp32 SDXL does not fit, full stop |
| `madebyollin/sdxl-vae-fp16-fix` | the stock VAE overflows in fp16 → black/NaN images |
| `enable_model_cpu_offload()` | streams submodules to the GPU; peak ~4 GB instead of ~10 GB. Never `.to("cuda")` |
| `enable_vae_tiling()` | decoding 832x1216 in tiles avoids a spike at the last step |
| one image per pipeline call | a batch of four 832x1216 latents does not fit next to the UNet |
| `torch.cuda.empty_cache()` after each run | keeps a long demo session from fragmenting VRAM |
| Ollama with `num_gpu: 0` | llama3.2 stays on the CPU and never competes for VRAM |
| single-worker Gradio queue | two concurrent runs OOM the card instantly |

Roughly: ~6 s per 832x1216 image after the pipeline is loaded, ~30–60 s for the first
load from a cold cache.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Black or noise-only images | stock fp16 VAE overflow | ensure the fp16-fix VAE downloaded: `python scripts/download_models.py --skip-checkpoint`; `model.vae_repo` must stay `madebyollin/sdxl-vae-fp16-fix` |
| `CUDA out of memory` | something else is using the GPU | close browser tabs / Discord / other CUDA apps, generate 1 image, drop to `--size 832x1216`; check with `nvidia-smi` |
| First image takes 60 s | cold pipeline load and kernel compilation | run with `--warmup`, or `ANIMEGEN_UI_WARMUP=1 animegen ui` |
| `SDXL checkpoint not found` | weights missing or renamed | `animegen info` shows the expected path; re-run `scripts/download_models.py` or fix `model.checkpoint_filename` |
| Prompt comes out unchanged | Ollama unreachable → fallback | `ollama serve`, `ollama pull llama3.2:3b`; `animegen info` reports `DOWN`. Generation still works |
| First LLM call times out | model is loading into RAM | run one `ollama run llama3.2:3b "hi"` before the demo, or raise `ollama.timeout` |
| `torch.cuda.is_available()` is False | CPU-only torch wheel | reinstall with `--index-url https://download.pytorch.org/whl/cu121` |
| Downloaded checkpoint is a few KB | login/HTML page instead of the file | use `--token`, or download manually in a browser |
| Images look flat 2D | style suffix lost | it is applied by the orchestrator, not the LLM — check `style.suffix` in `settings.yaml` |

## Tests

```bash
pytest -q     # no GPU, no Ollama, no downloads required
```

`torch`, `diffusers` and the Ollama HTTP calls are mocked; the gradio test skips when
gradio is not installed.

## Project layout

```
config/settings.yaml        all tunables
scripts/download_models.py  one-time weight download
src/animegen/
  config.py                 typed settings (YAML + env overrides)
  llm/enhancer.py           OllamaEnhancer, CPU-pinned, graceful fallback
  sd/pipeline.py            SDXLGenerator, the 8 GB VRAM strategy
  core/orchestrator.py      LLM -> style -> SDXL -> disk + metadata
  cli.py                    animegen generate | ui | info
  ui/app.py                 Gradio demo, single worker
tests/                      pytest suite, all heavy deps mocked
outputs/                    generated images + JSON sidecars (git-ignored)
models/                     checkpoints (git-ignored)
```

See [DEMO.md](DEMO.md) for the live-demo runbook.
