# Anime Party Generator

Generate anime-style images locally from a web UI or CLI. Choose a quality
tier, optionally let `llama3.2:3b` improve the prompt, and get a PNG plus a JSON
sidecar containing the seed and render settings.

## Models

| Tier | Model | Default size | Estimated time* |
|---|---|---:|---:|
| `best` | [Animagine XL 4.0 Opt](https://huggingface.co/cagliostrolab/animagine-xl-4.0) (3B) | 832x1216 | 60-120 s |
| `balanced` (default) | [DreamShaper XL v2 Turbo](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo) (3B) | 832x1216 | 15-30 s |
| `fast` | [Eimis Anime Diffusion 1.0v](https://huggingface.co/eimiss/EimisAnimeDiffusion_1.0v) (0.9B) | 768x832 | 10-25 s |

*Rough time for one image on an RTX 3070 Laptop GPU after loading. Benchmark
your own card; size, driver, and first-run downloads affect the result.

All tiers use Quanto to quantize supported UNet and text-encoder linear weights
to INT8 at runtime. Cached source weights, computation, unsupported layers, and
the VAE remain FP16 because Diffusers does not support a fully integer diffusion
pipeline. The Llama model is unchanged and runs on the CPU.

## Requirements

- NVIDIA GPU with at least 8 GB VRAM
- Current NVIDIA driver and 16 GB RAM
- Internet access on first use
- About 4-8 GB of cache per model

## Docker (recommended)

Install Docker, Docker Compose, and NVIDIA Container Toolkit. Verify GPU access:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

Start the UI:

```bash
docker compose up --build
```

Open <http://localhost:7860>. Models download automatically on first use and
remain in the `hf-cache` volume. Images are written to `outputs/`.

Run the CLI in Docker:

```bash
docker compose stop animegen
docker compose run --rm animegen generate --no-llm \
  "friends at a rooftop party at sunset" --model best
```

Do not run the UI and CLI containers together on an 8 GB GPU. The standalone
command uses `--no-llm`; start `ollama` and `ollama-init` first if you want
prompt enhancement.

Stop the stack with `Ctrl+C`, then run:

```bash
docker compose down
```

Ollama is optional. Compose starts it and pulls `llama3.2:3b`, but the image app
starts independently; if Ollama is unavailable, generation falls back to the
original prompt. Add `--no-llm` to skip it explicitly.

## Local virtual environment

Use Python 3.10-3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[ui]"
python -c "import torch; print(torch.cuda.is_available())"
```

The last command must print `True`. Start the UI or CLI:

```bash
animegen ui
animegen generate "friends at a rooftop party at sunset" --model best
```

For optional prompt enhancement, install [Ollama](https://ollama.com/) and run:

```bash
ollama pull llama3.2:3b
```

## Useful commands

```bash
animegen info
animegen generate "beach bonfire" --model fast --no-llm
animegen generate "night market" --model balanced --images 2 --seed 1234
animegen generate "friends in an arcade" --model best --size 1024x1024
```

`animegen generate --help` lists every option. The main ones are `--model`,
`--images`, `--seed`, `--size`, `--steps`, `--guidance`, and `--no-llm`.

## Configuration

The supported runtime overrides are environment variables:

```bash
export ANIMEGEN_MODEL__DEFAULT=best
export ANIMEGEN_PATHS__OUTPUTS_DIR=/path/to/outputs
export ANIMEGEN_OLLAMA__HOST=http://127.0.0.1:11434
export ANIMEGEN_OLLAMA__MODEL=llama3.2:3b
```

Docker users can copy `.env.example` to `.env` to set the default tier, host
port, and an optional external Ollama address.

## Troubleshooting

| Problem | Fix |
|---|---|
| Docker socket permission denied | Add your user to the `docker` group, then fully log out/reboot before reopening the terminal or VS Code |
| Docker cannot select the `nvidia` driver | Install NVIDIA Container Toolkit and restart Docker |
| Docker Hub DNS/proxy timeout | Fix the host or Docker DNS/HTTPS proxy |
| CUDA is unavailable | Check `nvidia-smi`, the CUDA PyTorch build, and Docker GPU access |
| CUDA out of memory | Close GPU apps, use one image, or choose `fast`; do not run UI and CLI together |
| First generation is slow | The selected model is downloading/loading; later runs reuse the cache |
| Ollama is down | Start Ollama or use `--no-llm`; image generation still works |

Diagnostics:

```bash
docker compose ps
docker compose logs animegen
```

## Tests

```bash
python -m pip install -e ".[ui,dev]"
pytest
```

## Licenses

The SDXL tiers use
[CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md),
the fast tier uses [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license),
and `llama3.2:3b` uses the [Llama 3.2 Community License and Acceptable Use Policy](https://ollama.com/library/llama3.2:3b).
Review the model terms before redistribution or service use.
