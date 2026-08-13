# Anime Party Generator

Generate anime-style images locally from the web UI or CLI. Choose one of three
image-quality tiers; Ollama can optionally expand short prompts. Every PNG is
saved in `outputs/` with a JSON sidecar containing its prompt, seed, model, and
settings.

## Image models

| Choice | Image model | Size | Estimated time* | Use it for |
|---|---|---:|---:|---|
| `best` | [Animagine XL 4.0 Opt](https://huggingface.co/cagliostrolab/animagine-xl-4.0) (3B) | 832x1216 | 60-120 s | Highest-quality anime output |
| `balanced` (default) | [DreamShaper XL v2 Turbo](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo) (3B) | 832x1216 | 15-30 s | Good quality with much shorter waits |
| `fast` | [Dreamlike Anime 1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) (0.9B) | 768x768 | 10-25 s | Drafts and prompt experiments |

*Estimates are for one image on an RTX 3070 Laptop GPU after the model is
loaded. Hardware, drivers, dimensions, and first-run loading change the actual
time. The first use of each choice also downloads its source files.

Animagine XL 4.0 Opt is the largest high-quality anime specialist that fits the
project's 8 GB GPU and 16 GB system-memory target. Much larger 12B-17B models
remain impractical on that machine even with INT8 weights.

All three choices quantize supported denoiser and text-encoder linear weights
to **INT8 at runtime** with Quanto. Diffusion arithmetic, unsupported layers,
and the VAE remain FP16; current
[Diffusers quantization](https://huggingface.co/docs/diffusers/quantization/quanto)
does not provide a fully integer end-to-end pipeline. The cached source files
may therefore still be published as FP16. The Ollama model remains unchanged at
`llama3.2:3b` and runs on the CPU.

## Requirements

- NVIDIA GPU with at least 8 GB VRAM
- Current NVIDIA driver
- 16 GB RAM recommended
- Internet access for installation and the first model download
- Roughly 2-8 GB disk space per model, or about 16 GB when caching all tiers

## Run with Docker (recommended)

Docker installs packages directly in the container; it does not create a
virtual environment inside the container.

Install Docker Engine, Docker Compose, and NVIDIA Container Toolkit. Then
confirm that containers can use the GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

Build and start the application:

```bash
docker compose up --build
```

Open <http://localhost:7860>. Docker also starts Ollama and keeps downloaded
image models in the persistent `hf-cache` volume. Stop everything with
`Ctrl+C`, then:

```bash
docker compose down
```

The UI downloads a model when you first select it. To cache models before
starting the UI, download one tier or all tiers:

```bash
docker compose run --rm --no-deps animegen \
  python scripts/download_models.py --model best

docker compose run --rm --no-deps animegen \
  python scripts/download_models.py --model all
```

Run the CLI in Docker:

```bash
docker compose run --rm animegen generate \
  "friends at a rooftop party at sunset" --model best
```

Do not run the UI and a generation container together on an 8 GB GPU; each
process tries to load an image model.

## Run with a local virtual environment

Create and activate an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead.

Install CUDA PyTorch and the project:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
python -c "import torch; print(torch.cuda.is_available())"
```

The last command must print `True`. Models download automatically on first use,
or can be cached in advance:

```bash
python scripts/download_models.py --model balanced
```

Start the UI or generate from the CLI:

```bash
animegen ui
animegen generate "friends at a rooftop party at sunset" --model best
```

Rendering works without Ollama. For optional prompt enhancement, install
[Ollama](https://ollama.com/) and keep the configured model unchanged:

```bash
ollama pull llama3.2:3b
```

## Common commands

```bash
# Show the three profiles and active precision
animegen info

# Fast anime draft without Ollama
animegen generate "beach bonfire" --model fast --no-llm

# Two reproducible balanced images
animegen generate "night market celebration" \
  --model balanced --images 2 --seed 1234

# Best-quality square image
animegen generate "friends in an arcade" \
  --model best --size 1024x1024 --seed 42
```

| Option | Default | Description |
|---|---:|---|
| `--model` | `balanced` | `best`, `balanced`, or `fast` |
| `--images`, `-n` | `1` | Images to generate, up to 4 |
| `--seed` | random | First image seed |
| `--size` | model default | `WIDTHxHEIGHT` |
| `--steps` | model default | Denoising steps, clamped to the model's tested range |
| `--guidance` | model default | Classifier-free guidance scale |
| `--no-llm` | off | Skip Ollama and use the original prompt |
| `--warmup` | off | Run a one-step throwaway render first |

Use `animegen generate --help` for the full CLI reference.

## Configuration

Defaults live in [`config/settings.yaml`](config/settings.yaml). Set the default
image profile without changing the YAML:

```bash
ANIMEGEN_MODEL__DEFAULT=best animegen ui
```

Other nested settings use the same `ANIMEGEN_` prefix and `__` delimiter:

```bash
ANIMEGEN_OLLAMA__HOST=http://192.168.1.10:11434 animegen info
```

For Docker port and warmup defaults, copy `.env.example` to `.env`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Docker socket permission denied | Add your user to the `docker` group, then fully log out or reboot before reopening the terminal or VS Code |
| Docker cannot select the `nvidia` driver | Install NVIDIA Container Toolkit and restart Docker |
| Docker cannot reach Docker Hub | Fix the host/Docker DNS or HTTPS proxy; the Dockerfile syntax line is not the cause |
| `torch.cuda.is_available()` is `False` | Check `nvidia-smi`, the CUDA PyTorch build, and Docker GPU access |
| CUDA out of memory | Close GPU-heavy programs, generate one image at a time, and avoid running UI and CLI together |
| First generation is very slow | The model is downloading/loading; later images use the warm pipeline |
| Ollama is unavailable | Start Ollama, or use `--no-llm`; image generation still works |
| Black or corrupted output | Keep the configured `madebyollin/sdxl-vae-fp16-fix` VAE and re-download that profile |

Useful Docker diagnostics:

```bash
docker compose ps
docker compose logs animegen
```

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```

Or in Docker:

```bash
docker compose run --rm --no-deps animegen pytest
```

See [`DEMO.md`](DEMO.md) for the live-demo checklist.
