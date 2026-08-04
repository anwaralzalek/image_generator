# Anime Party Generator

Generate semi-realistic 2.5D anime images locally with DreamShaper XL v2 Turbo.
Ollama can optionally expand short ideas into detailed prompts; if it is
unavailable, the application renders the original prompt instead.

The project provides:

- A Gradio web interface and command-line interface.
- One to four images per request with reproducible seeds.
- PNG output with a matching JSON metadata file.
- Memory-saving defaults for NVIDIA GPUs with 8 GB of VRAM.

Generated files are saved in `outputs/`.

## Requirements

- An NVIDIA GPU with 8 GB or more VRAM is recommended.
- A current NVIDIA driver.
- About 8 GB of free disk space for the checkpoint and VAE.
- Internet access during the initial setup.

The required checkpoint is **DreamShaper XL v2 Turbo**. Download it from
[Civitai](https://civitai.com/models/112902) or the
[Hugging Face mirror](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo),
then save it as:

```text
models/DreamShaperXL_v2_Turbo.safetensors
```

You can also give the included download script a direct checkpoint URL, as
shown in each setup method below.

## Option 1: Docker (recommended)

Docker keeps the application and its dependencies isolated and also starts the
optional Ollama prompt enhancer. Python packages are installed directly in the
container; no virtual environment is created inside it.

### Prerequisites

- Docker Engine and Docker Compose.
- NVIDIA Container Toolkit on Linux, or Docker Desktop with WSL2 on Windows.

> Docker Desktop for Linux does not provide NVIDIA GPU access. On Linux, use
> native Docker Engine with NVIDIA Container Toolkit.

Confirm that Docker can access the GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

### Build and download the models

Build the application image:

```bash
docker compose build animegen
```

If you already placed the checkpoint in `models/`, download only the required
fp16-safe VAE:

```bash
docker compose run --rm --no-deps animegen \
  python scripts/download_models.py --skip-checkpoint
```

Alternatively, download both the checkpoint and VAE with a direct URL:

```bash
docker compose run --rm --no-deps animegen \
  python scripts/download_models.py --url "<direct-checkpoint-url>"
```

Add `--token "<access-token>"` if the download source requires authentication.

### Run the web interface

```bash
docker compose up
```

Open <http://localhost:7860>. The first start also downloads
`llama3.2:3b` for Ollama. Images are written to `outputs/` on the host.

Stop the application with `Ctrl+C`, followed by:

```bash
docker compose down
```

### Run from the command line

```bash
docker compose run --rm animegen generate \
  "friends at a rooftop party at sunset"
```

Do not run the web interface and a separate generation container at the same
time on an 8 GB GPU, because both processes load the image model.

## Option 2: Local virtual environment

Use this method when Docker GPU support is unavailable or when developing the
Python application directly.

### Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

### Install the dependencies

Install the CUDA-enabled PyTorch package first, followed by the project:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

Confirm that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The result should be `True`.

### Download the models

If the checkpoint is already in `models/`, download only the VAE:

```bash
python scripts/download_models.py --skip-checkpoint
```

Or download both with a direct checkpoint URL:

```bash
python scripts/download_models.py --url "<direct-checkpoint-url>"
```

### Optional: install Ollama

Rendering works without Ollama. To enable prompt enhancement, install
[Ollama](https://ollama.com/) and pull the configured model:

```bash
ollama pull llama3.2:3b
```

### Run the application

Check the setup:

```bash
animegen info
```

Start the web interface and open <http://127.0.0.1:7860>:

```bash
animegen ui
```

Or generate an image from the command line:

```bash
animegen generate "friends at a rooftop party at sunset"
```

## CLI examples

The same options work locally and in Docker. For Docker, place them after
`docker compose run --rm animegen`.

```bash
# Four reproducible images
animegen generate "night market celebration" --images 4 --seed 1234

# Skip Ollama and render the original prompt
animegen generate "beach bonfire" --no-llm

# Use a square image and custom generation settings
animegen generate "friends in an arcade" \
  --size 1024x1024 --steps 8 --guidance 2.0
```

| Option | Default | Description |
|---|---:|---|
| `--images`, `-n` | `1` | Number of images, up to 4. |
| `--seed` | random | Seed for reproducible output. |
| `--size` | `832x1216` | Output dimensions. |
| `--steps` | `6` | Denoising steps; configured range is 4-8. |
| `--guidance` | `2.0` | Classifier-free guidance scale. |
| `--no-llm` | off | Bypass Ollama. |
| `--warmup` | off | Perform a throwaway render before generation. |

Run `animegen generate --help` for the complete command reference.

## Configuration

Defaults are in [`config/settings.yaml`](config/settings.yaml). Environment
variables override them using the `ANIMEGEN_` prefix and `__` for nested keys:

```bash
ANIMEGEN_GENERATION__STEPS=8 animegen generate "city festival"
ANIMEGEN_OLLAMA__HOST=http://192.168.1.10:11434 animegen info
```

For Docker port and warmup settings, copy the example environment file:

```bash
cp .env.example .env
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `SDXL checkpoint not found` | Save the checkpoint with the exact filename shown above, then run `animegen info`. |
| Docker socket permission denied | Add your user to the `docker` group, then fully log out or reboot before reopening the terminal or VS Code. |
| Docker cannot select the `nvidia` driver | Install NVIDIA Container Toolkit and restart Docker. |
| `torch.cuda.is_available()` is `False` | Check `nvidia-smi`, the CUDA PyTorch package, and Docker GPU access if applicable. |
| CUDA out of memory | Close GPU-heavy programs, generate one image at a time, and use `832x1216`. |
| Ollama is unavailable | Start Ollama and pull `llama3.2:3b`, or use `--no-llm`. Rendering still works. |
| Images are black or corrupted | Run the VAE download command again and keep `madebyollin/sdxl-vae-fp16-fix` configured. |

For Docker diagnostics:

```bash
docker compose ps
docker compose logs animegen
```

## Tests

Local environment:

```bash
python -m pip install -e ".[dev]"
pytest
```

Docker:

```bash
docker compose run --rm --no-deps animegen pytest
```

See [`DEMO.md`](DEMO.md) for the live demonstration checklist.
