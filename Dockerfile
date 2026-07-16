# syntax=docker/dockerfile:1.7
#
# Anime Party Generator — CUDA-capable image for an 8 GB card.
#
# Base is python:3.10-slim, not nvidia/cuda: the cu121 torch wheels already ship
# the CUDA runtime libraries they need, and the NVIDIA Container Toolkit injects
# the driver at run time. That keeps the image ~2 GB smaller than a CUDA base
# image carrying a second, unused copy of the toolkit.
#
# Model weights are never baked in (6.5 GB, licence-gated): mount them at /models.
#
# Build:  docker build -t animegen .
# Run:    docker run --rm --gpus all -p 7860:7860 -v ./models:/models -v ./outputs:/outputs animegen

ARG PYTHON_VERSION=3.10
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# --------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG TORCH_INDEX_URL
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

# torch first, from the CUDA index. The default PyPI wheel is CPU-only and would
# fail silently into 10-minute renders instead of erroring.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url "${TORCH_INDEX_URL}"

# Dependency layer: only invalidated when the manifest changes, not on every edit.
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[dev]"

# --------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Consumed by the NVIDIA Container Toolkit when the container is started with
# --gpus all; ignored on a machine without it (the app then falls back to CPU).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# curl is the healthcheck; libgomp1 is required by torch's CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser . /app
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Mount points. Weights and images are volumes, never image layers.
RUN mkdir -p /models /outputs /hf-cache \
    && chown appuser:appuser /models /outputs /hf-cache

# Paths inside the container. Every one is overridable at run time, since the
# app reads ANIMEGEN_* env vars ahead of settings.yaml.
ENV ANIMEGEN_CONFIG_FILE=/app/config/settings.yaml \
    ANIMEGEN_PATHS__MODELS_DIR=/models \
    ANIMEGEN_PATHS__OUTPUTS_DIR=/outputs \
    HF_HOME=/hf-cache \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0

USER appuser
EXPOSE 7860

# start-period covers the checkpoint load; the UI answers before the model is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/ || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["ui", "--host", "0.0.0.0"]
