#
# Anime Party Generator — CUDA-capable image for an 8 GB card.
#
# The container is already an isolated environment, so dependencies are
# installed into the image's system Python rather than a nested virtualenv.
# CUDA-enabled torch wheels provide the CUDA runtime libraries; NVIDIA
# Container Toolkit injects the host driver when the container starts.
#
# Model weights are not baked in; the Compose hf-cache volume stores them.

FROM python:3.10-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Install the package from the minimum files needed at runtime.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install torch==2.13.0 --index-url "${TORCH_INDEX_URL}" \
    && pip install ".[ui]"

RUN mkdir -p /outputs /hf-cache \
    && chown appuser:appuser /outputs /hf-cache

ENV ANIMEGEN_PATHS__OUTPUTS_DIR=/outputs \
    HF_HOME=/hf-cache \
    GRADIO_ANALYTICS_ENABLED=False

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:7860', timeout=2)"]

ENTRYPOINT ["animegen"]
CMD ["ui", "--host", "0.0.0.0"]
