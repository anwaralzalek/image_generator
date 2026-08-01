#
# Image Generator — CUDA-capable image for an 8 GB card.
#
# The container is already an isolated environment, so dependencies are
# installed into the image's Python environment rather than a nested virtualenv.
# The pinned PyTorch runtime image provides Torch, CUDA, and cuDNN; NVIDIA
# Container Toolkit injects the host driver when the container starts.
#
# Model weights are not baked in; the Compose hf-cache volume stores them.

FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install the package from the minimum files needed at runtime.
COPY pyproject.toml README.md ./
COPY src ./src
# This image marks its Python installation as externally managed (PEP 668).
# Installing into it is intentional here so the app reuses the bundled
# PyTorch/CUDA runtime instead of downloading a second copy into a virtualenv.
RUN python -m pip install --break-system-packages ".[ui]"

RUN mkdir -p /outputs /hf-cache \
    && chown 1000:1000 /outputs /hf-cache

ENV ANIMEGEN_PATHS__OUTPUTS_DIR=/outputs \
    HF_HOME=/hf-cache \
    GRADIO_ANALYTICS_ENABLED=False

# The Ubuntu-based PyTorch image already provides the host-friendly UID/GID 1000.
USER 1000:1000
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:7860', timeout=2)"]

ENTRYPOINT ["animegen"]
CMD ["ui", "--host", "0.0.0.0"]
