#
# Anime Party Generator — CUDA-capable image for an 8 GB card.
#
# The container is already an isolated environment, so dependencies are
# installed into the image's system Python rather than a nested virtualenv.
# CUDA-enabled torch wheels provide the CUDA runtime libraries; NVIDIA
# Container Toolkit injects the host driver when the container starts.
#
# Model weights are not baked in; the Compose hf-cache volume stores them.

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# curl serves the healthcheck; libgomp1 is required by torch's CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Keep dependency installation above the full source copy so ordinary code
# edits do not invalidate this expensive layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url "${TORCH_INDEX_URL}" \
    && pip install ".[dev]"

COPY --chown=appuser:appuser . /app
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN mkdir -p /models /outputs /hf-cache \
    && chown appuser:appuser /app /models /outputs /hf-cache

ENV ANIMEGEN_CONFIG_FILE=/app/config/settings.yaml \
    ANIMEGEN_PATHS__MODELS_DIR=/models \
    ANIMEGEN_PATHS__OUTPUTS_DIR=/outputs \
    HF_HOME=/hf-cache \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/ || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["ui", "--host", "0.0.0.0"]
