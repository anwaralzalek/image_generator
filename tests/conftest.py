"""Shared test fixtures.

The container image and docker-compose both export ANIMEGEN_* variables
(ANIMEGEN_OLLAMA__HOST, ANIMEGEN_PATHS__*, ANIMEGEN_CONFIG_FILE). Those beat
settings.yaml by design, which means a suite that reads ambient configuration
would pass on a laptop and fail inside the container. Every test therefore
starts from a clean environment and opts in to whatever it needs.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_animegen_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ANIMEGEN_* variables so tests never read the host/container config."""
    for key in list(os.environ):
        if key.startswith("ANIMEGEN_"):
            monkeypatch.delenv(key, raising=False)
