"""Keep ambient ANIMEGEN_* variables out of unit tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_animegen_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ANIMEGEN_* variables so tests never read the host/container config."""
    for key in list(os.environ):
        if key.startswith("ANIMEGEN_"):
            monkeypatch.delenv(key, raising=False)
