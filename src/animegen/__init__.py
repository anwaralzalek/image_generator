"""Anime Party Generator: local 2.5D anime image generation on 8 GB VRAM.

Pipeline: a user idea is expanded by a CPU-bound Ollama model, the style
contract is appended, and DreamShaper XL v2 Turbo renders it through
``diffusers`` with SDXL memory optimisations.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
