"""Model profiles, runtime constants, and environment-backed settings."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

LINEAR_WEIGHT_DTYPE = "int8"
COMPUTE_DTYPE = "float16"
QUANTIZATION_BACKEND = "quanto"
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"
MAX_IMAGES = 4
SEED_MAX = 2**32 - 1

STYLE_SUFFIX = (
    ", semi-realistic 2.5D anime style, 3D-shaded characters, volumetric lighting, "
    "glossy rendering, detailed faces, cinematic composition, high detail"
)
NEGATIVE_PROMPT = (
    "bad anatomy, deformed hands, extra fingers, extra limbs, mutated, lowres, "
    "blurry, watermark, text, jpeg artifacts, flat 2D shading"
)
SYSTEM_PROMPT = (
    "You are a Stable Diffusion prompt engineer specializing in Semi-3D Anime. "
    "Transform every user idea into exactly ONE comma-separated visual prompt. "
    "Preserve the subject, character count, identity, action, setting, and mood. "
    "Always include the exact phrase 'semi-3D anime style' and enrich the image "
    "with expressive anime features, stylized proportions, layered hair and "
    "clothing, clean linework, cel-shaded color blended with soft 3D forms and "
    "materials, realistic depth, cinematic composition, volumetric lighting, and "
    "rim light. Reinterpret any requested aesthetic through Semi-3D Anime; never "
    "omit or replace the required style. Output ONLY the final prompt, with no "
    "quotes, label, markdown, or explanation, and keep it under 80 tokens."
)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    repo_id: str
    architecture: Literal["sd", "sdxl"]
    quality: str
    parameter_count: str
    estimated_seconds: tuple[int, int]
    width: int
    height: int
    allowed_sizes: tuple[str, ...]
    steps: int
    min_steps: int
    max_steps: int
    guidance_scale: float
    scheduler: str
    variant: str | None = None
    use_karras_sigmas: bool = False
    prompt_suffix: str = ""
    negative_prompt: str | None = None

    @property
    def estimate(self) -> str:
        low, high = self.estimated_seconds
        return f"{low}-{high} seconds"

    @property
    def default_size(self) -> str:
        return f"{self.width}x{self.height}"


MODEL_PROFILES: dict[str, ModelProfile] = {
    "best": ModelProfile(
        name="Animagine XL 4.0 Opt",
        repo_id="cagliostrolab/animagine-xl-4.0",
        architecture="sdxl",
        quality="Best quality",
        parameter_count="3B",
        estimated_seconds=(60, 120),
        width=832,
        height=1216,
        allowed_sizes=("832x1216", "1024x1024"),
        steps=28,
        min_steps=25,
        max_steps=32,
        guidance_scale=5.0,
        scheduler="EulerAncestralDiscreteScheduler",
        prompt_suffix=", masterpiece, high score, great score, absurdres",
        negative_prompt=(
            "lowres, bad anatomy, bad hands, text, error, missing finger, extra "
            "digits, fewer digits, cropped, worst quality, low quality, low score, "
            "bad score, average score, signature, watermark, username, blurry"
        ),
    ),
    "balanced": ModelProfile(
        name="DreamShaper XL v2 Turbo",
        repo_id="Lykon/dreamshaper-xl-v2-turbo",
        architecture="sdxl",
        quality="Balanced",
        parameter_count="3B",
        estimated_seconds=(15, 30),
        width=832,
        height=1216,
        allowed_sizes=("832x1216", "1024x1024"),
        steps=6,
        min_steps=4,
        max_steps=8,
        guidance_scale=2.0,
        scheduler="DPMSolverMultistepScheduler",
        variant="fp16",
        use_karras_sigmas=True,
    ),
    "fast": ModelProfile(
        name="Eimis Anime Diffusion 1.0v",
        repo_id="eimiss/EimisAnimeDiffusion_1.0v",
        architecture="sd",
        quality="Fast / lowest tier",
        parameter_count="0.9B",
        estimated_seconds=(10, 25),
        width=768,
        height=832,
        allowed_sizes=("768x832", "896x640", "768x768"),
        steps=20,
        min_steps=20,
        max_steps=35,
        guidance_scale=9.0,
        scheduler="DPMSolverSinglestepScheduler",
        use_karras_sigmas=True,
        prompt_suffix=", anime illustration, masterpiece, best quality, highres",
        negative_prompt=(
            "lowres, bad anatomy, bad hands, text, error, missing fingers, extra "
            "digits, cropped, worst quality, low quality, normal quality, signature, "
            "watermark, username, blurry"
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout: float = 30.0
    temperature: float = 0.7

    def __post_init__(self) -> None:
        host = self.host.rstrip("/")
        if not host.startswith(("http://", "https://")):
            raise ValueError("Ollama host must start with http:// or https://")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Ollama timeout must be a positive finite number")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Ollama temperature must be a finite, non-negative number")
        object.__setattr__(self, "host", host)

    @property
    def generate_url(self) -> str:
        return f"{self.host}/api/generate"


@dataclass(slots=True)
class Settings:
    default_model: str = "balanced"
    outputs_dir: Path = Path("outputs")
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    profiles: dict[str, ModelProfile] = field(
        default_factory=lambda: dict(MODEL_PROFILES)
    )

    def __post_init__(self) -> None:
        self.outputs_dir = Path(self.outputs_dir).expanduser().resolve()
        if self.default_model not in self.profiles:
            choices = ", ".join(self.profiles)
            raise ValueError(
                f"Unknown default image model '{self.default_model}'. Choose: {choices}"
            )

    def image_model(self, key: str | None = None) -> tuple[str, ModelProfile]:
        selected = self.default_model if key is None else key
        try:
            return selected, self.profiles[selected]
        except KeyError:
            choices = ", ".join(self.profiles)
            raise ValueError(
                f"Unknown image model '{selected}'. Choose one of: {choices}"
            ) from None


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def load_settings() -> Settings:
    """Read the small set of supported runtime overrides from the environment."""
    return Settings(
        default_model=os.getenv("ANIMEGEN_MODEL__DEFAULT", "balanced"),
        outputs_dir=Path(os.getenv("ANIMEGEN_PATHS__OUTPUTS_DIR", "outputs")),
        ollama=OllamaSettings(
            host=os.getenv("ANIMEGEN_OLLAMA__HOST", "http://localhost:11434"),
            model=os.getenv("ANIMEGEN_OLLAMA__MODEL", "llama3.2:3b"),
            timeout=_env_float("ANIMEGEN_OLLAMA__TIMEOUT", 30.0),
            temperature=_env_float("ANIMEGEN_OLLAMA__TEMPERATURE", 0.7),
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def parse_dimensions(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT and enforce the diffusion latent stride."""
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"'{value}' is not a WIDTHxHEIGHT size, e.g. 832x1216")
    width, height = map(int, parts)
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise ValueError(f"'{value}': width and height must be positive multiples of 8")
    return width, height
