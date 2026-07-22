"""Typed, YAML-backed configuration for the Anime Party Generator.

All tunables live in ``config/settings.yaml``. Values can be overridden by
environment variables using the ``ANIMEGEN_`` prefix and ``__`` as the nesting
delimiter::

    ANIMEGEN_MODEL__DEFAULT=best
    ANIMEGEN_OLLAMA__HOST=http://127.0.0.1:11434

Precedence, highest first: explicit keyword arguments, environment variables,
``.env`` file, ``settings.yaml``, in-code defaults. The YAML file is optional:
every field carries a default identical to the shipped file, so an installed
package without a config directory still runs.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

LOGGER = logging.getLogger(__name__)

#: ``src/animegen/config.py`` -> ``src/animegen`` -> ``src`` -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
CONFIG_FILE_ENV_VAR = "ANIMEGEN_CONFIG_FILE"

DEFAULT_STYLE_SUFFIX = (
    ", semi-realistic 2.5D anime style, 3D-shaded characters, volumetric lighting, "
    "glossy rendering, detailed faces, cinematic composition, high detail"
)
DEFAULT_NEGATIVE_PROMPT = (
    "bad anatomy, deformed hands, extra fingers, extra limbs, mutated, lowres, "
    "blurry, watermark, text, jpeg artifacts, flat 2D shading"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a Stable Diffusion XL prompt engineer. Expand the user's idea into "
    "ONE comma-separated visual prompt: subjects, count of people, setting, "
    "lighting, mood, camera angle, clothing. Output ONLY the prompt, no quotes, "
    "no explanations, under 60 tokens."
)

_CONFIG_FILE_OVERRIDE: ContextVar[Path | None] = ContextVar(
    "animegen_config_file", default=None
)


def active_config_file() -> Path:
    """Return the settings file that :func:`load_settings` will read.

    Resolution order: explicit path passed to :func:`load_settings`, the
    ``ANIMEGEN_CONFIG_FILE`` environment variable, ``./config/settings.yaml``,
    then the file shipped next to the source tree.
    """
    override = _CONFIG_FILE_OVERRIDE.get()
    if override is not None:
        return override

    env_value = os.getenv(CONFIG_FILE_ENV_VAR)
    if env_value:
        return Path(env_value)

    cwd_candidate = Path.cwd() / "config" / "settings.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate
    return DEFAULT_CONFIG_FILE


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Settings source that reads a flat mapping of sections from a YAML file.

    A missing file is not an error: the source yields an empty mapping and the
    model defaults apply.
    """

    def __init__(self, settings_cls: type[BaseSettings], yaml_file: Path) -> None:
        super().__init__(settings_cls)
        self.yaml_file = yaml_file
        self._data = self._read_file()

    def _read_file(self) -> dict[str, Any]:
        if not self.yaml_file.is_file():
            LOGGER.warning(
                "Settings file %s not found; falling back to built-in defaults",
                self.yaml_file,
            )
            return {}
        with self.yaml_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError(
                f"{self.yaml_file} must contain a YAML mapping, got {type(data).__name__}"
            )
        return data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class PathSettings(BaseModel):
    """Filesystem locations. Relative paths resolve against the process CWD."""

    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")


class ModelProfile(BaseModel):
    """One selectable image model and its recommended inference settings."""

    name: str
    repo_id: str
    variant: str | None = None
    architecture: str
    quality: str
    parameter_count: str
    estimated_seconds: tuple[int, int]
    width: int
    height: int
    allowed_sizes: list[str]
    steps: int
    min_steps: int
    max_steps: int
    guidance_scale: float
    scheduler: str
    use_karras_sigmas: bool = False
    quantize_diffusers: list[str] = Field(default_factory=lambda: ["unet"])
    quantize_transformers: list[str] = Field(default_factory=lambda: ["text_encoder"])
    prompt_suffix: str = ""
    negative_prompt: str | None = None

    @field_validator("architecture")
    @classmethod
    def _supported_architecture(cls, value: str) -> str:
        if value not in {"sdxl", "sd"}:
            raise ValueError("architecture must be 'sdxl' or 'sd'")
        return value

    @model_validator(mode="after")
    def _valid_ranges(self) -> "ModelProfile":
        low, high = self.estimated_seconds
        if low <= 0 or high < low:
            raise ValueError("estimated_seconds must be a positive (low, high) range")
        if not self.min_steps <= self.steps <= self.max_steps:
            raise ValueError("steps must be inside min_steps and max_steps")
        return self

    @property
    def estimate(self) -> str:
        """Human-readable warm-pipeline render estimate for one image."""
        low, high = self.estimated_seconds
        return f"{low}-{high} seconds"

    @property
    def default_size(self) -> str:
        return f"{self.width}x{self.height}"


def _default_model_profiles() -> dict[str, ModelProfile]:
    """Built-in profiles used when no YAML file is available."""
    return {
        "best": ModelProfile(
            name="Animagine XL 4.0 Opt",
            repo_id="cagliostrolab/animagine-xl-4.0",
            architecture="sdxl",
            quality="Best quality",
            parameter_count="3B",
            estimated_seconds=(60, 120),
            width=832,
            height=1216,
            allowed_sizes=["832x1216", "1024x1024"],
            steps=28,
            min_steps=25,
            max_steps=32,
            guidance_scale=5.0,
            scheduler="EulerAncestralDiscreteScheduler",
            quantize_transformers=["text_encoder", "text_encoder_2"],
            prompt_suffix=", masterpiece, high score, great score, absurdres",
            negative_prompt=(
                "lowres, bad anatomy, bad hands, text, error, missing finger, "
                "extra digits, fewer digits, cropped, worst quality, low quality, "
                "low score, bad score, average score, signature, watermark, "
                "username, blurry"
            ),
        ),
        "balanced": ModelProfile(
            name="DreamShaper XL v2 Turbo",
            repo_id="Lykon/dreamshaper-xl-v2-turbo",
            variant="fp16",
            architecture="sdxl",
            quality="Balanced",
            parameter_count="3B",
            estimated_seconds=(15, 30),
            width=832,
            height=1216,
            allowed_sizes=["832x1216", "1024x1024"],
            steps=6,
            min_steps=4,
            max_steps=8,
            guidance_scale=2.0,
            scheduler="DPMSolverMultistepScheduler",
            use_karras_sigmas=True,
            quantize_transformers=["text_encoder", "text_encoder_2"],
        ),
        "fast": ModelProfile(
            name="Dreamlike Anime 1.0",
            repo_id="dreamlike-art/dreamlike-anime-1.0",
            architecture="sd",
            quality="Fast / lowest tier",
            parameter_count="0.9B",
            estimated_seconds=(10, 25),
            width=768,
            height=768,
            allowed_sizes=["768x768", "704x832", "832x704"],
            steps=20,
            min_steps=15,
            max_steps=30,
            guidance_scale=7.5,
            scheduler="DPMSolverMultistepScheduler",
            use_karras_sigmas=True,
            prompt_suffix=", photo anime, masterpiece, high quality, absurdres",
            negative_prompt=(
                "simple background, duplicate, retro style, low quality, lowest "
                "quality, bad anatomy, bad proportions, extra digits, lowres, "
                "username, artist name, error, watermark, signature, text, jpeg "
                "artifacts, blurry"
            ),
        ),
    }


class ModelSettings(BaseModel):
    """Selectable image-model registry and shared SDXL VAE."""

    default: str = "balanced"
    vae_repo: str = "madebyollin/sdxl-vae-fp16-fix"
    profiles: dict[str, ModelProfile] = Field(default_factory=_default_model_profiles)

    @model_validator(mode="after")
    def _default_exists(self) -> "ModelSettings":
        if self.default not in self.profiles:
            choices = ", ".join(self.profiles)
            raise ValueError(f"default model '{self.default}' is not one of: {choices}")
        return self

    def get(self, key: str | None = None) -> tuple[str, ModelProfile]:
        """Resolve a profile key, raising a useful error for CLI/UI callers."""
        selected = key or self.default
        try:
            return selected, self.profiles[selected]
        except KeyError:
            choices = ", ".join(self.profiles)
            raise ValueError(
                f"Unknown image model '{selected}'. Choose one of: {choices}"
            ) from None


class GenerationSettings(BaseModel):
    """Limits shared by every image-model profile."""

    images_per_run: int = 1
    max_images_per_run: int = 4


class VramSettings(BaseModel):
    """8 GB VRAM strategy switches. Defaults are the supported configuration."""

    weight_dtype: str = "int8"
    dtype: str = "float16"
    quantization_backend: str = "quanto"
    enable_model_cpu_offload: bool = True
    enable_vae_tiling: bool = True
    empty_cache_after_run: bool = True


class OllamaSettings(BaseModel):
    """Local LLM endpoint used for prompt enhancement."""

    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout: float = 30.0
    temperature: float = 0.7
    num_gpu: int = 0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @field_validator("host")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def generate_url(self) -> str:
        """Full URL of the Ollama ``/api/generate`` endpoint."""
        return f"{self.host}/api/generate"


class StyleSettings(BaseModel):
    """The style contract applied to every generation."""

    suffix: str = DEFAULT_STYLE_SUFFIX
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="ANIMEGEN_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    paths: PathSettings = Field(default_factory=PathSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    vram: VramSettings = Field(default_factory=VramSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    style: StyleSettings = Field(default_factory=StyleSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML source below env vars but above in-code defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls, active_config_file()),
            file_secret_settings,
        )

    def image_model(self, key: str | None = None) -> tuple[str, ModelProfile]:
        """Return the selected image model key and profile."""
        return self.model.get(key)

    @property
    def allowed_sizes(self) -> list[str]:
        """Ordered union of the sizes supported by all model profiles."""
        sizes: list[str] = []
        for profile in self.model.profiles.values():
            for size in profile.allowed_sizes:
                if size not in sizes:
                    sizes.append(size)
        return sizes

    @property
    def outputs_dir(self) -> Path:
        """Absolute path of the directory receiving images and metadata."""
        return self.paths.outputs_dir.resolve()


def load_settings(config_path: str | Path | None = None, **overrides: Any) -> Settings:
    """Build a :class:`Settings` instance.

    Args:
        config_path: Optional settings file overriding the default lookup.
        **overrides: Section-level values that win over env vars and YAML.

    Returns:
        A fully validated :class:`Settings` instance.
    """
    token = _CONFIG_FILE_OVERRIDE.set(Path(config_path) if config_path else None)
    try:
        return Settings(**overrides)
    finally:
        _CONFIG_FILE_OVERRIDE.reset(token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings, loaded once and cached."""
    return load_settings()


def reset_settings_cache() -> None:
    """Drop the :func:`get_settings` cache (used by tests and the UI reloader)."""
    get_settings.cache_clear()
