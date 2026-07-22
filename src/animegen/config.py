"""Typed, YAML-backed configuration for the Anime Party Generator.

All tunables live in ``config/settings.yaml``. Values can be overridden by
environment variables using the ``ANIMEGEN_`` prefix and ``__`` as the nesting
delimiter::

    ANIMEGEN_GENERATION__STEPS=8
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
from pydantic import BaseModel, Field, field_validator
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


class ModelSettings(BaseModel):
    """Checkpoint and VAE identifiers."""

    checkpoint_filename: str = "DreamShaperXL_v2_Turbo.safetensors"
    checkpoint_name: str = "DreamShaper XL v2 Turbo"
    checkpoint_url: str | None = None
    vae_repo: str = "madebyollin/sdxl-vae-fp16-fix"


class GenerationSettings(BaseModel):
    """Turbo-class inference defaults."""

    scheduler: str = "DPMSolverSinglestepScheduler"
    use_karras_sigmas: bool = True
    steps: int = 6
    min_steps: int = 4
    max_steps: int = 8
    guidance_scale: float = 2.0
    width: int = 832
    height: int = 1216
    allowed_sizes: list[str] = Field(default_factory=lambda: ["832x1216", "1024x1024"])
    images_per_run: int = 1
    max_images_per_run: int = 4


class VramSettings(BaseModel):
    """8 GB VRAM strategy switches. Defaults are the supported configuration."""

    dtype: str = "float16"
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

    @property
    def checkpoint_path(self) -> Path:
        """Absolute path of the SDXL single-file checkpoint."""
        return (self.paths.models_dir / self.model.checkpoint_filename).resolve()

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
