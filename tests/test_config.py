"""Tests for the YAML-backed settings loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from animegen.config import (
    DEFAULT_CONFIG_FILE,
    Settings,
    active_config_file,
    get_settings,
    load_settings,
    reset_settings_cache,
)

REPO_CONFIG = DEFAULT_CONFIG_FILE


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any ANIMEGEN_* variables leaking in from the developer's shell."""
    import os

    for key in list(os.environ):
        if key.startswith("ANIMEGEN_"):
            monkeypatch.delenv(key, raising=False)


def test_shipped_config_file_exists() -> None:
    assert REPO_CONFIG.is_file(), "config/settings.yaml must ship with the repo"


def test_defaults_match_the_documented_contract(clean_env: None) -> None:
    settings = load_settings(REPO_CONFIG)

    assert settings.model.default == "balanced"
    assert list(settings.model.profiles) == ["best", "balanced", "fast"]
    best = settings.model.profiles["best"]
    balanced = settings.model.profiles["balanced"]
    fast = settings.model.profiles["fast"]
    assert best.name == "Animagine XL 4.0 Opt"
    assert best.parameter_count == "3B"
    assert best.steps == 28
    assert best.estimate == "60-120 seconds"
    assert balanced.name == "DreamShaper XL v2 Turbo"
    assert balanced.steps == 6
    assert fast.name == "Dreamlike Anime 1.0"
    assert fast.default_size == "768x768"
    assert settings.allowed_sizes == [
        "832x1216",
        "1024x1024",
        "768x768",
        "704x832",
        "832x704",
    ]
    assert settings.model.vae_repo == "madebyollin/sdxl-vae-fp16-fix"

    assert settings.vram.weight_dtype == "int8"
    assert settings.vram.dtype == "float16"
    assert settings.vram.quantization_backend == "quanto"
    assert settings.vram.enable_model_cpu_offload is True
    assert settings.vram.enable_vae_tiling is True

    assert settings.ollama.model == "llama3.2:3b"
    assert settings.ollama.host == "http://localhost:11434"
    assert settings.ollama.timeout == pytest.approx(30.0)
    assert settings.ollama.num_gpu == 0, "the LLM must stay off the GPU"
    assert "Stable Diffusion XL prompt engineer" in settings.ollama.system_prompt


def test_style_contract_is_verbatim(clean_env: None) -> None:
    settings = load_settings(REPO_CONFIG)

    assert settings.style.suffix == (
        ", semi-realistic 2.5D anime style, 3D-shaded characters, volumetric lighting,"
        " glossy rendering, detailed faces, cinematic composition, high detail"
    )
    assert settings.style.negative_prompt == (
        "bad anatomy, deformed hands, extra fingers, extra limbs, mutated, lowres,"
        " blurry, watermark, text, jpeg artifacts, flat 2D shading"
    )


def test_env_var_overrides_yaml_value(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANIMEGEN_MODEL__DEFAULT", "best")
    monkeypatch.setenv("ANIMEGEN_OLLAMA__HOST", "http://192.168.1.5:11434/")

    settings = load_settings(REPO_CONFIG)

    assert settings.model.default == "best"
    assert settings.ollama.host == "http://192.168.1.5:11434"
    # Untouched sibling keys still come from the YAML file.
    assert settings.model.profiles["balanced"].guidance_scale == pytest.approx(2.0)
    assert settings.ollama.model == "llama3.2:3b"


def test_keyword_overrides_beat_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANIMEGEN_MODEL__DEFAULT", "best")

    settings = load_settings(REPO_CONFIG, model={"default": "fast"})

    assert settings.model.default == "fast"


def test_custom_config_file_is_read(clean_env: None, tmp_path: Path) -> None:
    config_file = tmp_path / "custom.yaml"
    config_file.write_text(
        "model:\n  default: best\ngeneration:\n  max_images_per_run: 2\n",
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.model.default == "best"
    assert settings.generation.max_images_per_run == 2
    # Profiles absent from the custom file fall back to in-code defaults.
    assert settings.model.profiles["balanced"].steps == 6


def test_config_file_env_var_selects_the_file(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_file = tmp_path / "from_env.yaml"
    config_file.write_text("ollama:\n  model: llama3.2:1b\n", encoding="utf-8")
    monkeypatch.setenv("ANIMEGEN_CONFIG_FILE", str(config_file))

    assert active_config_file() == config_file
    assert load_settings().ollama.model == "llama3.2:1b"


def test_missing_config_file_falls_back_to_defaults(
    clean_env: None, tmp_path: Path
) -> None:
    settings = load_settings(tmp_path / "does-not-exist.yaml")

    assert settings.model.default == "balanced"
    assert settings.style.negative_prompt.startswith("bad anatomy")


def test_malformed_config_file_raises(clean_env: None, tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(TypeError):
        load_settings(config_file)


def test_derived_paths_and_urls(clean_env: None, tmp_path: Path) -> None:
    settings = load_settings(
        REPO_CONFIG,
        paths={"models_dir": tmp_path / "m", "outputs_dir": tmp_path / "o"},
    )

    assert settings.paths.models_dir == tmp_path / "m"
    assert settings.outputs_dir == tmp_path / "o"
    assert settings.ollama.generate_url == "http://localhost:11434/api/generate"


def test_unknown_image_model_lists_valid_choices(clean_env: None) -> None:
    settings = load_settings(REPO_CONFIG)

    with pytest.raises(ValueError, match="best, balanced, fast"):
        settings.image_model("unknown")


def test_get_settings_is_cached(clean_env: None) -> None:
    first = get_settings()
    second = get_settings()

    assert first is second
    assert isinstance(first, Settings)

    reset_settings_cache()
    assert get_settings() is not first
