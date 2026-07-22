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

    assert settings.generation.steps == 6
    assert settings.generation.min_steps == 4
    assert settings.generation.max_steps == 8
    assert settings.generation.guidance_scale == pytest.approx(2.0)
    assert (settings.generation.width, settings.generation.height) == (832, 1216)
    assert settings.generation.allowed_sizes == ["832x1216", "1024x1024"]
    assert settings.generation.scheduler == "DPMSolverSinglestepScheduler"
    assert settings.generation.use_karras_sigmas is True

    assert settings.model.vae_repo == "madebyollin/sdxl-vae-fp16-fix"
    assert settings.model.checkpoint_filename.endswith(".safetensors")
    assert settings.model.checkpoint_url is None

    assert settings.vram.dtype == "float16"
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
    monkeypatch.setenv("ANIMEGEN_GENERATION__STEPS", "8")
    monkeypatch.setenv("ANIMEGEN_OLLAMA__HOST", "http://192.168.1.5:11434/")

    settings = load_settings(REPO_CONFIG)

    assert settings.generation.steps == 8
    assert settings.ollama.host == "http://192.168.1.5:11434"
    # Untouched sibling keys still come from the YAML file.
    assert settings.generation.guidance_scale == pytest.approx(2.0)
    assert settings.ollama.model == "llama3.2:3b"


def test_keyword_overrides_beat_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANIMEGEN_GENERATION__STEPS", "8")

    settings = load_settings(REPO_CONFIG, generation={"steps": 4})

    assert settings.generation.steps == 4


def test_custom_config_file_is_read(clean_env: None, tmp_path: Path) -> None:
    config_file = tmp_path / "custom.yaml"
    config_file.write_text(
        "generation:\n  steps: 7\n  width: 1024\n  height: 1024\n",
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.generation.steps == 7
    assert (settings.generation.width, settings.generation.height) == (1024, 1024)
    # Keys absent from the custom file fall back to in-code defaults.
    assert settings.generation.guidance_scale == pytest.approx(2.0)


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

    assert settings.generation.steps == 6
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

    assert settings.checkpoint_path == (
        tmp_path / "m" / settings.model.checkpoint_filename
    )
    assert settings.outputs_dir == tmp_path / "o"
    assert settings.ollama.generate_url == "http://localhost:11434/api/generate"


def test_get_settings_is_cached(clean_env: None) -> None:
    first = get_settings()
    second = get_settings()

    assert first is second
    assert isinstance(first, Settings)

    reset_settings_cache()
    assert get_settings() is not first
