"""Tests for the lightweight environment-backed configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from animegen.config import (
    COMPUTE_DTYPE,
    MAX_IMAGES,
    MODEL_PROFILES,
    NEGATIVE_PROMPT,
    QUANTIZATION_BACKEND,
    SEED_MAX,
    STYLE_SUFFIX,
    SYSTEM_PROMPT,
    VAE_REPO,
    LINEAR_WEIGHT_DTYPE,
    OllamaSettings,
    Settings,
    get_settings,
    load_settings,
    parse_dimensions,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_match_the_runtime_contract() -> None:
    settings = load_settings()

    assert settings.default_model == "balanced"
    assert list(settings.profiles) == ["best", "balanced", "fast"]
    assert settings.profiles == MODEL_PROFILES

    best = settings.profiles["best"]
    balanced = settings.profiles["balanced"]
    fast = settings.profiles["fast"]
    assert (best.name, best.parameter_count, best.steps, best.estimate) == (
        "Animagine XL 4.0 Opt",
        "3B",
        28,
        "60-120 seconds",
    )
    assert (balanced.name, balanced.steps, balanced.guidance_scale) == (
        "DreamShaper XL v2 Turbo",
        6,
        pytest.approx(2.0),
    )
    assert (fast.name, fast.default_size, fast.architecture) == (
        "Eimis Anime Diffusion 1.0v",
        "768x832",
        "sd",
    )

    assert LINEAR_WEIGHT_DTYPE == "int8"
    assert COMPUTE_DTYPE == "float16"
    assert QUANTIZATION_BACKEND == "quanto"
    assert VAE_REPO == "madebyollin/sdxl-vae-fp16-fix"
    assert (MAX_IMAGES, SEED_MAX) == (4, 2**32 - 1)


def test_prompt_constants_are_stable() -> None:
    assert STYLE_SUFFIX == (
        ", realistic 3D anime style, 3D-shaded characters, volumetric lighting, "
        "glossy rendering, detailed faces, cinematic composition, high detail"
    )
    assert NEGATIVE_PROMPT == (
        "bad anatomy, deformed hands, extra fingers, extra limbs, mutated, lowres, "
        "blurry, watermark, text, jpeg artifacts, flat 2D shading"
    )
    assert "Stable Diffusion prompt engineer specializing in 3D Anime" in (
        SYSTEM_PROMPT
    )
    assert "Always include the exact phrase '3D anime style'" in SYSTEM_PROMPT
    assert "cel-shaded color blended with soft 3D forms and materials" in SYSTEM_PROMPT
    assert "never omit or replace the required style" in SYSTEM_PROMPT
    assert "Output ONLY the final prompt" in SYSTEM_PROMPT


def test_environment_overrides_supported_runtime_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANIMEGEN_MODEL__DEFAULT", "best")
    monkeypatch.setenv("ANIMEGEN_PATHS__OUTPUTS_DIR", str(tmp_path / "renders"))
    monkeypatch.setenv("ANIMEGEN_OLLAMA__HOST", "http://gpu-box:11434/")
    monkeypatch.setenv("ANIMEGEN_OLLAMA__MODEL", "llama3.2:1b")
    monkeypatch.setenv("ANIMEGEN_OLLAMA__TIMEOUT", "12.5")
    monkeypatch.setenv("ANIMEGEN_OLLAMA__TEMPERATURE", "0.25")

    settings = load_settings()

    assert settings.default_model == "best"
    assert settings.outputs_dir == (tmp_path / "renders").resolve()
    assert settings.ollama == OllamaSettings(
        host="http://gpu-box:11434",
        model="llama3.2:1b",
        timeout=12.5,
        temperature=0.25,
    )
    assert settings.ollama.generate_url == "http://gpu-box:11434/api/generate"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ANIMEGEN_OLLAMA__TIMEOUT", "0", "positive finite"),
        ("ANIMEGEN_OLLAMA__TIMEOUT", "nan", "positive finite"),
        ("ANIMEGEN_OLLAMA__TEMPERATURE", "-0.1", "non-negative"),
        ("ANIMEGEN_OLLAMA__TEMPERATURE", "inf", "finite"),
        ("ANIMEGEN_OLLAMA__TIMEOUT", "not-a-number", "could not convert"),
    ],
)
def test_invalid_numeric_environment_values_raise(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        load_settings()


def test_unknown_default_model_from_environment_lists_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANIMEGEN_MODEL__DEFAULT", "unknown")

    with pytest.raises(ValueError, match="best, balanced, fast"):
        load_settings()


def test_invalid_ollama_host_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMEGEN_OLLAMA__HOST", "localhost:11434")

    with pytest.raises(ValueError, match="http"):
        load_settings()


def test_settings_normalise_output_path(tmp_path: Path) -> None:
    settings = Settings(outputs_dir=tmp_path / "nested" / ".." / "renders")

    assert settings.outputs_dir == (tmp_path / "renders").resolve()


def test_image_model_selection_and_validation() -> None:
    settings = Settings(default_model="fast")

    assert settings.image_model() == ("fast", MODEL_PROFILES["fast"])
    assert settings.image_model("best") == ("best", MODEL_PROFILES["best"])
    for unknown in ("", "unknown"):
        with pytest.raises(ValueError, match="best, balanced, fast"):
            settings.image_model(unknown)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("832x1216", (832, 1216)), (" 1024 X 1024 ", (1024, 1024))],
)
def test_parse_dimensions(raw: str, expected: tuple[int, int]) -> None:
    assert parse_dimensions(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "1024", "x1024", "1024x", "abcx1024", "0x1024", "801x1024"],
)
def test_parse_dimensions_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_dimensions(raw)


def test_get_settings_is_cached() -> None:
    first = get_settings()

    assert get_settings() is first
    assert isinstance(first, Settings)

    get_settings.cache_clear()
    assert get_settings() is not first
