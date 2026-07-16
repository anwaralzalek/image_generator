"""Tests for the Gradio demo. The handler logic is tested without gradio installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from animegen.config import DEFAULT_CONFIG_FILE, Settings, load_settings
from animegen.core.orchestrator import Orchestrator
from animegen.ui.app import DemoApp, env_flag, parse_seed, parse_size
from tests.test_orchestrator import (
    LLM_OUTPUT,
    STYLE_SUFFIX,
    USER_PROMPT,
    FakeEnhancer,
    FakeGenerator,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(
        DEFAULT_CONFIG_FILE,
        paths={"models_dir": tmp_path / "models", "outputs_dir": tmp_path / "outputs"},
    )


@pytest.fixture
def generator(settings: Settings) -> FakeGenerator:
    return FakeGenerator(settings)


@pytest.fixture
def enhancer() -> FakeEnhancer:
    return FakeEnhancer()


@pytest.fixture
def demo(
    settings: Settings, enhancer: FakeEnhancer, generator: FakeGenerator
) -> DemoApp:
    orchestrator = Orchestrator(
        settings=settings,
        enhancer=enhancer,  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
    )
    return DemoApp(settings=settings, orchestrator=orchestrator)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", None), ("   ", None), (None, None), ("1234", 1234), (" 42 ", 42), (7, 7)],
)
def test_parse_seed_accepts_blank_and_numbers(
    value: Any, expected: int | None
) -> None:
    assert parse_seed(value) == expected


@pytest.mark.parametrize("value", ["abc", "12.5", "-1", "1e5"])
def test_parse_seed_rejects_junk(value: str) -> None:
    with pytest.raises(ValueError):
        parse_seed(value)


@pytest.mark.parametrize(
    ("value", "expected"), [("832x1216", (832, 1216)), ("1024X1024", (1024, 1024))]
)
def test_parse_size_accepts_dropdown_values(
    value: str, expected: tuple[int, int]
) -> None:
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", ["huge", "832", "833x1216", "0x0"])
def test_parse_size_rejects_junk(value: str) -> None:
    with pytest.raises(ValueError):
        parse_size(value)


def test_env_flag_reads_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANIMEGEN_UI_WARMUP", raising=False)
    assert env_flag("ANIMEGEN_UI_WARMUP") is False
    assert env_flag("ANIMEGEN_UI_WARMUP", default=True) is True

    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("ANIMEGEN_UI_WARMUP", truthy)
        assert env_flag("ANIMEGEN_UI_WARMUP") is True

    monkeypatch.setenv("ANIMEGEN_UI_WARMUP", "0")
    assert env_flag("ANIMEGEN_UI_WARMUP") is False


def test_generate_returns_gallery_details_and_status(
    demo: DemoApp, generator: FakeGenerator
) -> None:
    gallery, details, status = demo.generate(
        USER_PROMPT, images=2, seed_text="1234", size="832x1216", use_llm=True
    )

    assert [caption for _, caption in gallery] == ["seed 1234", "seed 1235"]
    assert all(Path(path).is_file() for path, _ in gallery)
    assert LLM_OUTPUT + STYLE_SUFFIX in details
    assert "1234, 1235" in details
    assert "2 image(s)" in status


def test_generate_uses_the_same_orchestrator_as_the_cli(
    demo: DemoApp, generator: FakeGenerator
) -> None:
    demo.generate(USER_PROMPT, images=1, seed_text="5", size="1024x1024", use_llm=True)

    call = generator.calls[0]
    assert call["prompt"] == LLM_OUTPUT + STYLE_SUFFIX
    assert call["seed"] == 5
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["images"] == 1


def test_blank_seed_means_random(demo: DemoApp, generator: FakeGenerator) -> None:
    demo.generate(USER_PROMPT, images=1, seed_text="", size="832x1216", use_llm=True)

    assert generator.calls[0]["seed"] is None


def test_llm_checkbox_off_skips_ollama(
    demo: DemoApp, enhancer: FakeEnhancer, generator: FakeGenerator
) -> None:
    _, details, status = demo.generate(
        USER_PROMPT, images=1, seed_text="1", size="832x1216", use_llm=False
    )

    assert enhancer.calls == []
    assert generator.calls[0]["prompt"] == USER_PROMPT + STYLE_SUFFIX
    assert "LLM off" in status
    assert USER_PROMPT in details


def test_status_reports_llm_fallback(
    settings: Settings, generator: FakeGenerator
) -> None:
    from animegen.llm.enhancer import EnhancedPrompt

    fallback = EnhancedPrompt(
        original=USER_PROMPT, enhanced=USER_PROMPT, used_fallback=True, duration_s=5.0
    )
    orchestrator = Orchestrator(
        settings=settings,
        enhancer=FakeEnhancer(fallback),  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
    )

    _, _, status = DemoApp(settings=settings, orchestrator=orchestrator).generate(
        USER_PROMPT, seed_text="1"
    )

    assert "LLM unreachable" in status


def test_details_panel_lists_settings_and_files(demo: DemoApp) -> None:
    _, details, _ = demo.generate(USER_PROMPT, images=1, seed_text="1234")

    assert "DreamShaper XL v2 Turbo" in details
    assert "832x1216" in details
    assert "6 steps" in details
    assert "bad anatomy" in details  # negative prompt is shown
    assert "seed1234.png" in details
    assert "seed1234.json" in details


@pytest.mark.parametrize("prompt", ["", "   "])
def test_blank_prompt_raises(demo: DemoApp, prompt: str) -> None:
    with pytest.raises(ValueError, match="Enter a prompt"):
        demo.generate(prompt)


def test_bad_seed_input_raises_before_touching_the_gpu(
    demo: DemoApp, generator: FakeGenerator
) -> None:
    with pytest.raises(ValueError, match="whole number"):
        demo.generate(USER_PROMPT, seed_text="abc")

    assert generator.calls == []


def test_startup_loads_the_pipeline_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = MagicMock(spec=Orchestrator)
    monkeypatch.delenv("ANIMEGEN_UI_WARMUP", raising=False)

    DemoApp(settings=settings, orchestrator=orchestrator).startup()

    orchestrator.generator.load.assert_called_once_with()
    orchestrator.warmup.assert_not_called()


def test_startup_warmup_toggle_honours_the_env_var(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = MagicMock(spec=Orchestrator)
    monkeypatch.setenv("ANIMEGEN_UI_WARMUP", "1")

    DemoApp(settings=settings, orchestrator=orchestrator).startup()

    orchestrator.warmup.assert_called_once_with()


def test_startup_warmup_argument_overrides_the_env_var(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = MagicMock(spec=Orchestrator)
    monkeypatch.setenv("ANIMEGEN_UI_WARMUP", "1")

    DemoApp(settings=settings, orchestrator=orchestrator).startup(warmup=False)

    orchestrator.warmup.assert_not_called()
    orchestrator.generator.load.assert_called_once_with()


def test_startup_survives_a_missing_checkpoint(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    orchestrator = MagicMock(spec=Orchestrator)
    orchestrator.generator.load.side_effect = FileNotFoundError("no checkpoint")

    with caplog.at_level("WARNING"):
        DemoApp(settings=settings, orchestrator=orchestrator).startup()

    assert "will load on first use" in caplog.text


def test_generations_are_serialised_by_a_lock(demo: DemoApp) -> None:
    import threading

    assert isinstance(demo._lock, threading.Lock().__class__)  # noqa: SLF001


def test_build_requires_gradio_and_wires_a_single_worker_queue(
    demo: DemoApp,
) -> None:
    gradio = pytest.importorskip("gradio", reason="gradio is optional for unit tests")

    blocks = demo.build()

    assert isinstance(blocks, gradio.Blocks)
    assert blocks.max_threads >= 1
