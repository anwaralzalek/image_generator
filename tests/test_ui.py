"""Tests for the Gradio demo. The handler logic is tested without gradio installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from animegen.config import SEED_MAX, Settings
from animegen.core.orchestrator import Orchestrator
from animegen.ui.app import DemoApp, parse_seed, parse_size
from tests.test_orchestrator import (
    LLM_OUTPUT,
    STYLE_SUFFIX,
    USER_PROMPT,
    FakeEnhancer,
    FakeGenerator,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(outputs_dir=tmp_path / "outputs")


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
def test_parse_seed_accepts_blank_and_numbers(value: Any, expected: int | None) -> None:
    assert parse_seed(value) == expected


@pytest.mark.parametrize("value", ["abc", "12.5", "-1", "1e5", -1, SEED_MAX + 1, True])
def test_parse_seed_rejects_junk(value: Any) -> None:
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


def test_model_selection_uses_the_profile_defaults(
    demo: DemoApp, generator: FakeGenerator
) -> None:
    _, details, _ = demo.generate(USER_PROMPT, images=1, seed_text="5", model="fast")

    call = generator.calls[0]
    assert call["model"] == "fast"
    assert (call["width"], call["height"]) == (768, 832)
    assert "Eimis Anime Diffusion 1.0v" in details
    assert "10-25 seconds" in details


def test_model_selection_updates_the_recommended_size(demo: DemoApp) -> None:
    description, choices, size = demo._model_selection("best")  # noqa: SLF001

    assert "Animagine XL 4.0 Opt" in description
    assert "INT8" in description
    assert choices == ("832x1216", "1024x1024")
    assert size == "832x1216"


def test_size_must_belong_to_selected_model(
    demo: DemoApp, generator: FakeGenerator
) -> None:
    with pytest.raises(ValueError, match="supports"):
        demo.generate(USER_PROMPT, model="fast", size="1024x1024")

    assert generator.calls == []


def test_image_count_must_be_a_whole_number(
    demo: DemoApp, generator: FakeGenerator
) -> None:
    with pytest.raises(ValueError, match="whole number"):
        demo.generate(USER_PROMPT, images=1.5)  # type: ignore[arg-type]

    assert generator.calls == []


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
    assert "seed1234_" in details
    assert ".png" in details and ".json" in details


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


def test_build_requires_gradio_and_wires_a_single_worker_queue(
    demo: DemoApp,
) -> None:
    gradio = pytest.importorskip("gradio", reason="gradio is optional for unit tests")

    blocks = demo.build()

    assert isinstance(blocks, gradio.Blocks)
    assert blocks.max_threads >= 1
