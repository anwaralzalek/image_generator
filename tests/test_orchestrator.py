"""Tests for the orchestrator. Enhancer and generator are mocked: no GPU, no Ollama."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from animegen.config import DEFAULT_CONFIG_FILE, Settings, load_settings
from animegen.core.orchestrator import Orchestrator, RunResult
from animegen.llm.enhancer import EnhancedPrompt
from animegen.sd.pipeline import GenerationResult

USER_PROMPT = "American teenagers having fun at a party"
LLM_OUTPUT = "six american teenagers dancing, house party, string lights, low angle"
STYLE_SUFFIX = (
    ", semi-realistic 2.5D anime style, 3D-shaded characters, volumetric lighting,"
    " glossy rendering, detailed faces, cinematic composition, high detail"
)


class FakeImage:
    """Stand-in for a PIL image that records where it was saved."""

    def __init__(self, name: str = "img") -> None:
        self.name = name
        self.saved_to: Path | None = None

    def save(self, path: Path) -> None:
        self.saved_to = Path(path)
        Path(path).write_bytes(b"\x89PNG-fake")


class FakeGenerator:
    """Stand-in for SDXLGenerator recording calls and replaying results."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.calls: list[dict[str, Any]] = []
        self.warmup = MagicMock(name="warmup")
        self.unload = MagicMock(name="unload")

    def generate(self, **kwargs: Any) -> list[GenerationResult]:
        self.calls.append(kwargs)
        model_key, profile = self._settings.image_model(kwargs.get("model"))
        count = kwargs.get("images", 1)
        base_seed = kwargs.get("seed")
        base_seed = 555 if base_seed is None else base_seed
        return [
            GenerationResult(
                image=FakeImage(f"img{index}"),
                seed=base_seed + index,
                settings={
                    "prompt": kwargs["prompt"],
                    "negative_prompt": (
                        profile.negative_prompt
                        or self._settings.style.negative_prompt
                    ),
                    "width": kwargs.get("width") or profile.width,
                    "height": kwargs.get("height") or profile.height,
                    "steps": kwargs.get("steps") or profile.steps,
                    "guidance_scale": (
                        kwargs.get("guidance") or profile.guidance_scale
                    ),
                    "scheduler": profile.scheduler,
                    "model_key": model_key,
                    "model": profile.name,
                    "weight_dtype": self._settings.vram.weight_dtype,
                    "estimated_render_time": profile.estimate,
                },
                duration_s=1.5,
            )
            for index in range(count)
        ]


class FakeEnhancer:
    """Stand-in for OllamaEnhancer returning a scripted EnhancedPrompt."""

    def __init__(self, result: EnhancedPrompt | None = None) -> None:
        self.result = result
        self.calls: list[str] = []

    def enhance(self, prompt: str) -> EnhancedPrompt:
        self.calls.append(prompt)
        if self.result is not None:
            return self.result
        return EnhancedPrompt(
            original=prompt,
            enhanced=LLM_OUTPUT,
            used_fallback=False,
            duration_s=0.4,
            model="llama3.2:3b",
        )

    def is_available(self) -> bool:
        return True


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
def orchestrator(
    settings: Settings, enhancer: FakeEnhancer, generator: FakeGenerator
) -> Orchestrator:
    return Orchestrator(
        settings=settings,
        enhancer=enhancer,  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
    )


def read_sidecar(image_path: Path) -> dict[str, Any]:
    return json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))


def test_style_suffix_is_appended_to_the_enhanced_prompt(
    orchestrator: Orchestrator, generator: FakeGenerator
) -> None:
    result = orchestrator.run(USER_PROMPT, seed=1234)

    assert result.enhanced_prompt == LLM_OUTPUT + STYLE_SUFFIX
    assert generator.calls[0]["prompt"] == LLM_OUTPUT + STYLE_SUFFIX


def test_style_suffix_is_not_duplicated(orchestrator: Orchestrator) -> None:
    already_styled = LLM_OUTPUT + STYLE_SUFFIX

    assert orchestrator.apply_style(already_styled) == already_styled


def test_style_application_avoids_double_commas(orchestrator: Orchestrator) -> None:
    assert orchestrator.apply_style("teenagers dancing,") == (
        "teenagers dancing" + STYLE_SUFFIX
    )


def test_filenames_follow_the_timestamp_seed_convention(
    orchestrator: Orchestrator, settings: Settings
) -> None:
    result = orchestrator.run(USER_PROMPT, seed=1234, images=2)

    assert [path.name for path in result.paths] == [
        path.name for path in sorted(settings.outputs_dir.glob("*.png"))
    ]
    for image in result.images:
        assert re.fullmatch(r"\d{8}_\d{6}_seed\d+\.png", image.path.name), image.path
        assert image.path.parent == settings.outputs_dir
        assert image.path.name.endswith(f"seed{image.seed}.png")
        assert image.path.is_file()


def test_each_image_gets_a_sidecar(orchestrator: Orchestrator) -> None:
    result = orchestrator.run(USER_PROMPT, seed=1234, images=3)

    for image in result.images:
        assert image.metadata_path == image.path.with_suffix(".json")
        assert image.metadata_path.is_file()


def test_sidecar_records_prompts_settings_and_timings(
    orchestrator: Orchestrator,
) -> None:
    result = orchestrator.run(USER_PROMPT, seed=1234, steps=8, guidance=3.0)
    metadata = read_sidecar(result.paths[0])

    assert metadata["original_prompt"] == USER_PROMPT
    assert metadata["enhanced_prompt"] == LLM_OUTPUT
    assert metadata["final_prompt"] == LLM_OUTPUT + STYLE_SUFFIX
    assert metadata["style_suffix"] == STYLE_SUFFIX
    assert metadata["used_fallback"] is False
    assert metadata["llm_enabled"] is True
    assert metadata["llm_model"] == "llama3.2:3b"
    assert metadata["seed"] == 1234
    assert metadata["index"] == 0
    assert metadata["settings"]["steps"] == 8
    assert metadata["settings"]["guidance_scale"] == pytest.approx(3.0)
    assert metadata["settings"]["model"] == "DreamShaper XL v2 Turbo"
    assert metadata["timings"]["llm_s"] == pytest.approx(0.4)
    assert metadata["timings"]["render_s"] == pytest.approx(1.5)
    assert metadata["app_version"]
    assert metadata["created_at"]


def test_sidecar_is_valid_json_for_every_image(orchestrator: Orchestrator) -> None:
    result = orchestrator.run(USER_PROMPT, seed=7, images=2)

    seeds = [read_sidecar(path)["seed"] for path in result.paths]
    indices = [read_sidecar(path)["index"] for path in result.paths]

    assert seeds == [7, 8]
    assert indices == [0, 1]


def test_fallback_flag_is_recorded(settings: Settings, generator: FakeGenerator) -> None:
    fallback = EnhancedPrompt(
        original=USER_PROMPT,
        enhanced=USER_PROMPT,
        used_fallback=True,
        duration_s=5.0,
        error="ConnectionError: refused",
    )
    orchestrator = Orchestrator(
        settings=settings,
        enhancer=FakeEnhancer(fallback),  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
    )

    result = orchestrator.run(USER_PROMPT, seed=1)
    metadata = read_sidecar(result.paths[0])

    assert result.used_fallback is True
    assert result.enhanced_prompt == USER_PROMPT + STYLE_SUFFIX
    assert metadata["used_fallback"] is True
    assert metadata["llm_error"] == "ConnectionError: refused"
    assert metadata["llm_model"] is None


def test_no_llm_path_skips_the_enhancer_entirely(
    orchestrator: Orchestrator, enhancer: FakeEnhancer, generator: FakeGenerator
) -> None:
    result = orchestrator.run(USER_PROMPT, seed=1, use_llm=False)
    metadata = read_sidecar(result.paths[0])

    assert enhancer.calls == [], "Ollama must not be contacted with use_llm=False"
    assert result.llm_enabled is False
    assert result.used_fallback is False
    assert result.enhanced_prompt == USER_PROMPT + STYLE_SUFFIX
    assert generator.calls[0]["prompt"] == USER_PROMPT + STYLE_SUFFIX
    assert metadata["llm_enabled"] is False
    assert metadata["llm_model"] is None
    assert metadata["llm_duration_s"] == 0.0


def test_generation_arguments_are_forwarded(
    orchestrator: Orchestrator, generator: FakeGenerator
) -> None:
    orchestrator.run(
        USER_PROMPT, images=3, seed=99, width=1024, height=1024, steps=4, guidance=1.5
    )

    call = generator.calls[0]
    assert call["images"] == 3
    assert call["seed"] == 99
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["steps"] == 4
    assert call["guidance"] == pytest.approx(1.5)


def test_model_selection_is_forwarded(
    orchestrator: Orchestrator, generator: FakeGenerator
) -> None:
    result = orchestrator.run(USER_PROMPT, seed=1, model="best")

    assert generator.calls[0]["model"] == "best"
    assert "masterpiece, high score, great score, absurdres" in result.enhanced_prompt
    assert result.images[0].metadata["settings"]["model"] == "Animagine XL 4.0 Opt"


def test_run_result_exposes_seeds_and_timings(orchestrator: Orchestrator) -> None:
    result = orchestrator.run(USER_PROMPT, seed=1234, images=2)

    assert isinstance(result, RunResult)
    assert result.seeds == [1234, 1235]
    assert result.llm_duration_s == pytest.approx(0.4)
    assert result.sd_duration_s >= 0.0
    assert result.total_duration_s >= result.sd_duration_s


def test_outputs_directory_is_created_on_demand(
    orchestrator: Orchestrator, settings: Settings
) -> None:
    assert not settings.outputs_dir.exists()

    orchestrator.run(USER_PROMPT, seed=1)

    assert settings.outputs_dir.is_dir()


def test_existing_filename_is_not_overwritten(
    orchestrator: Orchestrator, settings: Settings
) -> None:
    first = orchestrator.run(USER_PROMPT, seed=1234).paths[0]
    second = orchestrator.run(USER_PROMPT, seed=1234).paths[0]

    assert first != second
    assert first.is_file() and second.is_file()
    assert second.stem.endswith("_2")


def test_blank_prompt_is_rejected(orchestrator: Orchestrator) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        orchestrator.run("   ")


def test_warmup_and_close_delegate_to_the_generator(
    orchestrator: Orchestrator, generator: FakeGenerator
) -> None:
    orchestrator.warmup()
    orchestrator.close()

    generator.warmup.assert_called_once_with(model=None)
    generator.unload.assert_called_once_with()
