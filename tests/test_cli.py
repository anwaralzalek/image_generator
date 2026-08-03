"""Tests for the Typer CLI. The orchestrator is replaced by mocked collaborators."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from animegen.cli import app, parse_size
from animegen.config import DEFAULT_CONFIG_FILE, Settings, load_settings
from animegen.core.orchestrator import Orchestrator
from tests.test_orchestrator import (
    LLM_OUTPUT,
    STYLE_SUFFIX,
    USER_PROMPT,
    FakeEnhancer,
    FakeGenerator,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI styling from CLI output.

    Rich renders help and error text with colour escapes *inside* option names
    ("--images" becomes "--" + escape + "images"), so asserting on raw output
    passes or fails depending on whether the environment enables colour
    (TERM, NO_COLOR, FORCE_COLOR, whether stdout is a tty). Normalising first
    makes these assertions depend on the text and nothing else.
    """
    return _ANSI.sub("", text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(
        DEFAULT_CONFIG_FILE,
        paths={"models_dir": tmp_path / "models", "outputs_dir": tmp_path / "outputs"},
    )

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cli_orchestrator(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> dict[str, Any]:
    """Patch the CLI's Orchestrator with one wired to fakes; record its calls."""
    import animegen.cli as cli_module

    captured: dict[str, Any] = {}

    def factory(settings: Settings | None = None, **_: Any) -> Orchestrator:
        active = settings or load_settings(DEFAULT_CONFIG_FILE)
        generator = FakeGenerator(active)
        enhancer = FakeEnhancer()
        orchestrator = Orchestrator(
            settings=active,
            enhancer=enhancer,  # type: ignore[arg-type]
            generator=generator,  # type: ignore[arg-type]
        )
        captured.update(
            {"orchestrator": orchestrator, "generator": generator, "enhancer": enhancer}
        )
        return orchestrator

    monkeypatch.setattr(cli_module, "Orchestrator", factory)
    return captured


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the CLI's settings at a temporary outputs directory."""
    monkeypatch.setenv("ANIMEGEN_PATHS__OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("ANIMEGEN_PATHS__MODELS_DIR", str(tmp_path / "models"))


def test_cli_help_lists_the_generate_command(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "generate" in plain(result.stdout)


def test_cli_generate_help_lists_every_documented_flag(runner: CliRunner) -> None:
    result = runner.invoke(app, ["generate", "--help"])

    assert result.exit_code == 0
    for flag in ("--images", "--seed", "--size", "--steps", "--no-llm", "--warmup"):
        assert flag in plain(result.stdout)


def test_cli_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "animegen" in plain(result.stdout)


def test_cli_generate_renders_and_reports(
    runner: CliRunner, cli_orchestrator: dict[str, Any], cli_env: None
) -> None:
    result = runner.invoke(app, ["generate", USER_PROMPT, "--seed", "1234", "-n", "2"])

    assert result.exit_code == 0, result.stdout
    generator: FakeGenerator = cli_orchestrator["generator"]
    assert generator.calls[0]["images"] == 2
    assert generator.calls[0]["seed"] == 1234
    assert "seed 1234" in plain(result.stdout)
    assert "seed 1235" in plain(result.stdout)
    assert LLM_OUTPUT in plain(result.stdout)


def test_cli_no_llm_flag_bypasses_ollama(
    runner: CliRunner, cli_orchestrator: dict[str, Any], cli_env: None
) -> None:
    result = runner.invoke(app, ["generate", USER_PROMPT, "--no-llm", "--seed", "5"])

    assert result.exit_code == 0, result.stdout
    enhancer: FakeEnhancer = cli_orchestrator["enhancer"]
    generator: FakeGenerator = cli_orchestrator["generator"]
    assert enhancer.calls == []
    assert generator.calls[0]["prompt"] == USER_PROMPT + STYLE_SUFFIX
    assert "skipped (--no-llm)" in plain(result.stdout)


def test_cli_warmup_flag_runs_a_throwaway_generation(
    runner: CliRunner, cli_orchestrator: dict[str, Any], cli_env: None
) -> None:
    result = runner.invoke(app, ["generate", USER_PROMPT, "--warmup", "--seed", "1"])

    assert result.exit_code == 0, result.stdout
    cli_orchestrator["generator"].warmup.assert_called_once_with()
    assert "Warming up" in plain(result.stdout)


def test_cli_size_flag_is_parsed(
    runner: CliRunner, cli_orchestrator: dict[str, Any], cli_env: None
) -> None:
    result = runner.invoke(
        app, ["generate", USER_PROMPT, "--size", "1024x1024", "--seed", "1"]
    )

    assert result.exit_code == 0, result.stdout
    call = cli_orchestrator["generator"].calls[0]
    assert (call["width"], call["height"]) == (1024, 1024)


def test_cli_rejects_a_malformed_size(
    runner: CliRunner, cli_orchestrator: dict[str, Any], cli_env: None
) -> None:
    result = runner.invoke(app, ["generate", USER_PROMPT, "--size", "big"])

    assert result.exit_code != 0
    assert "WIDTHxHEIGHT" in plain(result.stdout + (result.stderr or ""))


def test_cli_writes_images_where_settings_point(
    runner: CliRunner, cli_orchestrator: dict[str, Any], cli_env: None, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["generate", USER_PROMPT, "--seed", "1234"])

    assert result.exit_code == 0, result.stdout
    images = list((tmp_path / "outputs").glob("*.png"))
    sidecars = list((tmp_path / "outputs").glob("*.json"))
    assert len(images) == 1
    assert len(sidecars) == 1


def test_cli_info_reports_configuration(
    runner: CliRunner, cli_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "animegen.llm.enhancer.OllamaEnhancer.is_available", lambda self: False
    )

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0, result.stdout
    assert "MISSING" in plain(result.stdout)  # no checkpoint in the temp models dir
    assert "DOWN" in plain(result.stdout)


@pytest.mark.parametrize(
    ("value", "expected"), [("832x1216", (832, 1216)), ("1024X1024", (1024, 1024))]
)
def test_parse_size_accepts_valid_sizes(
    settings: Settings, value: str, expected: tuple[int, int]
) -> None:
    assert parse_size(value, settings) == expected


@pytest.mark.parametrize("value", ["832", "832x", "axb", "833x1216", "0x0", "-8x8"])
def test_parse_size_rejects_invalid_sizes(settings: Settings, value: str) -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        parse_size(value, settings)
