"""Contract tests for the container setup. No Docker daemon required.

These guard the decisions that are invisible until they fail on demo day: the
LLM must not be given a GPU, weights must stay out of the image, and the app
must be pointed at the bundled Ollama rather than localhost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from animegen.config import PROJECT_ROOT

COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
ENTRYPOINT = PROJECT_ROOT / "docker" / "entrypoint.sh"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def gpu_reservations(service: dict[str, Any]) -> list[dict[str, Any]]:
    """The GPU device reservations of a compose service, if any."""
    resources = service.get("deploy", {}).get("resources", {})
    return resources.get("reservations", {}).get("devices", [])


def test_container_files_exist() -> None:
    for path in (COMPOSE_FILE, DOCKERFILE, DOCKERIGNORE, ENTRYPOINT):
        assert path.is_file(), f"{path} is missing"


def test_compose_declares_the_three_services(compose: dict[str, Any]) -> None:
    assert set(compose["services"]) == {"ollama", "ollama-init", "animegen"}


def test_ollama_is_never_given_a_gpu(compose: dict[str, Any]) -> None:
    # An LLM holding VRAM is the fastest way to OOM SDXL on an 8 GB card.
    assert gpu_reservations(compose["services"]["ollama"]) == []
    assert gpu_reservations(compose["services"]["ollama-init"]) == []


def test_animegen_reserves_the_gpu(compose: dict[str, Any]) -> None:
    devices = gpu_reservations(compose["services"]["animegen"])

    assert devices, "the generator must request the NVIDIA device"
    assert devices[0]["driver"] == "nvidia"
    assert devices[0]["capabilities"] == ["gpu"]


def test_app_points_at_the_bundled_ollama(compose: dict[str, Any]) -> None:
    env = compose["services"]["animegen"]["environment"]

    assert env["ANIMEGEN_OLLAMA__HOST"] == "http://ollama:11434"


def test_app_waits_for_the_model_pull_to_finish(compose: dict[str, Any]) -> None:
    depends = compose["services"]["animegen"]["depends_on"]

    assert depends["ollama-init"]["condition"] == "service_completed_successfully"


def test_init_job_pulls_the_configured_model(compose: dict[str, Any]) -> None:
    init = compose["services"]["ollama-init"]

    assert init["entrypoint"] == ["ollama", "pull", "llama3.2:3b"]
    assert init["environment"]["OLLAMA_HOST"] == "http://ollama:11434"
    assert init["restart"] == "no", "the pull must run once, not on a loop"


def test_ollama_model_cache_survives_restarts(compose: dict[str, Any]) -> None:
    assert "ollama-models:/root/.ollama" in compose["services"]["ollama"]["volumes"]
    assert "ollama-models" in compose["volumes"]


def test_weights_and_outputs_are_mounted_not_baked(compose: dict[str, Any]) -> None:
    service = compose["services"]["animegen"]

    assert "./models:/models" in service["volumes"]
    assert "./outputs:/outputs" in service["volumes"]
    assert service["environment"]["ANIMEGEN_PATHS__MODELS_DIR"] == "/models"
    assert service["environment"]["ANIMEGEN_PATHS__OUTPUTS_DIR"] == "/outputs"


def test_hf_cache_is_a_named_volume(compose: dict[str, Any]) -> None:
    service = compose["services"]["animegen"]

    assert "hf-cache:/hf-cache" in service["volumes"]
    assert service["environment"]["HF_HOME"] == "/hf-cache"
    assert "hf-cache" in compose["volumes"]


def test_ui_port_is_published(compose: dict[str, Any]) -> None:
    assert compose["services"]["animegen"]["ports"] == ["${HOST_PORT:-7860}:7860"]


def test_warmup_toggle_is_wired_through_compose(compose: dict[str, Any]) -> None:
    env = compose["services"]["animegen"]["environment"]

    assert env["ANIMEGEN_UI_WARMUP"] == "${ANIMEGEN_UI_WARMUP:-0}"


def test_dockerfile_installs_cuda_torch(dockerfile: str) -> None:
    # The default PyPI wheel is CPU-only: it would "work" and be 100x too slow.
    assert "download.pytorch.org/whl/cu121" in dockerfile
    assert "pip install torch --index-url" in dockerfile


def test_dockerfile_uses_system_python_and_runs_unprivileged(
    dockerfile: str,
) -> None:
    assert "AS runtime" in dockerfile
    assert "VIRTUAL_ENV" not in dockerfile
    assert "python -m venv" not in dockerfile
    assert "/opt/venv" not in dockerfile
    assert 'pip install ".[dev]"' in dockerfile
    assert "USER appuser" in dockerfile


def test_dockerfile_exposes_the_ui_and_binds_all_interfaces(dockerfile: str) -> None:
    assert "EXPOSE 7860" in dockerfile
    assert 'CMD ["ui", "--host", "0.0.0.0"]' in dockerfile


def test_dockerfile_declares_nvidia_capabilities(dockerfile: str) -> None:
    assert "NVIDIA_VISIBLE_DEVICES=all" in dockerfile
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in dockerfile


def test_dockerignore_keeps_weights_out_of_the_build_context() -> None:
    ignored = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    for pattern in ("models/", "outputs/", "*.safetensors", ".git/", ".env"):
        assert pattern in ignored, f"{pattern} must not enter the build context"


def test_entrypoint_dispatches_cli_and_arbitrary_commands() -> None:
    script = ENTRYPOINT.read_text(encoding="utf-8")

    assert "exec animegen" in script
    assert 'exec "$@"' in script
    # Subcommands and global flags (--verbose info) go to the CLI; anything
    # else (pytest, python) runs as its own command.
    assert "generate | ui | info | -*)" in script


def test_env_example_documents_the_compose_variables() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for variable in ("ANIMEGEN_UI_WARMUP", "HOST_PORT", "ANIMEGEN_OLLAMA__HOST"):
        assert variable in text
