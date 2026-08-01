"""Static Docker/Compose contract tests; no daemon is required."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def gpu_reservations(service: dict[str, Any]) -> list[dict[str, Any]]:
    resources = service.get("deploy", {}).get("resources", {})
    return resources.get("reservations", {}).get("devices", [])


def test_container_files_are_minimal() -> None:
    for path in (COMPOSE_FILE, DOCKERFILE, DOCKERIGNORE):
        assert path.is_file(), f"{path} is missing"

    assert not (PROJECT_ROOT / "docker" / "entrypoint.sh").exists()
    assert not (PROJECT_ROOT / "config" / "settings.yaml").exists()


def test_compose_declares_the_three_services(compose: dict[str, Any]) -> None:
    assert set(compose["services"]) == {"ollama", "ollama-init", "animegen"}


def test_ollama_images_are_pinned_and_cpu_only(compose: dict[str, Any]) -> None:
    for name in ("ollama", "ollama-init"):
        service = compose["services"][name]
        assert service["image"] == "ollama/ollama:0.32.1"
        assert gpu_reservations(service) == []


def test_init_job_pulls_the_expected_model_once(compose: dict[str, Any]) -> None:
    init = compose["services"]["ollama-init"]

    assert init["entrypoint"] == ["ollama", "pull", "llama3.2:3b"]
    assert init["environment"]["OLLAMA_HOST"] == "http://ollama:11434"
    assert init["restart"] == "no"


def test_animegen_does_not_wait_for_optional_ollama(
    compose: dict[str, Any],
) -> None:
    assert "depends_on" not in compose["services"]["animegen"]


def test_animegen_reserves_the_gpu(compose: dict[str, Any]) -> None:
    devices = gpu_reservations(compose["services"]["animegen"])

    assert devices, "the image generator must request the NVIDIA device"
    assert devices[0]["driver"] == "nvidia"
    assert devices[0]["capabilities"] == ["gpu"]


def test_app_points_at_bundled_ollama_without_requiring_it(
    compose: dict[str, Any],
) -> None:
    service = compose["services"]["animegen"]
    env = service["environment"]
    assert env["ANIMEGEN_OLLAMA__HOST"] == (
        "${ANIMEGEN_OLLAMA__HOST:-http://ollama:11434}"
    )
    assert "host.docker.internal:host-gateway" in service["extra_hosts"]


def test_only_outputs_and_hf_cache_are_mounted_for_animegen(
    compose: dict[str, Any],
) -> None:
    service = compose["services"]["animegen"]
    volumes = service["volumes"]

    assert "./outputs:/outputs" in volumes
    assert "hf-cache:/hf-cache" in volumes
    assert not any("models" in volume for volume in volumes)
    assert service["environment"]["ANIMEGEN_PATHS__OUTPUTS_DIR"] == "/outputs"
    assert service["environment"]["HF_HOME"] == "/hf-cache"
    assert set(compose["volumes"]) == {"ollama-models", "hf-cache"}


def test_ollama_cache_survives_restarts(compose: dict[str, Any]) -> None:
    assert "ollama-models:/root/.ollama" in compose["services"]["ollama"]["volumes"]


def test_ui_settings_are_wired_through_compose(compose: dict[str, Any]) -> None:
    service = compose["services"]["animegen"]

    assert service["ports"] == ["127.0.0.1:${HOST_PORT:-7860}:7860"]


def test_dockerfile_uses_exact_pytorch_cuda_runtime_and_installs_app(
    dockerfile: str,
) -> None:
    assert dockerfile.count(
        "FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime"
    ) == 1
    assert "FROM python:" not in dockerfile
    assert "TORCH_INDEX_URL" not in dockerfile
    assert "pip install torch" not in dockerfile
    assert "pip install --upgrade pip" not in dockerfile
    assert 'python -m pip install --break-system-packages ".[ui]"' in dockerfile
    assert "COPY pyproject.toml README.md ./" in dockerfile
    assert "--mount=type=cache" not in dockerfile
    assert "PIP_NO_CACHE_DIR=1" in dockerfile
    assert '".[dev]"' not in dockerfile
    assert "COPY config" not in dockerfile


def test_dockerfile_reuses_the_base_image_nonroot_user(dockerfile: str) -> None:
    for fragment in ("VIRTUAL_ENV", "python -m venv", "/opt/venv"):
        assert fragment not in dockerfile

    assert "useradd" not in dockerfile
    assert "chown 1000:1000 /outputs /hf-cache" in dockerfile
    assert "USER 1000:1000" in dockerfile


def test_dockerfile_uses_python_healthcheck_without_curl(dockerfile: str) -> None:
    assert "HEALTHCHECK" in dockerfile
    assert "urllib.request.urlopen" in dockerfile
    assert "curl" not in dockerfile


def test_dockerfile_invokes_animegen_directly(dockerfile: str) -> None:
    assert 'ENTRYPOINT ["animegen"]' in dockerfile
    assert 'CMD ["ui", "--host", "0.0.0.0"]' in dockerfile
    assert "EXPOSE 7860" in dockerfile


def test_dockerfile_declares_nvidia_capabilities(dockerfile: str) -> None:
    assert "NVIDIA_VISIBLE_DEVICES=all" in dockerfile
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in dockerfile


def test_dockerignore_keeps_runtime_state_out_of_the_image() -> None:
    ignored = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    for pattern in (
        "models/",
        "outputs/",
        "*.safetensors",
        ".git/",
        ".agents/",
        ".codex/",
        ".env",
    ):
        assert pattern in ignored, f"{pattern} must not enter the build context"


def test_pyyaml_is_a_test_dependency_not_a_runtime_dependency() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8").lower()

    assert 'dev = ["pytest>=8,<10", "pyyaml>=6,<7"]' in pyproject
    assert "pyyaml" not in DOCKERFILE.read_text(encoding="utf-8").lower()


def test_env_example_documents_compose_variables() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for variable in ("HOST_PORT", "ANIMEGEN_OLLAMA__HOST", "HF_TOKEN"):
        assert variable in text
