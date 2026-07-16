"""Tests for the SDXL generator. torch and diffusers are fully mocked: no GPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from animegen.config import DEFAULT_CONFIG_FILE, Settings, load_settings
from animegen.sd import pipeline as pipeline_module
from animegen.sd.pipeline import SEED_MAX, GenerationResult, SDXLGenerator

PROMPT = "six teenagers dancing, string lights, 2.5D anime style"


class FakeTorchGenerator:
    """Stand-in for ``torch.Generator``."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> "FakeTorchGenerator":
        self.seed = seed
        return self


def make_fake_torch(cuda_available: bool = True) -> SimpleNamespace:
    """Build a minimal torch stub that records generator seeds and cache flushes."""
    created: list[FakeTorchGenerator] = []

    def generator_factory(device: str = "cpu") -> FakeTorchGenerator:
        instance = FakeTorchGenerator(device)
        created.append(instance)
        return instance

    return SimpleNamespace(
        float16="dtype:float16",
        float32="dtype:float32",
        bfloat16="dtype:bfloat16",
        Generator=generator_factory,
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            empty_cache=MagicMock(name="empty_cache"),
        ),
        generators=created,
    )


class FakePipe:
    """Stand-in for ``StableDiffusionXLPipeline``."""

    def __init__(self) -> None:
        self.scheduler: Any = SimpleNamespace(config={"stock": "scheduler-config"})
        self.enable_model_cpu_offload = MagicMock(name="enable_model_cpu_offload")
        self.enable_vae_tiling = MagicMock(name="enable_vae_tiling")
        self.set_progress_bar_config = MagicMock(name="set_progress_bar_config")
        self.to = MagicMock(name="to")
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(images=[f"image-{len(self.calls)}"])


def make_fake_diffusers(pipe: FakePipe) -> SimpleNamespace:
    """Build a diffusers stub exposing only what the generator touches."""
    return SimpleNamespace(
        AutoencoderKL=SimpleNamespace(
            from_pretrained=MagicMock(return_value="fp16-fix-vae")
        ),
        StableDiffusionXLPipeline=SimpleNamespace(
            from_single_file=MagicMock(return_value=pipe)
        ),
        DPMSolverSinglestepScheduler=SimpleNamespace(
            from_config=MagicMock(return_value="karras-scheduler")
        ),
        DPMSolverSDEScheduler=SimpleNamespace(
            from_config=MagicMock(return_value="sde-scheduler")
        ),
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A stand-in checkpoint file so load() passes its existence check."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    file = models_dir / "DreamShaperXL_v2_Turbo.safetensors"
    file.write_bytes(b"not-a-real-checkpoint")
    return file


@pytest.fixture
def settings(checkpoint: Path) -> Settings:
    return load_settings(
        DEFAULT_CONFIG_FILE, paths={"models_dir": checkpoint.parent}
    )


@pytest.fixture
def fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Patch the lazy torch/diffusers importers with stubs."""
    pipe = FakePipe()
    torch = make_fake_torch()
    diffusers = make_fake_diffusers(pipe)
    monkeypatch.setattr(pipeline_module, "_torch", lambda: torch)
    monkeypatch.setattr(pipeline_module, "_diffusers", lambda: diffusers)
    return SimpleNamespace(pipe=pipe, torch=torch, diffusers=diffusers)


@pytest.fixture
def generator(settings: Settings, fakes: SimpleNamespace) -> SDXLGenerator:
    return SDXLGenerator(settings=settings)


def test_load_uses_single_file_checkpoint_in_fp16(
    generator: SDXLGenerator, fakes: SimpleNamespace, checkpoint: Path
) -> None:
    generator.load()

    call = fakes.diffusers.StableDiffusionXLPipeline.from_single_file.call_args
    assert call.args[0] == str(checkpoint)
    assert call.kwargs["torch_dtype"] == fakes.torch.float16
    assert call.kwargs["use_safetensors"] is True
    assert call.kwargs["vae"] == "fp16-fix-vae"


def test_load_swaps_in_the_fp16_fix_vae(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    fakes.diffusers.AutoencoderKL.from_pretrained.assert_called_once_with(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=fakes.torch.float16
    )


def test_load_enables_cpu_offload_and_never_moves_to_cuda(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    fakes.pipe.enable_model_cpu_offload.assert_called_once_with()
    fakes.pipe.to.assert_not_called()
    assert generator.device == "cuda"


def test_load_enables_vae_tiling(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    fakes.pipe.enable_vae_tiling.assert_called_once_with()


def test_load_installs_karras_scheduler_from_checkpoint_config(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    fakes.diffusers.DPMSolverSinglestepScheduler.from_config.assert_called_once_with(
        {"stock": "scheduler-config"}, use_karras_sigmas=True
    )
    assert fakes.pipe.scheduler == "karras-scheduler"


def test_configured_scheduler_is_honoured(
    checkpoint: Path, fakes: SimpleNamespace
) -> None:
    settings = load_settings(
        DEFAULT_CONFIG_FILE,
        paths={"models_dir": checkpoint.parent},
        generation={"scheduler": "DPMSolverSDEScheduler"},
    )

    SDXLGenerator(settings=settings).load()

    fakes.diffusers.DPMSolverSDEScheduler.from_config.assert_called_once()
    assert fakes.pipe.scheduler == "sde-scheduler"


def test_unknown_scheduler_raises(checkpoint: Path, fakes: SimpleNamespace) -> None:
    settings = load_settings(
        DEFAULT_CONFIG_FILE,
        paths={"models_dir": checkpoint.parent},
        generation={"scheduler": "NoSuchScheduler"},
    )

    with pytest.raises(ValueError, match="Unknown scheduler"):
        SDXLGenerator(settings=settings).load()


def test_missing_checkpoint_points_at_the_download_script(
    tmp_path: Path, fakes: SimpleNamespace
) -> None:
    settings = load_settings(DEFAULT_CONFIG_FILE, paths={"models_dir": tmp_path})

    with pytest.raises(FileNotFoundError, match="download_models.py"):
        SDXLGenerator(settings=settings).load()


def test_load_is_idempotent(generator: SDXLGenerator, fakes: SimpleNamespace) -> None:
    generator.load()
    generator.load()

    assert fakes.diffusers.StableDiffusionXLPipeline.from_single_file.call_count == 1
    assert generator.is_loaded is True


def test_cpu_only_machine_falls_back_to_fp32_without_offload(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = FakePipe()
    torch = make_fake_torch(cuda_available=False)
    diffusers = make_fake_diffusers(pipe)
    monkeypatch.setattr(pipeline_module, "_torch", lambda: torch)
    monkeypatch.setattr(pipeline_module, "_diffusers", lambda: diffusers)

    gen = SDXLGenerator(settings=settings)
    gen.load()

    assert gen.device == "cpu"
    pipe.enable_model_cpu_offload.assert_not_called()
    pipe.to.assert_called_once_with("cpu")
    call = diffusers.StableDiffusionXLPipeline.from_single_file.call_args
    assert call.kwargs["torch_dtype"] == torch.float32


def test_generate_returns_one_result_per_image(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=1234, images=3)

    assert [r.image for r in results] == ["image-1", "image-2", "image-3"]
    assert all(isinstance(r, GenerationResult) for r in results)
    assert len(fakes.pipe.calls) == 3, "images must be rendered one at a time"


def test_seeds_are_deterministic_and_sequential(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=1234, images=3)

    assert [r.seed for r in results] == [1234, 1235, 1236]
    assert [g.seed for g in fakes.torch.generators] == [1234, 1235, 1236]
    assert {g.device for g in fakes.torch.generators} == {"cpu"}


def test_random_seed_is_drawn_and_recorded(
    generator: SDXLGenerator, fakes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module.random, "randint", lambda a, b: 999)

    results = generator.generate(PROMPT, images=2)

    assert [r.seed for r in results] == [999, 1000]
    assert fakes.torch.generators[0].seed == 999


def test_seed_wraps_at_the_seed_space_boundary(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=SEED_MAX, images=2)

    assert [r.seed for r in results] == [SEED_MAX, 0]


def test_generate_uses_configured_defaults(
    generator: SDXLGenerator, fakes: SimpleNamespace, settings: Settings
) -> None:
    generator.generate(PROMPT, seed=1)

    call = fakes.pipe.calls[0]
    assert call["prompt"] == PROMPT
    assert call["negative_prompt"] == settings.style.negative_prompt
    assert call["num_inference_steps"] == 6
    assert call["guidance_scale"] == pytest.approx(2.0)
    assert (call["width"], call["height"]) == (832, 1216)


def test_explicit_overrides_reach_the_pipeline(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.generate(
        PROMPT,
        negative_prompt="custom negative",
        seed=7,
        width=1024,
        height=1024,
        steps=4,
        guidance=1.5,
    )

    call = fakes.pipe.calls[0]
    assert call["negative_prompt"] == "custom negative"
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == pytest.approx(1.5)


@pytest.mark.parametrize(("requested", "expected"), [(1, 4), (2, 4), (6, 6), (12, 8)])
def test_steps_are_clamped_to_the_turbo_range(
    generator: SDXLGenerator, fakes: SimpleNamespace, requested: int, expected: int
) -> None:
    generator.generate(PROMPT, seed=1, steps=requested)

    assert fakes.pipe.calls[0]["num_inference_steps"] == expected


def test_image_count_is_capped_for_the_vram_budget(
    generator: SDXLGenerator, fakes: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        results = generator.generate(PROMPT, seed=1, images=9)

    assert len(results) == 4
    assert "capping at 4" in caplog.text


def test_result_settings_capture_reproduction_details(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    result = generator.generate(PROMPT, seed=42, steps=6)[0]

    assert result.settings == {
        "prompt": PROMPT,
        "negative_prompt": (
            "bad anatomy, deformed hands, extra fingers, extra limbs, mutated, "
            "lowres, blurry, watermark, text, jpeg artifacts, flat 2D shading"
        ),
        "width": 832,
        "height": 1216,
        "steps": 6,
        "guidance_scale": 2.0,
        "scheduler": "DPMSolverSinglestepScheduler",
        "use_karras_sigmas": True,
        "model": "DreamShaper XL v2 Turbo",
        "checkpoint": "DreamShaperXL_v2_Turbo.safetensors",
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "dtype": "float16",
        "device": "cuda",
    }
    assert result.duration_s >= 0.0


def test_generate_loads_the_pipeline_on_demand(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    assert generator.is_loaded is False

    generator.generate(PROMPT, seed=1)

    assert generator.is_loaded is True


def test_cuda_cache_is_emptied_after_each_run(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.generate(PROMPT, seed=1, images=2)

    fakes.torch.cuda.empty_cache.assert_called_once_with()


def test_cache_emptying_can_be_disabled(
    checkpoint: Path, fakes: SimpleNamespace
) -> None:
    settings = load_settings(
        DEFAULT_CONFIG_FILE,
        paths={"models_dir": checkpoint.parent},
        vram={"empty_cache_after_run": False},
    )

    SDXLGenerator(settings=settings).generate(PROMPT, seed=1)

    fakes.torch.cuda.empty_cache.assert_not_called()


def test_unload_releases_the_pipeline_and_vram(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    generator.unload()

    assert generator.is_loaded is False
    assert generator.device is None
    fakes.torch.cuda.empty_cache.assert_called_once_with()
    generator.unload()  # no-op, must not raise


def test_context_manager_loads_and_unloads(
    settings: Settings, fakes: SimpleNamespace
) -> None:
    with SDXLGenerator(settings=settings) as gen:
        assert gen.is_loaded is True

    assert gen.is_loaded is False


def test_warmup_renders_a_tiny_throwaway_image(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.warmup()

    call = fakes.pipe.calls[0]
    assert (call["width"], call["height"]) == (512, 512)
    assert call["num_inference_steps"] == 4, "clamped up to the Turbo minimum"
    assert generator.is_loaded is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": "   "}, "prompt must not be empty"),
        ({"prompt": PROMPT, "images": 0}, "images must be >= 1"),
    ],
)
def test_invalid_arguments_raise(
    generator: SDXLGenerator,
    fakes: SimpleNamespace,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generator.generate(**kwargs)
