"""Unit tests for the lazy image pipeline with INT8 linear weights."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from animegen.config import SEED_MAX, Settings
from animegen.sd import pipeline as pipeline_module
from animegen.sd.pipeline import GenerationResult, ImageGenerator

PROMPT = "six teenagers dancing under string lights"


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeTorchGenerator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> FakeTorchGenerator:
        self.seed = seed
        return self


def make_fake_torch(cuda_available: bool = True) -> SimpleNamespace:
    created: list[FakeTorchGenerator] = []

    def generator_factory(device: str = "cpu") -> FakeTorchGenerator:
        generator = FakeTorchGenerator(device)
        created.append(generator)
        return generator

    return SimpleNamespace(
        float16="dtype:float16",
        Generator=generator_factory,
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeOutOfMemoryError,
            is_available=lambda: cuda_available,
            empty_cache=MagicMock(name="empty_cache"),
        ),
        generators=created,
    )


class FakePipe:
    def __init__(self) -> None:
        self.scheduler: Any = SimpleNamespace(config={"stock": "scheduler-config"})
        self.vae = SimpleNamespace(enable_tiling=MagicMock(name="enable_tiling"))
        self.enable_model_cpu_offload = MagicMock(name="enable_model_cpu_offload")
        self.set_progress_bar_config = MagicMock(name="set_progress_bar_config")
        self.calls: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.images: list[Any] | None = None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        images = [f"image-{len(self.calls)}"] if self.images is None else self.images
        return SimpleNamespace(images=images)

    def reset_scheduler(self) -> None:
        self.scheduler = SimpleNamespace(config={"stock": "scheduler-config"})


def make_fake_diffusers(pipe: FakePipe) -> SimpleNamespace:
    return SimpleNamespace(
        AutoencoderKL=SimpleNamespace(
            from_pretrained=MagicMock(return_value="fp16-fix-vae")
        ),
        DiffusionPipeline=SimpleNamespace(from_pretrained=MagicMock(return_value=pipe)),
        DPMSolverMultistepScheduler=SimpleNamespace(
            from_config=MagicMock(return_value="karras-multistep")
        ),
        DPMSolverSinglestepScheduler=SimpleNamespace(
            from_config=MagicMock(return_value="karras-singlestep")
        ),
        EulerAncestralDiscreteScheduler=SimpleNamespace(
            from_config=MagicMock(return_value="euler-a")
        ),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(outputs_dir=tmp_path / "outputs")


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    pipe = FakePipe()
    torch = make_fake_torch()
    diffusers = make_fake_diffusers(pipe)
    quantization = MagicMock(return_value="quanto-int8")
    monkeypatch.setattr(pipeline_module, "_torch", lambda: torch)
    monkeypatch.setattr(pipeline_module, "_diffusers", lambda: diffusers)
    monkeypatch.setattr(pipeline_module, "_int8_quantization_config", quantization)
    return SimpleNamespace(
        pipe=pipe,
        torch=torch,
        diffusers=diffusers,
        quantization=quantization,
    )


@pytest.fixture
def generator(settings: Settings, fakes: SimpleNamespace) -> ImageGenerator:
    return ImageGenerator(settings)


def test_balanced_profile_loads_int8_sdxl_pipeline(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    call = fakes.diffusers.DiffusionPipeline.from_pretrained.call_args
    assert call.args == ("Lykon/dreamshaper-xl-v2-turbo",)
    assert call.kwargs == {
        "torch_dtype": fakes.torch.float16,
        "quantization_config": "quanto-int8",
        "use_safetensors": True,
        "variant": "fp16",
        "vae": "fp16-fix-vae",
    }
    fakes.diffusers.AutoencoderKL.from_pretrained.assert_called_once_with(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=fakes.torch.float16,
        use_safetensors=True,
    )
    fakes.diffusers.DPMSolverMultistepScheduler.from_config.assert_called_once_with(
        {"stock": "scheduler-config"}, use_karras_sigmas=True
    )
    assert fakes.pipe.scheduler == "karras-multistep"


def test_fast_profile_uses_bundled_vae_and_singlestep_scheduler(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("fast")

    call = fakes.diffusers.DiffusionPipeline.from_pretrained.call_args
    assert call.args == ("eimiss/EimisAnimeDiffusion_1.0v",)
    assert "vae" not in call.kwargs
    assert "variant" not in call.kwargs
    fakes.diffusers.AutoencoderKL.from_pretrained.assert_not_called()
    fakes.diffusers.DPMSolverSinglestepScheduler.from_config.assert_called_once_with(
        {"stock": "scheduler-config"}, use_karras_sigmas=True
    )
    assert fakes.pipe.scheduler == "karras-singlestep"


def test_best_profile_uses_euler_ancestral_scheduler(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("best")

    fakes.diffusers.EulerAncestralDiscreteScheduler.from_config.assert_called_once_with(
        {"stock": "scheduler-config"}
    )
    assert fakes.pipe.scheduler == "euler-a"


def test_load_enables_cpu_offload_tiling_and_quiet_progress(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    fakes.pipe.enable_model_cpu_offload.assert_called_once_with()
    fakes.pipe.vae.enable_tiling.assert_called_once_with()
    fakes.pipe.set_progress_bar_config.assert_called_once_with(disable=True)


def test_int8_pipeline_requires_cuda(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = make_fake_torch(cuda_available=False)
    monkeypatch.setattr(pipeline_module, "_torch", lambda: torch)

    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA GPU"):
        ImageGenerator(settings).load()


def test_load_is_idempotent(generator: ImageGenerator, fakes: SimpleNamespace) -> None:
    generator.load()
    generator.load()

    assert fakes.diffusers.DiffusionPipeline.from_pretrained.call_count == 1


def test_switching_profiles_unloads_the_previous_pipeline(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("balanced")
    fakes.pipe.reset_scheduler()
    generator.load("best")

    assert fakes.diffusers.DiffusionPipeline.from_pretrained.call_count == 2
    fakes.torch.cuda.empty_cache.assert_called_once_with()


def test_unload_allows_a_clean_reload(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()
    generator.unload()
    fakes.pipe.reset_scheduler()
    generator.load()

    assert fakes.diffusers.DiffusionPipeline.from_pretrained.call_count == 2
    fakes.torch.cuda.empty_cache.assert_called_once_with()


def test_generate_returns_sequential_seeded_results(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=1234, images=3)

    assert [result.image for result in results] == ["image-1", "image-2", "image-3"]
    assert [result.seed for result in results] == [1234, 1235, 1236]
    assert all(isinstance(result, GenerationResult) for result in results)
    assert [item.seed for item in fakes.torch.generators] == [1234, 1235, 1236]
    assert {item.device for item in fakes.torch.generators} == {"cpu"}


def test_random_seed_is_recorded(
    generator: ImageGenerator,
    fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module.random, "randint", lambda start, end: 999)

    results = generator.generate(PROMPT, images=2)

    assert [result.seed for result in results] == [999, 1000]


def test_seed_wraps_at_uint32_boundary(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=SEED_MAX, images=2)

    assert [result.seed for result in results] == [SEED_MAX, 0]


@pytest.mark.parametrize(
    ("model", "steps", "guidance", "size"),
    [
        ("balanced", 6, 2.0, (832, 1216)),
        ("best", 28, 5.0, (832, 1216)),
        ("fast", 20, 9.0, (768, 832)),
    ],
)
def test_profile_defaults_reach_the_pipeline(
    generator: ImageGenerator,
    fakes: SimpleNamespace,
    model: str,
    steps: int,
    guidance: float,
    size: tuple[int, int],
) -> None:
    generator.generate(PROMPT, seed=1, model=model)

    call = fakes.pipe.calls[0]
    assert call["num_inference_steps"] == steps
    assert call["guidance_scale"] == pytest.approx(guidance)
    assert (call["width"], call["height"]) == size


def test_explicit_generation_options_reach_the_pipeline(
    generator: ImageGenerator, fakes: SimpleNamespace
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


@pytest.mark.parametrize(("requested", "expected"), [(1, 4), (6, 6), (12, 8)])
def test_steps_are_clamped_to_the_profile(
    generator: ImageGenerator,
    fakes: SimpleNamespace,
    requested: int,
    expected: int,
) -> None:
    generator.generate(PROMPT, seed=1, steps=requested)

    assert fakes.pipe.calls[0]["num_inference_steps"] == expected


def test_image_count_is_capped(
    generator: ImageGenerator,
    fakes: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        results = generator.generate(PROMPT, seed=1, images=9)

    assert len(results) == 4
    assert "capping at 4" in caplog.text


def test_result_metadata_records_model_and_precision(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    result = generator.generate(PROMPT, seed=42)[0]

    assert result.settings["model_key"] == "balanced"
    assert result.settings["model"] == "DreamShaper XL v2 Turbo"
    assert result.settings["model_repo"] == "Lykon/dreamshaper-xl-v2-turbo"
    assert result.settings["linear_weight_dtype"] == "int8"
    assert result.settings["compute_dtype"] == "float16"
    assert result.settings["quantization_backend"] == "quanto"
    assert result.settings["estimated_render_time"] == "15-30 seconds"
    assert result.duration_s >= 0.0


@pytest.mark.parametrize("seed", [-1, SEED_MAX + 1])
def test_invalid_seed_is_rejected_before_loading(
    generator: ImageGenerator, fakes: SimpleNamespace, seed: int
) -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        generator.generate(PROMPT, seed=seed)

    fakes.diffusers.DiffusionPipeline.from_pretrained.assert_not_called()


@pytest.mark.parametrize(("width", "height"), [(0, 512), (513, 512), (512, 0)])
def test_invalid_dimensions_are_rejected_before_loading(
    generator: ImageGenerator,
    fakes: SimpleNamespace,
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError):
        generator.generate(PROMPT, width=width, height=height)

    fakes.diffusers.DiffusionPipeline.from_pretrained.assert_not_called()


@pytest.mark.parametrize("guidance", [-0.1, float("inf"), float("nan")])
def test_invalid_guidance_is_rejected_before_loading(
    generator: ImageGenerator, fakes: SimpleNamespace, guidance: float
) -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        generator.generate(PROMPT, guidance=guidance)

    fakes.diffusers.DiffusionPipeline.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": "   "}, "prompt must not be empty"),
        ({"prompt": PROMPT, "images": 0}, "images must be >= 1"),
        ({"prompt": PROMPT, "steps": 0}, "steps must be >= 1"),
        ({"prompt": PROMPT, "images": 1.5}, "images must be a whole number"),
        ({"prompt": PROMPT, "width": 512.5}, "width must be a whole number"),
        ({"prompt": PROMPT, "steps": True}, "steps must be a whole number"),
        ({"prompt": PROMPT, "seed": 1.5}, "seed must be a whole number"),
    ],
)
def test_other_invalid_arguments_raise(
    generator: ImageGenerator,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generator.generate(**kwargs)


def test_cuda_cache_is_emptied_when_generation_fails(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    fakes.pipe.error = RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        generator.generate(PROMPT, seed=1)

    fakes.torch.cuda.empty_cache.assert_called_once_with()


def test_oom_unloads_pipeline_and_reports_actionable_error(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    fakes.pipe.error = FakeOutOfMemoryError("out of memory")

    with pytest.raises(RuntimeError, match="close other GPU apps"):
        generator.generate(PROMPT, seed=1)

    assert fakes.torch.cuda.empty_cache.call_count == 2
    fakes.pipe.error = None
    fakes.pipe.reset_scheduler()
    generator.load()
    assert fakes.diffusers.DiffusionPipeline.from_pretrained.call_count == 2


def test_failed_load_cleans_cache_and_can_be_retried(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    loader = fakes.diffusers.DiffusionPipeline.from_pretrained
    loader.side_effect = RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        generator.load()

    fakes.torch.cuda.empty_cache.assert_called_once_with()
    loader.side_effect = None
    loader.return_value = fakes.pipe
    generator.load()
    assert loader.call_count == 2


def test_missing_output_image_is_an_error_and_cleans_cache(
    generator: ImageGenerator, fakes: SimpleNamespace
) -> None:
    fakes.pipe.images = []

    with pytest.raises(RuntimeError, match="returned no image"):
        generator.generate(PROMPT, seed=1)

    fakes.torch.cuda.empty_cache.assert_called_once_with()


def test_unknown_scheduler_is_reported(
    settings: Settings, fakes: SimpleNamespace
) -> None:
    settings.profiles["balanced"] = replace(
        settings.profiles["balanced"], scheduler="NoSuchScheduler"
    )

    with pytest.raises(ValueError, match="Unknown scheduler"):
        ImageGenerator(settings).load()

    fakes.torch.cuda.empty_cache.assert_called_once_with()
