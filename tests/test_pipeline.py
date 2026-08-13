"""Tests for selectable INT8 image pipelines; no real GPU or weights are used."""

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
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> "FakeTorchGenerator":
        self.seed = seed
        return self


def make_fake_torch(cuda_available: bool = True) -> SimpleNamespace:
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
    return SimpleNamespace(
        AutoencoderKL=SimpleNamespace(
            from_pretrained=MagicMock(return_value="fp16-fix-vae")
        ),
        DiffusionPipeline=SimpleNamespace(
            from_pretrained=MagicMock(return_value=pipe)
        ),
        DPMSolverMultistepScheduler=SimpleNamespace(
            from_config=MagicMock(return_value="karras-scheduler")
        ),
        EulerAncestralDiscreteScheduler=SimpleNamespace(
            from_config=MagicMock(return_value="euler-a-scheduler")
        ),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(
        DEFAULT_CONFIG_FILE,
        paths={"models_dir": tmp_path / "models", "outputs_dir": tmp_path / "outputs"},
    )


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    pipe = FakePipe()
    torch = make_fake_torch()
    diffusers = make_fake_diffusers(pipe)
    monkeypatch.setattr(pipeline_module, "_torch", lambda: torch)
    monkeypatch.setattr(pipeline_module, "_diffusers", lambda: diffusers)
    monkeypatch.setattr(
        pipeline_module, "_int8_quantization_config", lambda profile: "quanto-int8"
    )
    return SimpleNamespace(pipe=pipe, torch=torch, diffusers=diffusers)


@pytest.fixture
def generator(settings: Settings, fakes: SimpleNamespace) -> SDXLGenerator:
    return SDXLGenerator(settings=settings)


def test_default_profile_loads_dreamshaper_with_int8_config(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    call = fakes.diffusers.DiffusionPipeline.from_pretrained.call_args
    assert call.args == ("Lykon/dreamshaper-xl-v2-turbo",)
    assert call.kwargs["quantization_config"] == "quanto-int8"
    assert call.kwargs["torch_dtype"] == fakes.torch.float16
    assert call.kwargs["use_safetensors"] is True
    assert call.kwargs["variant"] == "fp16"
    assert call.kwargs["vae"] == "fp16-fix-vae"
    assert generator.model_key == "balanced"


def test_sdxl_profiles_use_the_stable_fp16_vae(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("best")

    fakes.diffusers.AutoencoderKL.from_pretrained.assert_called_once_with(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=fakes.torch.float16,
        use_safetensors=True,
    )


def test_fast_profile_uses_its_bundled_vae(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("fast")

    fakes.diffusers.AutoencoderKL.from_pretrained.assert_not_called()
    call = fakes.diffusers.DiffusionPipeline.from_pretrained.call_args
    assert call.args == ("dreamlike-art/dreamlike-anime-1.0",)
    assert "vae" not in call.kwargs


def test_load_enables_cpu_offload_and_vae_tiling(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    fakes.pipe.enable_model_cpu_offload.assert_called_once_with()
    fakes.pipe.enable_vae_tiling.assert_called_once_with()
    fakes.pipe.to.assert_not_called()
    assert generator.device == "cuda"


def test_load_uses_the_current_vae_tiling_api(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    vae = SimpleNamespace(enable_tiling=MagicMock(name="enable_tiling"))
    fakes.pipe.vae = vae

    generator.load()

    vae.enable_tiling.assert_called_once_with()
    fakes.pipe.enable_vae_tiling.assert_not_called()


def test_default_profile_installs_karras_scheduler(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()

    fakes.diffusers.DPMSolverMultistepScheduler.from_config.assert_called_once_with(
        {"stock": "scheduler-config"}, use_karras_sigmas=True
    )
    assert fakes.pipe.scheduler == "karras-scheduler"


def test_best_profile_uses_euler_ancestral_without_karras(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("best")

    fakes.diffusers.EulerAncestralDiscreteScheduler.from_config.assert_called_once_with(
        {"stock": "scheduler-config"}
    )
    assert fakes.pipe.scheduler == "euler-a-scheduler"


def test_unknown_scheduler_raises(settings: Settings, fakes: SimpleNamespace) -> None:
    settings.model.profiles["balanced"].scheduler = "NoSuchScheduler"

    with pytest.raises(ValueError, match="Unknown scheduler"):
        SDXLGenerator(settings=settings).load()


def test_int8_mode_requires_cuda(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = FakePipe()
    torch = make_fake_torch(cuda_available=False)
    monkeypatch.setattr(pipeline_module, "_torch", lambda: torch)
    monkeypatch.setattr(pipeline_module, "_diffusers", lambda: make_fake_diffusers(pipe))

    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA GPU"):
        SDXLGenerator(settings=settings).load()


def test_non_int8_configuration_is_rejected(
    settings: Settings, fakes: SimpleNamespace
) -> None:
    settings.vram.weight_dtype = "float16"

    with pytest.raises(ValueError, match="weight_dtype=int8"):
        SDXLGenerator(settings=settings).load()


def test_load_is_idempotent(generator: SDXLGenerator, fakes: SimpleNamespace) -> None:
    generator.load()
    generator.load()

    assert fakes.diffusers.DiffusionPipeline.from_pretrained.call_count == 1
    assert generator.is_loaded is True


def test_pipe_property_does_not_replace_an_active_non_default_profile(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("best")

    assert generator.pipe is fakes.pipe
    assert generator.model_key == "best"
    assert fakes.diffusers.DiffusionPipeline.from_pretrained.call_count == 1


def test_switching_profiles_replaces_the_loaded_pipeline(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load("balanced")
    # Real Diffusers calls return a fresh pipeline. This lightweight fake is
    # intentionally reused, so restore the scheduler object before reloading.
    fakes.pipe.scheduler = SimpleNamespace(config={"stock": "scheduler-config"})
    generator.load("best")

    assert fakes.diffusers.DiffusionPipeline.from_pretrained.call_count == 2
    assert generator.model_key == "best"
    assert fakes.torch.cuda.empty_cache.call_count == 1


def test_generate_returns_one_result_per_image(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=1234, images=3)

    assert [r.image for r in results] == ["image-1", "image-2", "image-3"]
    assert all(isinstance(r, GenerationResult) for r in results)
    assert len(fakes.pipe.calls) == 3


def test_seeds_are_deterministic_and_sequential(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=1234, images=3)

    assert [r.seed for r in results] == [1234, 1235, 1236]
    assert [g.seed for g in fakes.torch.generators] == [1234, 1235, 1236]
    assert {g.device for g in fakes.torch.generators} == {"cpu"}


def test_random_seed_is_drawn_and_recorded(
    generator: SDXLGenerator,
    fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module.random, "randint", lambda a, b: 999)

    results = generator.generate(PROMPT, images=2)

    assert [r.seed for r in results] == [999, 1000]


def test_seed_wraps_at_boundary(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    results = generator.generate(PROMPT, seed=SEED_MAX, images=2)

    assert [r.seed for r in results] == [SEED_MAX, 0]


def test_balanced_profile_defaults_reach_the_pipeline(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.generate(PROMPT, seed=1, model="balanced")

    call = fakes.pipe.calls[0]
    assert call["num_inference_steps"] == 6
    assert call["guidance_scale"] == pytest.approx(2.0)
    assert (call["width"], call["height"]) == (832, 1216)


def test_best_profile_defaults_reach_the_pipeline(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.generate(PROMPT, seed=1, model="best")

    call = fakes.pipe.calls[0]
    assert call["num_inference_steps"] == 28
    assert call["guidance_scale"] == pytest.approx(5.0)
    assert (call["width"], call["height"]) == (832, 1216)


def test_fast_profile_defaults_reach_the_pipeline(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.generate(PROMPT, seed=1, model="fast")

    call = fakes.pipe.calls[0]
    assert call["num_inference_steps"] == 20
    assert call["guidance_scale"] == pytest.approx(7.5)
    assert (call["width"], call["height"]) == (768, 768)


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


@pytest.mark.parametrize(("requested", "expected"), [(1, 4), (6, 6), (12, 8)])
def test_steps_are_clamped_to_the_selected_profile(
    generator: SDXLGenerator,
    fakes: SimpleNamespace,
    requested: int,
    expected: int,
) -> None:
    generator.generate(PROMPT, seed=1, model="balanced", steps=requested)

    assert fakes.pipe.calls[0]["num_inference_steps"] == expected


def test_image_count_is_capped(
    generator: SDXLGenerator,
    fakes: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        results = generator.generate(PROMPT, seed=1, images=9)

    assert len(results) == 4
    assert "capping at 4" in caplog.text


def test_result_metadata_records_model_and_precision(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    result = generator.generate(PROMPT, seed=42, model="balanced")[0]

    assert result.settings["model_key"] == "balanced"
    assert result.settings["model"] == "DreamShaper XL v2 Turbo"
    assert result.settings["model_repo"] == "Lykon/dreamshaper-xl-v2-turbo"
    assert result.settings["weight_dtype"] == "int8"
    assert result.settings["compute_dtype"] == "float16"
    assert result.settings["quantization_backend"] == "quanto"
    assert result.settings["estimated_render_time"] == "15-30 seconds"
    assert result.duration_s >= 0.0


def test_generate_loads_on_demand(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    assert generator.is_loaded is False
    generator.generate(PROMPT, seed=1)
    assert generator.is_loaded is True


def test_cuda_cache_is_emptied_after_run(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.generate(PROMPT, seed=1, images=2)

    fakes.torch.cuda.empty_cache.assert_called_once_with()


def test_cache_emptying_can_be_disabled(
    settings: Settings, fakes: SimpleNamespace
) -> None:
    settings.vram.empty_cache_after_run = False

    SDXLGenerator(settings=settings).generate(PROMPT, seed=1)

    fakes.torch.cuda.empty_cache.assert_not_called()


def test_unload_releases_pipeline_and_vram(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.load()
    generator.unload()

    assert generator.is_loaded is False
    assert generator.device is None
    assert generator.model_key is None
    fakes.torch.cuda.empty_cache.assert_called_once_with()


def test_context_manager_loads_and_unloads(
    settings: Settings, fakes: SimpleNamespace
) -> None:
    with SDXLGenerator(settings=settings) as gen:
        assert gen.is_loaded is True
    assert gen.is_loaded is False


def test_warmup_uses_one_step_without_profile_clamping(
    generator: SDXLGenerator, fakes: SimpleNamespace
) -> None:
    generator.warmup("best")

    call = fakes.pipe.calls[0]
    assert (call["width"], call["height"]) == (512, 512)
    assert call["num_inference_steps"] == 1
    assert generator.model_key == "best"


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
