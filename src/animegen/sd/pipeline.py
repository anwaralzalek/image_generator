"""Lazy Diffusers pipelines with INT8 linear weights for an 8 GB NVIDIA GPU."""

from __future__ import annotations

import gc
import logging
import math
import random
import time
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

from animegen.config import (
    COMPUTE_DTYPE,
    LINEAR_WEIGHT_DTYPE,
    MAX_IMAGES,
    NEGATIVE_PROMPT,
    QUANTIZATION_BACKEND,
    SEED_MAX,
    VAE_REPO,
    ModelProfile,
    Settings,
    get_settings,
    parse_dimensions,
)

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

LOGGER = logging.getLogger(__name__)


def _whole_number(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number")
    return value


def _torch() -> ModuleType:
    import torch

    return torch


def _diffusers() -> ModuleType:
    import diffusers

    return diffusers


def _int8_quantization_config(profile: ModelProfile) -> Any:
    try:
        from diffusers import QuantoConfig as DiffusersQuantoConfig
        from diffusers.quantizers import PipelineQuantizationConfig
        from transformers import QuantoConfig as TransformersQuantoConfig
    except ImportError as exc:  # pragma: no cover - runtime dependency failure
        raise RuntimeError(
            "INT8 linear weights require Diffusers Quanto support; reinstall AnimeGen"
        ) from exc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`Quanto(?:Config|Quantizer)` is deprecated.*",
            category=FutureWarning,
        )
        mapping: dict[str, Any] = {
            "unet": DiffusersQuantoConfig(weights_dtype=LINEAR_WEIGHT_DTYPE),
            "text_encoder": TransformersQuantoConfig(weights=LINEAR_WEIGHT_DTYPE),
        }
        if profile.architecture == "sdxl":
            mapping["text_encoder_2"] = TransformersQuantoConfig(
                weights=LINEAR_WEIGHT_DTYPE
            )
    return PipelineQuantizationConfig(quant_mapping=mapping)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    image: "Image"
    seed: int
    settings: dict[str, Any]
    duration_s: float


class ImageGenerator:
    """Load one selected SD/SDXL profile at a time and render sequentially."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pipe: Any | None = None
        self._model_key: str | None = None

    def load(self, model: str | None = None) -> None:
        model_key, profile = self._settings.image_model(model)
        if self._pipe is not None and self._model_key == model_key:
            return
        self.unload()

        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Quanto image generation requires an NVIDIA CUDA GPU; "
                "torch.cuda.is_available() is False"
            )

        diffusers = _diffusers()
        dtype = torch.float16
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "quantization_config": _int8_quantization_config(profile),
            "use_safetensors": True,
        }
        if profile.variant:
            kwargs["variant"] = profile.variant

        started = time.perf_counter()
        LOGGER.info(
            "Loading %s from %s (%s linear weights, %s compute)",
            profile.name,
            profile.repo_id,
            LINEAR_WEIGHT_DTYPE.upper(),
            COMPUTE_DTYPE,
        )
        try:
            if profile.architecture == "sdxl":
                kwargs["vae"] = diffusers.AutoencoderKL.from_pretrained(
                    VAE_REPO, torch_dtype=dtype, use_safetensors=True
                )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`Quanto(?:Config|Quantizer)` is deprecated.*",
                    category=FutureWarning,
                )
                pipe = diffusers.DiffusionPipeline.from_pretrained(
                    profile.repo_id, **kwargs
                )
            pipe.scheduler = self._build_scheduler(
                diffusers, pipe.scheduler.config, profile
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
        except Exception:
            gc.collect()
            self._empty_cache()
            raise

        self._pipe = pipe
        self._model_key = model_key
        LOGGER.info("%s ready in %.1fs", profile.name, time.perf_counter() - started)

    def unload(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        self._model_key = None
        gc.collect()
        self._empty_cache()
        LOGGER.info("Image pipeline unloaded")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        images: int = 1,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        model: str | None = None,
    ) -> list[GenerationResult]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        prompt = prompt.strip()
        images = _whole_number(images, "images")
        if images < 1:
            raise ValueError("images must be >= 1")

        model_key, profile = self._settings.image_model(model)
        width = profile.width if width is None else _whole_number(width, "width")
        height = profile.height if height is None else _whole_number(height, "height")
        parse_dimensions(f"{width}x{height}")
        if f"{width}x{height}" not in profile.allowed_sizes:
            LOGGER.warning(
                "Size %dx%d is outside %s's tested set (%s); expect slower renders or OOM",
                width,
                height,
                profile.name,
                ", ".join(profile.allowed_sizes),
            )

        guidance = profile.guidance_scale if guidance is None else float(guidance)
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be a finite, non-negative number")
        selected_steps = (
            profile.steps if steps is None else _whole_number(steps, "steps")
        )
        if selected_steps < 1:
            raise ValueError("steps must be >= 1")
        steps = self._clamp_profile_steps(selected_steps, profile)

        if seed is not None:
            seed = _whole_number(seed, "seed")
            if not 0 <= seed <= SEED_MAX:
                raise ValueError(f"seed must be between 0 and {SEED_MAX}")
        if images > MAX_IMAGES:
            LOGGER.warning(
                "Requested %d images, capping at %d to stay inside the VRAM budget",
                images,
                MAX_IMAGES,
            )
            images = MAX_IMAGES

        negative = (
            profile.negative_prompt or NEGATIVE_PROMPT
            if negative_prompt is None
            else negative_prompt
        )
        base_seed = random.randint(0, SEED_MAX) if seed is None else seed
        self.load(model_key)
        torch = _torch()
        results: list[GenerationResult] = []

        try:
            for index in range(images):
                image_seed = (base_seed + index) % (SEED_MAX + 1)
                generator = torch.Generator(device="cpu").manual_seed(image_seed)
                started = time.perf_counter()
                output = self._pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                )
                duration = time.perf_counter() - started
                if not getattr(output, "images", None):
                    raise RuntimeError("The image model returned no image")
                results.append(
                    GenerationResult(
                        image=output.images[0],
                        seed=image_seed,
                        settings=self._record_settings(
                            model_key,
                            profile,
                            negative,
                            width,
                            height,
                            steps,
                            guidance,
                        ),
                        duration_s=duration,
                    )
                )
                LOGGER.info(
                    "Image %d/%d rendered by %s in %.1fs (seed=%d)",
                    index + 1,
                    images,
                    profile.name,
                    duration,
                    image_seed,
                )
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise RuntimeError(
                "CUDA ran out of memory; close other GPU apps or use a smaller model/size"
            ) from exc
        finally:
            self._empty_cache()
        return results

    @staticmethod
    def _record_settings(
        model_key: str,
        profile: ModelProfile,
        negative: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
    ) -> dict[str, Any]:
        return {
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "scheduler": profile.scheduler,
            "use_karras_sigmas": profile.use_karras_sigmas,
            "model_key": model_key,
            "model": profile.name,
            "model_repo": profile.repo_id,
            "quality_tier": profile.quality,
            "parameter_count": profile.parameter_count,
            "estimated_render_time": profile.estimate,
            "linear_weight_dtype": LINEAR_WEIGHT_DTYPE,
            "compute_dtype": COMPUTE_DTYPE,
            "quantization_backend": QUANTIZATION_BACKEND,
            "vae": VAE_REPO if profile.architecture == "sdxl" else "bundled with model",
            "device": "cuda",
        }

    @staticmethod
    def _clamp_profile_steps(steps: int, profile: ModelProfile) -> int:
        clamped = max(profile.min_steps, min(profile.max_steps, steps))
        if clamped != steps:
            LOGGER.warning(
                "Steps %d outside %s's %d-%d range; using %d",
                steps,
                profile.name,
                profile.min_steps,
                profile.max_steps,
                clamped,
            )
        return clamped

    @staticmethod
    def _build_scheduler(
        diffusers: ModuleType, config: Any, profile: ModelProfile
    ) -> Any:
        scheduler_cls = getattr(diffusers, profile.scheduler, None)
        if scheduler_cls is None:
            raise ValueError(
                f"Unknown scheduler '{profile.scheduler}' for {profile.name}"
            )
        kwargs = {"use_karras_sigmas": True} if profile.use_karras_sigmas else {}
        return scheduler_cls.from_config(config, **kwargs)

    @staticmethod
    def _empty_cache() -> None:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
