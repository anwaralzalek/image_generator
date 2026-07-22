"""Selectable INT8 diffusion pipelines tuned for an 8 GB NVIDIA GPU.

Every profile quantizes the supported denoiser and text-encoder ``Linear``
weights to INT8 with Quanto. Diffusers still performs arithmetic and keeps
unsupported layers (including the VAE) in fp16; a fully integer end-to-end
diffusion pipeline is not supported. Model CPU offload, VAE tiling, sequential
image generation, and cache cleanup keep the supported profiles within the
reference RTX 3070 Laptop memory budget.

Heavy dependencies are imported lazily so the module remains unit-testable
without loading a real model.
"""

from __future__ import annotations

import gc
import logging
import random
import time
import warnings
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

from animegen.config import ModelProfile, Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

LOGGER = logging.getLogger(__name__)

SEED_MAX = 2**32 - 1
WARMUP_PROMPT = "warmup"
WARMUP_SIZE = 512
WARMUP_STEPS = 1


def _torch() -> ModuleType:
    import torch

    return torch


def _diffusers() -> ModuleType:
    import diffusers

    return diffusers


def _int8_quantization_config(profile: ModelProfile) -> Any:
    """Build a component-aware Quanto INT8 configuration for ``profile``."""
    try:
        from diffusers import QuantoConfig as DiffusersQuantoConfig
        from diffusers.quantizers import PipelineQuantizationConfig
        from transformers import QuantoConfig as TransformersQuantoConfig
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "INT8 image models require Diffusers quantization support. "
            "Install the project dependencies again."
        ) from exc

    # Diffusers 0.39 warns about a future backend rename even though Quanto is
    # still its documented INT8 path. The project is pinned below 1.0 until a
    # supported migration exists, so keep normal CLI output free of that noise.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`Quanto(?:Config|Quantizer)` is deprecated.*",
            category=FutureWarning,
        )
        mapping: dict[str, Any] = {
            name: DiffusersQuantoConfig(weights_dtype="int8")
            for name in profile.quantize_diffusers
        }
        mapping.update(
            {
                name: TransformersQuantoConfig(weights="int8")
                for name in profile.quantize_transformers
            }
        )
    return PipelineQuantizationConfig(quant_mapping=mapping)


@dataclass(frozen=True)
class GenerationResult:
    """One rendered image plus its reproducibility data."""

    image: "Image"
    seed: int
    settings: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


class SDXLGenerator:
    """Lazy loader for all configured SD/SDXL image model profiles.

    The historical class name is retained for API compatibility. The active
    pipeline may now be either SDXL or compact Stable Diffusion, depending on
    the selected profile.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pipe: Any | None = None
        self._device: str | None = None
        self._model_key: str | None = None
        self._profile: ModelProfile | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def model_key(self) -> str | None:
        return self._model_key

    @property
    def profile(self) -> ModelProfile | None:
        return self._profile

    @property
    def pipe(self) -> Any:
        if self._pipe is None:
            self.load()
        return self._pipe

    def load(self, model: str | None = None) -> None:
        """Load ``model`` in mandatory INT8 mode, replacing any active model."""
        model_key, profile = self._settings.image_model(model)
        if self._pipe is not None and self._model_key == model_key:
            return
        if self._pipe is not None:
            self.unload()

        torch = _torch()
        diffusers = _diffusers()
        vram = self._settings.vram

        if not torch.cuda.is_available():
            raise RuntimeError(
                "INT8 image generation requires an NVIDIA CUDA GPU; "
                "torch.cuda.is_available() is False"
            )
        if vram.weight_dtype != "int8" or vram.quantization_backend != "quanto":
            raise ValueError(
                "This application requires vram.weight_dtype=int8 and "
                "vram.quantization_backend=quanto"
            )

        device = "cuda"
        dtype = self._resolve_dtype(torch)
        quantization_config = _int8_quantization_config(profile)

        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "quantization_config": quantization_config,
            # Never fall back to pickle-based .bin/.ckpt weights.
            "use_safetensors": True,
        }
        if profile.variant:
            kwargs["variant"] = profile.variant
        if profile.architecture == "sdxl":
            kwargs["vae"] = diffusers.AutoencoderKL.from_pretrained(
                self._settings.model.vae_repo,
                torch_dtype=dtype,
                use_safetensors=True,
            )

        started = time.perf_counter()
        LOGGER.info(
            "Loading %s from %s (INT8 weights, %s compute)",
            profile.name,
            profile.repo_id,
            vram.dtype,
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

        if vram.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if vram.enable_vae_tiling:
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            elif hasattr(pipe, "enable_vae_tiling"):
                # Compatibility with older Diffusers pipelines.
                pipe.enable_vae_tiling()

        self._pipe = pipe
        self._device = device
        self._model_key = model_key
        self._profile = profile
        LOGGER.info("%s ready in %.1fs", profile.name, time.perf_counter() - started)

    def unload(self) -> None:
        """Release the active pipeline, RAM, and VRAM."""
        if self._pipe is None:
            return
        self._pipe = None
        self._device = None
        self._model_key = None
        self._profile = None
        gc.collect()
        self._empty_cache()
        LOGGER.info("Image pipeline unloaded")

    def warmup(self, model: str | None = None) -> None:
        """Load a model and run a one-step 512px throwaway render."""
        LOGGER.info("Warming up image pipeline")
        started = time.perf_counter()
        self.generate(
            prompt=WARMUP_PROMPT,
            negative_prompt="",
            seed=0,
            images=1,
            width=WARMUP_SIZE,
            height=WARMUP_SIZE,
            steps=WARMUP_STEPS,
            guidance=1.0,
            model=model,
            _clamp_steps=False,
        )
        LOGGER.info("Warmup finished in %.1fs", time.perf_counter() - started)

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
        _clamp_steps: bool = True,
    ) -> list[GenerationResult]:
        """Render images sequentially with the selected model profile."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if images < 1:
            raise ValueError("images must be >= 1")

        model_key, profile = self._settings.image_model(model)
        negative = (
            profile.negative_prompt or self._settings.style.negative_prompt
            if negative_prompt is None
            else negative_prompt
        )
        width = width or profile.width
        height = height or profile.height
        guidance = profile.guidance_scale if guidance is None else guidance
        selected_steps = profile.steps if steps is None else int(steps)
        steps = (
            self._clamp_profile_steps(selected_steps, profile)
            if _clamp_steps
            else selected_steps
        )

        max_images = self._settings.generation.max_images_per_run
        if images > max_images:
            LOGGER.warning(
                "Requested %d images, capping at %d to stay inside the VRAM budget",
                images,
                max_images,
            )
            images = max_images

        self.load(model_key)
        torch = _torch()
        results: list[GenerationResult] = []
        base_seed = random.randint(0, SEED_MAX) if seed is None else int(seed)

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
            results.append(
                GenerationResult(
                    image=output.images[0],
                    seed=image_seed,
                    settings=self._record_settings(
                        model_key,
                        profile,
                        prompt,
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

        if self._settings.vram.empty_cache_after_run:
            self._empty_cache()
        return results

    def _record_settings(
        self,
        model_key: str,
        profile: ModelProfile,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
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
            "weight_dtype": self._settings.vram.weight_dtype,
            "compute_dtype": self._settings.vram.dtype,
            "quantization_backend": self._settings.vram.quantization_backend,
            "vae": (
                self._settings.model.vae_repo
                if profile.architecture == "sdxl"
                else "bundled with model"
            ),
            "device": self._device,
        }

    @staticmethod
    def _clamp_profile_steps(steps: int, profile: ModelProfile) -> int:
        clamped = max(profile.min_steps, min(profile.max_steps, int(steps)))
        if clamped != steps:
            LOGGER.warning(
                "Steps %s outside %s's %d-%d range; using %d",
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

    def _resolve_dtype(self, torch: ModuleType) -> Any:
        dtype = getattr(torch, self._settings.vram.dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown compute dtype '{self._settings.vram.dtype}'")
        return dtype

    @staticmethod
    def _empty_cache() -> None:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "SDXLGenerator":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()
