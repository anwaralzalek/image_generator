"""SDXL Turbo generation pipeline tuned for an 8 GB RTX 3070 Laptop.

The VRAM strategy is not optional and is applied on every load:

* ``torch.float16`` weights,
* ``madebyollin/sdxl-vae-fp16-fix`` in place of the stock VAE (the original
  overflows in fp16 and yields black or NaN images),
* ``enable_model_cpu_offload()`` instead of ``.to("cuda")`` so only the
  currently executing submodule occupies VRAM,
* ``enable_vae_tiling()`` so decoding 832x1216 does not spike memory,
* one image per pipeline call, with ``torch.cuda.empty_cache()`` after the run.

``torch`` and ``diffusers`` are imported lazily inside :func:`_torch` and
:func:`_diffusers`, which keeps this module importable -- and unit-testable --
on a machine with neither installed.
"""

from __future__ import annotations

import gc
import logging
import random
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

from animegen.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

LOGGER = logging.getLogger(__name__)

#: Upper bound of the seed space; matches what SD front-ends conventionally use.
SEED_MAX = 2**32 - 1

#: Throwaway generation used to compile CUDA kernels before a demo.
WARMUP_PROMPT = "warmup"
WARMUP_SIZE = 512
WARMUP_STEPS = 1


def _torch() -> ModuleType:
    """Import ``torch`` lazily so the module imports without a GPU stack."""
    import torch

    return torch


def _diffusers() -> ModuleType:
    """Import ``diffusers`` lazily so the module imports without a GPU stack."""
    import diffusers

    return diffusers


@dataclass(frozen=True)
class GenerationResult:
    """One rendered image plus everything needed to reproduce it.

    Attributes:
        image: The decoded PIL image.
        seed: The exact seed used for this image.
        settings: Flat, JSON-serialisable record of the inference parameters.
        duration_s: Wall-clock seconds spent rendering this image.
    """

    image: "Image"
    seed: int
    settings: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


class SDXLGenerator:
    """Lazy-loading wrapper around ``StableDiffusionXLPipeline``.

    The pipeline is built on the first :meth:`load` (or first :meth:`generate`)
    and reused afterwards, because loading a 6.5 GB checkpoint takes far longer
    than a 6-step render.

    Args:
        settings: Application settings; the process-wide settings are used when
            omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._pipe: Any | None = None
        self._device: str | None = None

    @property
    def is_loaded(self) -> bool:
        """True once the diffusers pipeline is in memory."""
        return self._pipe is not None

    @property
    def device(self) -> str | None:
        """Compute device chosen at load time, or None before loading."""
        return self._device

    @property
    def pipe(self) -> Any:
        """The underlying diffusers pipeline (loading it if necessary)."""
        self.load()
        return self._pipe

    def load(self) -> None:
        """Build the pipeline and apply the 8 GB VRAM strategy.

        Idempotent: a second call is a no-op.

        Raises:
            FileNotFoundError: If the single-file checkpoint is missing.
            ValueError: If the configured scheduler is unknown to diffusers.
        """
        if self._pipe is not None:
            return

        torch = _torch()
        diffusers = _diffusers()
        vram = self._settings.vram
        checkpoint = self._settings.checkpoint_path

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SDXL checkpoint not found at {checkpoint}. "
                "Run: python scripts/download_models.py --url <checkpoint-url>"
            )

        self._device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, self._device)

        started = time.perf_counter()
        LOGGER.info("Loading %s (%s, %s)", checkpoint.name, dtype, self._device)

        # The stock SDXL VAE overflows in fp16 and renders black images.
        vae = diffusers.AutoencoderKL.from_pretrained(
            self._settings.model.vae_repo, torch_dtype=dtype
        )
        pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            str(checkpoint),
            torch_dtype=dtype,
            vae=vae,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.scheduler = self._build_scheduler(diffusers, pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        if self._device == "cuda" and vram.enable_model_cpu_offload:
            # Sequentially streams submodules to the GPU: peak VRAM ~4 GB
            # instead of ~10 GB for a plain .to("cuda").
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)

        if vram.enable_vae_tiling:
            pipe.enable_vae_tiling()

        self._pipe = pipe
        LOGGER.info("Pipeline ready in %.1fs", time.perf_counter() - started)

    def unload(self) -> None:
        """Drop the pipeline and release VRAM. Safe to call when not loaded."""
        if self._pipe is None:
            return
        self._pipe = None
        self._device = None
        gc.collect()
        self._empty_cache()
        LOGGER.info("Pipeline unloaded")

    def warmup(self) -> None:
        """Render a tiny throwaway image so the first demo shot is not the slow one.

        The first CUDA call of a process pays for kernel compilation and offload
        hook setup; doing that on a 512x512 single-step render costs a few
        seconds instead of stalling the audience-facing generation.
        """
        LOGGER.info("Warming up CUDA kernels")
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
    ) -> list[GenerationResult]:
        """Render ``images`` pictures, one pipeline call each.

        Images are rendered sequentially rather than as a batch: a batch of four
        832x1216 latents does not fit alongside the UNet on 8 GB, and per-image
        calls give every image its own recorded seed.

        Args:
            prompt: Full positive prompt, style suffix already applied.
            negative_prompt: Negative prompt; the configured default when None.
            seed: Seed of the first image; subsequent images use ``seed + i``.
                A random seed is drawn when None.
            images: How many images to render.
            width: Image width; the configured default when None.
            height: Image height; the configured default when None.
            steps: Denoising steps, clamped to the configured Turbo range.
            guidance: Classifier-free guidance scale; configured default when None.

        Returns:
            One :class:`GenerationResult` per image, in render order.

        Raises:
            ValueError: If ``prompt`` is blank or ``images`` is below 1.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if images < 1:
            raise ValueError("images must be >= 1")

        gen_cfg = self._settings.generation
        negative = (
            self._settings.style.negative_prompt
            if negative_prompt is None
            else negative_prompt
        )
        width = width or gen_cfg.width
        height = height or gen_cfg.height
        guidance = gen_cfg.guidance_scale if guidance is None else guidance
        steps = self._clamp_steps(gen_cfg.steps if steps is None else steps)
        max_images = gen_cfg.max_images_per_run
        if images > max_images:
            LOGGER.warning(
                "Requested %d images, capping at %d to stay inside the VRAM budget",
                images,
                max_images,
            )
            images = max_images

        self.load()
        torch = _torch()
        results: list[GenerationResult] = []
        base_seed = random.randint(0, SEED_MAX) if seed is None else int(seed)

        for index in range(images):
            image_seed = (base_seed + index) % (SEED_MAX + 1)
            # A CPU generator keeps seeds reproducible regardless of which
            # device the offloaded submodules currently live on.
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
                        prompt, negative, width, height, steps, guidance
                    ),
                    duration_s=duration,
                )
            )
            LOGGER.info(
                "Image %d/%d rendered in %.1fs (seed=%d)",
                index + 1,
                images,
                duration,
                image_seed,
            )

        if self._settings.vram.empty_cache_after_run:
            self._empty_cache()
        return results

    def _record_settings(
        self,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
    ) -> dict[str, Any]:
        """Snapshot the inference parameters for the metadata sidecar."""
        return {
            "prompt": prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "scheduler": self._settings.generation.scheduler,
            "use_karras_sigmas": self._settings.generation.use_karras_sigmas,
            "model": self._settings.model.checkpoint_name,
            "checkpoint": self._settings.model.checkpoint_filename,
            "vae": self._settings.model.vae_repo,
            "dtype": self._settings.vram.dtype,
            "device": self._device,
        }

    def _clamp_steps(self, steps: int) -> int:
        """Keep steps inside the Turbo-friendly range from settings."""
        gen_cfg = self._settings.generation
        clamped = max(gen_cfg.min_steps, min(gen_cfg.max_steps, int(steps)))
        if clamped != steps:
            LOGGER.warning(
                "Steps %s outside the %d-%d Turbo range; using %d",
                steps,
                gen_cfg.min_steps,
                gen_cfg.max_steps,
                clamped,
            )
        return clamped

    def _build_scheduler(self, diffusers: ModuleType, config: Any) -> Any:
        """Instantiate the configured scheduler from the checkpoint's config."""
        name = self._settings.generation.scheduler
        scheduler_cls = getattr(diffusers, name, None)
        if scheduler_cls is None:
            raise ValueError(f"Unknown scheduler '{name}' for this diffusers version")
        return scheduler_cls.from_config(
            config, use_karras_sigmas=self._settings.generation.use_karras_sigmas
        )

    def _resolve_device(self, torch: ModuleType) -> str:
        """Return "cuda" when a GPU is usable, else "cpu" with a warning."""
        if torch.cuda.is_available():
            return "cuda"
        LOGGER.warning(
            "No CUDA device available; running SDXL on the CPU will be extremely slow"
        )
        return "cpu"

    def _resolve_dtype(self, torch: ModuleType, device: str) -> Any:
        """Map the configured dtype name to a torch dtype (fp32 on CPU)."""
        if device != "cuda":
            # fp16 maths is unsupported on most CPU backends.
            return torch.float32
        dtype = getattr(torch, self._settings.vram.dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype '{self._settings.vram.dtype}'")
        return dtype

    def _empty_cache(self) -> None:
        """Release cached CUDA blocks; a no-op without a GPU."""
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "SDXLGenerator":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()
