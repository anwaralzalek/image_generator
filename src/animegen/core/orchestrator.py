"""End-to-end flow: enhance the prompt, render it, save images and metadata.

This is the single place where the style contract is applied, so the CLI and
the Gradio UI cannot drift apart: both call :meth:`Orchestrator.run`.

Every image is written next to a JSON sidecar holding the original prompt, the
enhanced prompt, the fallback flag, the full inference settings and per-stage
timings -- enough to reproduce any demo shot months later.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animegen import __version__
from animegen.config import Settings, get_settings
from animegen.llm.enhancer import EnhancedPrompt, OllamaEnhancer
from animegen.sd.pipeline import GenerationResult, SDXLGenerator

LOGGER = logging.getLogger(__name__)

#: outputs/20260716_143012_seed1234.png
FILENAME_TIMESTAMP = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class SavedImage:
    """One image on disk plus its sidecar."""

    path: Path
    metadata_path: Path
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """Everything one :meth:`Orchestrator.run` produced.

    Attributes:
        original_prompt: What the user typed.
        enhanced_prompt: LLM output (or the original) *with* the style suffix;
            this is the string SDXL actually received.
        used_fallback: True when the LLM was asked but could not be reached.
        llm_enabled: False when enhancement was skipped with ``--no-llm``.
        images: The saved images, in render order.
        llm_duration_s: Seconds spent in the enhancement stage.
        sd_duration_s: Seconds spent in the generation stage (includes the
            one-off model load when it happens on this run).
        total_duration_s: Seconds for the whole run, saving included.
    """

    original_prompt: str
    enhanced_prompt: str
    used_fallback: bool
    llm_enabled: bool
    images: list[SavedImage]
    llm_duration_s: float
    sd_duration_s: float
    total_duration_s: float

    @property
    def seeds(self) -> list[int]:
        """Seeds of the saved images, in render order."""
        return [image.seed for image in self.images]

    @property
    def paths(self) -> list[Path]:
        """Filesystem paths of the saved images, in render order."""
        return [image.path for image in self.images]


class Orchestrator:
    """Wire the enhancer, the generator and disk output together.

    Args:
        settings: Application settings; process-wide settings when omitted.
        enhancer: Prompt enhancer; a default :class:`OllamaEnhancer` when omitted.
        generator: Image generator; a default :class:`SDXLGenerator` when omitted.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        enhancer: OllamaEnhancer | None = None,
        generator: SDXLGenerator | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._enhancer = enhancer or OllamaEnhancer(settings=self._settings)
        self._generator = generator or SDXLGenerator(settings=self._settings)

    @property
    def settings(self) -> Settings:
        """The settings this orchestrator was built with."""
        return self._settings

    @property
    def generator(self) -> SDXLGenerator:
        """The underlying SDXL generator."""
        return self._generator

    @property
    def enhancer(self) -> OllamaEnhancer:
        """The underlying prompt enhancer."""
        return self._enhancer

    def apply_style(self, prompt: str) -> str:
        """Append the style contract to ``prompt``.

        The suffix is owned here rather than by the LLM, so its wording cannot
        be paraphrased away by a small model.

        Args:
            prompt: An enhanced (or raw) prompt without the style suffix.

        Returns:
            The prompt SDXL should receive.
        """
        suffix = self._settings.style.suffix
        body = prompt.strip().rstrip(",").strip()
        if suffix.strip().lstrip(",").strip().lower() in body.lower():
            # Already styled (e.g. a prompt replayed from a sidecar).
            return body
        return f"{body}{suffix}"

    def warmup(self) -> None:
        """Load the pipeline and render a throwaway image (see :meth:`SDXLGenerator.warmup`)."""
        self._generator.warmup()

    def close(self) -> None:
        """Release the pipeline and its VRAM."""
        self._generator.unload()

    def run(
        self,
        prompt: str,
        images: int = 1,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        use_llm: bool = True,
    ) -> RunResult:
        """Enhance, render and save.

        Args:
            prompt: The raw user idea.
            images: How many images to render.
            seed: Seed of the first image; random when None.
            width: Image width; configured default when None.
            height: Image height; configured default when None.
            steps: Denoising steps; configured default when None.
            guidance: Guidance scale; configured default when None.
            use_llm: When False, skip Ollama entirely and style the raw prompt.

        Returns:
            A :class:`RunResult` describing the saved images.

        Raises:
            ValueError: If ``prompt`` is blank.
        """
        user_prompt = prompt.strip()
        if not user_prompt:
            raise ValueError("prompt must not be empty")

        run_started = time.perf_counter()
        enhancement = self._enhance(user_prompt, use_llm)
        styled_prompt = self.apply_style(enhancement.enhanced)
        LOGGER.info("Final SDXL prompt: %s", styled_prompt)

        sd_started = time.perf_counter()
        results = self._generator.generate(
            prompt=styled_prompt,
            seed=seed,
            images=images,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
        )
        sd_duration = time.perf_counter() - sd_started

        timestamp = datetime.now()
        saved = [
            self._save(result, enhancement, styled_prompt, use_llm, timestamp, index)
            for index, result in enumerate(results)
        ]
        total = time.perf_counter() - run_started

        LOGGER.info(
            "Run finished in %.1fs (llm %.1fs, sd %.1fs): %d image(s), seeds %s",
            total,
            enhancement.duration_s,
            sd_duration,
            len(saved),
            [image.seed for image in saved],
        )
        return RunResult(
            original_prompt=user_prompt,
            enhanced_prompt=styled_prompt,
            used_fallback=enhancement.used_fallback,
            llm_enabled=use_llm,
            images=saved,
            llm_duration_s=enhancement.duration_s,
            sd_duration_s=sd_duration,
            total_duration_s=total,
        )

    def _enhance(self, user_prompt: str, use_llm: bool) -> EnhancedPrompt:
        """Run the LLM stage, or synthesise a pass-through result for --no-llm."""
        if not use_llm:
            LOGGER.info("LLM enhancement disabled; using the raw prompt")
            return EnhancedPrompt(original=user_prompt, enhanced=user_prompt)
        return self._enhancer.enhance(user_prompt)

    def _save(
        self,
        result: GenerationResult,
        enhancement: EnhancedPrompt,
        styled_prompt: str,
        use_llm: bool,
        timestamp: datetime,
        index: int,
    ) -> SavedImage:
        """Write one image and its sidecar, returning the record."""
        outputs_dir = self._settings.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{timestamp.strftime(FILENAME_TIMESTAMP)}_seed{result.seed}"
        image_path = self._unique_path(outputs_dir / f"{stem}.png")
        metadata_path = image_path.with_suffix(".json")

        metadata = self._build_metadata(
            result, enhancement, styled_prompt, use_llm, timestamp, index
        )
        result.image.save(image_path)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Saved %s", image_path)
        return SavedImage(
            path=image_path,
            metadata_path=metadata_path,
            seed=result.seed,
            metadata=metadata,
        )

    def _build_metadata(
        self,
        result: GenerationResult,
        enhancement: EnhancedPrompt,
        styled_prompt: str,
        use_llm: bool,
        timestamp: datetime,
        index: int,
    ) -> dict[str, Any]:
        """Assemble the JSON sidecar payload for one image."""
        metadata: dict[str, Any] = {
            "app_version": __version__,
            "created_at": timestamp.isoformat(timespec="seconds"),
            "index": index,
            "seed": result.seed,
            "final_prompt": styled_prompt,
            "style_suffix": self._settings.style.suffix,
            "llm_enabled": use_llm,
            "settings": result.settings,
            "timings": {
                "llm_s": round(enhancement.duration_s, 3),
                "render_s": round(result.duration_s, 3),
            },
        }
        metadata.update(enhancement.as_metadata())
        return metadata

    @staticmethod
    def _unique_path(path: Path) -> Path:
        """Return ``path``, or the first free ``name_2.png``-style variant."""
        candidate = path
        counter = 2
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
            counter += 1
        return candidate
