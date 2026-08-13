"""Enhance a prompt, render images, and save reproducibility sidecars."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from animegen import __version__
from animegen.config import STYLE_SUFFIX, Settings, get_settings
from animegen.llm.enhancer import EnhancedPrompt, OllamaEnhancer
from animegen.sd.pipeline import GenerationResult, ImageGenerator

LOGGER = logging.getLogger(__name__)

#: outputs/20260716_143012_seed1234_<unique-id>.png
FILENAME_TIMESTAMP = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class SavedImage:
    path: Path
    metadata_path: Path
    seed: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
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
        return [image.seed for image in self.images]


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        enhancer: OllamaEnhancer | None = None,
        generator: ImageGenerator | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._enhancer = enhancer or OllamaEnhancer(settings=self._settings)
        self._generator = generator or ImageGenerator(settings=self._settings)

    def apply_style(self, prompt: str, model: str | None = None) -> str:
        _, profile = self._settings.image_model(model)
        body = prompt.strip().rstrip(",").strip()
        for suffix in (STYLE_SUFFIX, profile.prompt_suffix):
            content = suffix.lstrip(", ").lower()
            if content and content not in body.lower():
                body += suffix
        return body

    def run(
        self,
        prompt: str,
        images: int = 1,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        model: str | None = None,
        use_llm: bool = True,
    ) -> RunResult:
        user_prompt = prompt.strip()
        if not user_prompt:
            raise ValueError("prompt must not be empty")
        model_key, _ = self._settings.image_model(model)
        self._settings.outputs_dir.mkdir(parents=True, exist_ok=True)

        run_started = time.perf_counter()
        if use_llm:
            enhancement = self._enhancer.enhance(user_prompt)
        else:
            LOGGER.info("LLM enhancement disabled; using the raw prompt")
            enhancement = EnhancedPrompt(original=user_prompt, enhanced=user_prompt)
        styled_prompt = self.apply_style(enhancement.enhanced, model_key)
        LOGGER.info("Final image-model prompt: %s", styled_prompt)

        sd_started = time.perf_counter()
        results = self._generator.generate(
            prompt=styled_prompt,
            seed=seed,
            images=images,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            model=model_key,
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

    def _save(
        self,
        result: GenerationResult,
        enhancement: EnhancedPrompt,
        styled_prompt: str,
        use_llm: bool,
        timestamp: datetime,
        index: int,
    ) -> SavedImage:
        stem = (
            f"{timestamp.strftime(FILENAME_TIMESTAMP)}_seed{result.seed}_{uuid4().hex}"
        )
        image_path = self._settings.outputs_dir / f"{stem}.png"
        metadata_path = image_path.with_suffix(".json")
        _, profile = self._settings.image_model(result.settings["model_key"])
        metadata: dict[str, Any] = {
            "app_version": __version__,
            "created_at": timestamp.isoformat(timespec="seconds"),
            "index": index,
            "seed": result.seed,
            "final_prompt": styled_prompt,
            "style_suffix": f"{STYLE_SUFFIX}{profile.prompt_suffix}",
            "llm_enabled": use_llm,
            "settings": result.settings,
            "timings": {
                "llm_s": round(enhancement.duration_s, 3),
                "render_s": round(result.duration_s, 3),
            },
            **enhancement.as_metadata(),
        }
        sidecar = json.dumps(metadata, indent=2, ensure_ascii=False)
        try:
            result.image.save(image_path)
            metadata_path.write_text(sidecar, encoding="utf-8")
        except Exception:
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
        LOGGER.info("Saved %s", image_path)
        return SavedImage(
            path=image_path,
            metadata_path=metadata_path,
            seed=result.seed,
            metadata=metadata,
        )
