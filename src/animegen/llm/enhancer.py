"""Optional CPU-only prompt enhancement through Ollama."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from animegen.config import SYSTEM_PROMPT, OllamaSettings, Settings, get_settings

LOGGER = logging.getLogger(__name__)

_STRIPPABLE = " \t\r\n\"'`*"
_PREAMBLE_PREFIXES = (
    "prompt:",
    "here is the prompt:",
    "here's the prompt:",
    "enhanced prompt:",
    "sdxl prompt:",
    "output:",
)


@dataclass(frozen=True)
class EnhancedPrompt:
    original: str
    enhanced: str
    used_fallback: bool = False
    duration_s: float = 0.0
    model: str | None = None
    error: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original,
            "enhanced_prompt": self.enhanced,
            "used_fallback": self.used_fallback,
            "llm_duration_s": round(self.duration_s, 3),
            "llm_model": self.model,
            "llm_error": self.error,
        }


class OllamaEnhancer:
    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._config: OllamaSettings = (settings or get_settings()).ollama
        self._session = session or requests.Session()

    def enhance(self, prompt: str) -> EnhancedPrompt:
        user_prompt = prompt.strip()
        if not user_prompt:
            raise ValueError("prompt must not be empty")

        started = time.perf_counter()
        try:
            raw = self._request(user_prompt)
            enhanced = self._clean(raw)
            if not enhanced:
                raise ValueError("Ollama returned an empty prompt")
        except Exception as exc:  # noqa: BLE001 - the demo must never crash here
            reason = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "Ollama enhancement failed (%s); using the raw prompt", reason
            )
            return EnhancedPrompt(
                original=user_prompt,
                enhanced=user_prompt,
                used_fallback=True,
                duration_s=time.perf_counter() - started,
                error=reason,
            )

        duration = time.perf_counter() - started
        LOGGER.info(
            "Prompt enhanced by %s in %.2fs: %s",
            self._config.model,
            duration,
            enhanced,
        )
        return EnhancedPrompt(
            original=user_prompt,
            enhanced=enhanced,
            used_fallback=False,
            duration_s=duration,
            model=self._config.model,
        )

    def is_available(self) -> bool:
        try:
            response = self._session.get(
                f"{self._config.host}/api/tags",
                timeout=min(self._config.timeout, 2.0),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.debug("Ollama availability check failed: %s", exc)
            return False
        return True

    def _request(self, user_prompt: str) -> str:
        response = self._session.post(
            self._config.generate_url,
            json={
                "model": self._config.model,
                "system": SYSTEM_PROMPT,
                "prompt": user_prompt,
                "stream": False,
                "options": {"num_gpu": 0, "temperature": self._config.temperature},
            },
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(f"unexpected Ollama payload type: {type(data).__name__}")
        text = data.get("response")
        if not isinstance(text, str):
            raise KeyError("Ollama payload has no string 'response' field")
        return text

    @staticmethod
    def _clean(raw: str) -> str:
        for line in raw.splitlines():
            text = line.strip(_STRIPPABLE)
            lowered = text.lower()
            for prefix in _PREAMBLE_PREFIXES:
                if lowered.startswith(prefix):
                    text = text[len(prefix) :]
                    break
            text = text.rstrip(".").strip(_STRIPPABLE)
            if any(character.isalnum() for character in text):
                return text
        return ""
