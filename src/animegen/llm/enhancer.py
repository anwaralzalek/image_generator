"""Prompt enhancement through a local Ollama model.

The enhancer turns a short user idea ("American teenagers having fun at a
party") into a dense, comma-separated SDXL prompt. It runs llama3.2 on the CPU
(``num_gpu: 0``) so the 8 GB of VRAM stay reserved for SDXL, and it degrades
gracefully: if Ollama is down, slow or returns garbage, the caller still gets a
usable prompt instead of a traceback.

The style suffix is intentionally *not* added here -- the orchestrator owns the
style contract so the LLM cannot dilute or reorder it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from animegen.config import OllamaSettings, Settings, get_settings

LOGGER = logging.getLogger(__name__)

#: Characters stripped from both ends of a model response.
_STRIPPABLE = " \t\r\n\"'`*"

#: Prefixes small models like to emit despite being told not to.
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
    """Result of a single enhancement attempt.

    Attributes:
        original: The prompt exactly as typed by the user.
        enhanced: The prompt to hand to SDXL, style suffix not yet applied.
        used_fallback: True when Ollama could not be used and ``enhanced``
            is just the original prompt.
        duration_s: Wall-clock seconds spent in the enhancement stage.
        model: Ollama model that produced the text, or None on fallback.
        error: Short reason for the fallback, or None on success.
    """

    original: str
    enhanced: str
    used_fallback: bool = False
    duration_s: float = 0.0
    model: str | None = None
    error: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the output sidecar."""
        return {
            "original_prompt": self.original,
            "enhanced_prompt": self.enhanced,
            "used_fallback": self.used_fallback,
            "llm_duration_s": round(self.duration_s, 3),
            "llm_model": self.model,
            "llm_error": self.error,
        }


class OllamaEnhancer:
    """Expand a user idea into an SDXL prompt using a local Ollama model.

    Args:
        settings: Application settings; the process-wide settings are used
            when omitted.
        session: Optional ``requests.Session`` for connection reuse.

    Example:
        >>> enhancer = OllamaEnhancer()  # doctest: +SKIP
        >>> enhancer.enhance("teenagers at a party").enhanced  # doctest: +SKIP
        'group of five teenagers dancing, string lights, warm rim lighting, ...'
    """

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._session = session or requests.Session()

    @property
    def config(self) -> OllamaSettings:
        """The Ollama section of the active settings."""
        return self._settings.ollama

    def enhance(self, prompt: str) -> EnhancedPrompt:
        """Enhance ``prompt``, falling back to it verbatim on any failure.

        Args:
            prompt: The raw user idea. Must not be blank.

        Returns:
            An :class:`EnhancedPrompt`. ``used_fallback`` is True whenever the
            LLM was unreachable, too slow, or produced unusable output.

        Raises:
            ValueError: If ``prompt`` is empty or whitespace only.
        """
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
            return self._fallback(user_prompt, exc, time.perf_counter() - started)

        duration = time.perf_counter() - started
        LOGGER.info(
            "Prompt enhanced by %s in %.2fs: %s",
            self.config.model,
            duration,
            enhanced,
        )
        return EnhancedPrompt(
            original=user_prompt,
            enhanced=enhanced,
            used_fallback=False,
            duration_s=duration,
            model=self.config.model,
        )

    def is_available(self) -> bool:
        """Return True when the Ollama server answers within the timeout.

        Used by the CLI/UI for a pre-flight status line; never required for
        :meth:`enhance` to work.
        """
        try:
            response = self._session.get(
                f"{self.config.host}/api/tags", timeout=self.config.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.debug("Ollama availability check failed: %s", exc)
            return False
        return True

    def _payload(self, user_prompt: str) -> dict[str, Any]:
        """Build the ``/api/generate`` request body."""
        return {
            "model": self.config.model,
            "system": self.config.system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                # Keep llama3.2 on the CPU: SDXL owns the GPU.
                "num_gpu": self.config.num_gpu,
                "temperature": self.config.temperature,
            },
        }

    def _request(self, user_prompt: str) -> str:
        """POST to Ollama and return the raw ``response`` field."""
        response = self._session.post(
            self.config.generate_url,
            json=self._payload(user_prompt),
            timeout=self.config.timeout,
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
        """Normalise a model response into a single-line SDXL prompt."""
        text = raw.strip()
        # Small models sometimes answer with several lines; keep the first
        # non-empty one, which is the prompt itself.
        for line in text.splitlines():
            candidate = line.strip()
            if candidate:
                text = candidate
                break
        else:
            return ""

        lowered = text.lower()
        for prefix in _PREAMBLE_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix) :]
                break

        text = text.strip(_STRIPPABLE)
        # Trailing periods add nothing to a comma-separated prompt.
        return text.rstrip(".").strip()

    def _fallback(
        self, user_prompt: str, exc: Exception, duration: float
    ) -> EnhancedPrompt:
        """Log the failure and return the user's prompt unchanged."""
        reason = f"{type(exc).__name__}: {exc}"
        LOGGER.warning(
            "Ollama enhancement failed (%s); using the raw prompt instead", reason
        )
        return EnhancedPrompt(
            original=user_prompt,
            enhanced=user_prompt,
            used_fallback=True,
            duration_s=duration,
            model=None,
            error=reason,
        )
