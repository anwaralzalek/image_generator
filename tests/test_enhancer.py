"""Tests for the Ollama prompt enhancer. No Ollama server required."""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from animegen.config import SYSTEM_PROMPT, OllamaSettings, Settings, load_settings
from animegen.llm.enhancer import EnhancedPrompt, OllamaEnhancer

USER_PROMPT = "American teenagers having fun at a party"
MODEL_OUTPUT = (
    "group of six american teenagers dancing, living room house party, "
    "string lights, warm rim lighting, joyful mood, low angle, casual streetwear"
)


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(
        self,
        payload: Any = None,
        status_code: int = 200,
        body: str | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self) -> Any:
        if self._body is not None:
            return json.loads(self._body)  # raises on malformed JSON
        return self._payload


class FakeSession:
    """Records calls and replays a scripted response or exception."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def get(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def settings() -> Settings:
    return load_settings()


def make_enhancer(
    settings: Settings, result: Any
) -> tuple[OllamaEnhancer, FakeSession]:
    session = FakeSession(result)
    return OllamaEnhancer(settings=settings, session=session), session  # type: ignore[arg-type]


def test_successful_enhancement_returns_model_text(settings: Settings) -> None:
    enhancer, session = make_enhancer(
        settings, FakeResponse({"response": MODEL_OUTPUT})
    )

    result = enhancer.enhance(USER_PROMPT)

    assert isinstance(result, EnhancedPrompt)
    assert result.enhanced == MODEL_OUTPUT
    assert result.original == USER_PROMPT
    assert result.used_fallback is False
    assert result.model == "llama3.2:3b"
    assert result.error is None
    assert result.duration_s >= 0.0
    assert len(session.calls) == 1


def test_request_matches_the_ollama_contract(settings: Settings) -> None:
    enhancer, session = make_enhancer(
        settings, FakeResponse({"response": MODEL_OUTPUT})
    )

    enhancer.enhance(USER_PROMPT)
    call = session.calls[0]
    payload = call["json"]

    assert call["url"] == "http://localhost:11434/api/generate"
    assert call["timeout"] == pytest.approx(30.0)
    assert payload["model"] == "llama3.2:3b"
    assert payload["prompt"] == USER_PROMPT
    assert payload["stream"] is False
    assert payload["options"] == {"num_gpu": 0, "temperature": 0.7}
    assert payload["system"] == SYSTEM_PROMPT


def test_llm_never_touches_the_gpu(settings: Settings) -> None:
    enhancer, session = make_enhancer(
        settings, FakeResponse({"response": MODEL_OUTPUT})
    )

    enhancer.enhance(USER_PROMPT)

    assert session.calls[0]["json"]["options"]["num_gpu"] == 0


def test_style_suffix_is_not_applied_by_the_enhancer(settings: Settings) -> None:
    enhancer, _ = make_enhancer(settings, FakeResponse({"response": MODEL_OUTPUT}))

    result = enhancer.enhance(USER_PROMPT)

    assert "2.5D anime style" not in result.enhanced


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"quoted prompt, warm lighting"', "quoted prompt, warm lighting"),
        ("'single quoted, dusk'", "single quoted, dusk"),
        ("  padded prompt, neon  \n", "padded prompt, neon"),
        ("Prompt: teenagers dancing, confetti", "teenagers dancing, confetti"),
        ("**Prompt: teenagers dancing, confetti**", "teenagers dancing, confetti"),
        ("Here is the prompt: rooftop party, sunset", "rooftop party, sunset"),
        ("Here is the prompt:\nrooftop party, sunset", "rooftop party, sunset"),
        ("**bold prompt, candles**", "bold prompt, candles"),
        (
            "first line, is the prompt\n\nchatty explanation",
            "first line, is the prompt",
        ),
        ("trailing period, dusk.", "trailing period, dusk"),
    ],
)
def test_response_cleaning(settings: Settings, raw: str, expected: str) -> None:
    enhancer, _ = make_enhancer(settings, FakeResponse({"response": raw}))

    assert enhancer.enhance(USER_PROMPT).enhanced == expected


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("connection refused"),
        requests.Timeout("timed out after 5s"),
        requests.HTTPError("500 server error"),
    ],
)
def test_transport_failures_fall_back_to_the_raw_prompt(
    settings: Settings, failure: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    enhancer, _ = make_enhancer(settings, failure)

    with caplog.at_level("WARNING"):
        result = enhancer.enhance(USER_PROMPT)

    assert result.used_fallback is True
    assert result.enhanced == USER_PROMPT
    assert result.model is None
    assert type(failure).__name__ in (result.error or "")
    assert "Ollama enhancement failed" in caplog.text


def test_http_error_status_falls_back(settings: Settings) -> None:
    enhancer, _ = make_enhancer(settings, FakeResponse(status_code=503))

    result = enhancer.enhance(USER_PROMPT)

    assert result.used_fallback is True
    assert result.enhanced == USER_PROMPT


def test_malformed_json_falls_back(settings: Settings) -> None:
    enhancer, _ = make_enhancer(settings, FakeResponse(body="{not json"))

    result = enhancer.enhance(USER_PROMPT)

    assert result.used_fallback is True
    assert result.enhanced == USER_PROMPT


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": "shape"},
        {"response": None},
        {"response": "   "},
        {"response": ""},
        {"response": "...,,,"},
        ["not", "a", "mapping"],
    ],
)
def test_unusable_payloads_fall_back(settings: Settings, payload: Any) -> None:
    enhancer, _ = make_enhancer(settings, FakeResponse(payload))

    result = enhancer.enhance(USER_PROMPT)

    assert result.used_fallback is True
    assert result.enhanced == USER_PROMPT


def test_blank_prompt_is_rejected(settings: Settings) -> None:
    enhancer, _ = make_enhancer(settings, FakeResponse({"response": MODEL_OUTPUT}))

    with pytest.raises(ValueError, match="must not be empty"):
        enhancer.enhance("   ")


def test_is_available_reflects_server_state(settings: Settings) -> None:
    up, up_session = make_enhancer(settings, FakeResponse({"models": []}))
    down, _ = make_enhancer(settings, requests.ConnectionError("refused"))

    assert up.is_available() is True
    assert down.is_available() is False
    assert up_session.calls == [
        {"url": "http://localhost:11434/api/tags", "timeout": 2.0}
    ]


def test_metadata_view_is_json_serialisable(settings: Settings) -> None:
    enhancer, _ = make_enhancer(settings, FakeResponse({"response": MODEL_OUTPUT}))

    metadata = enhancer.enhance(USER_PROMPT).as_metadata()

    assert json.loads(json.dumps(metadata))["enhanced_prompt"] == MODEL_OUTPUT
    assert metadata["used_fallback"] is False
    assert metadata["llm_model"] == "llama3.2:3b"


def test_default_session_is_used_when_none_is_injected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: Any, url: str, **kwargs: Any) -> FakeResponse:
        captured.update({"url": url, **kwargs})
        return FakeResponse({"response": MODEL_OUTPUT})

    monkeypatch.setattr(requests.Session, "post", fake_post)

    result = OllamaEnhancer(settings=settings).enhance(USER_PROMPT)

    assert result.enhanced == MODEL_OUTPUT
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["options"]["num_gpu"] == 0


def test_custom_host_from_settings_is_used() -> None:
    settings = Settings(ollama=OllamaSettings(host="http://gpu-box:11434"))
    enhancer, session = make_enhancer(
        settings, FakeResponse({"response": MODEL_OUTPUT})
    )

    enhancer.enhance(USER_PROMPT)

    assert session.calls[0]["url"] == "http://gpu-box:11434/api/generate"
