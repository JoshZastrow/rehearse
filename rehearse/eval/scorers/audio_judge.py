"""`AudioJudge` — Gemini-backed audio LLM judge primitive.

Mirrors `LLMJudge` (text-only) but accepts one or more audio files
alongside the system+user prompt and parses a structured JSON response.
Used by `AffectPerceptionJudgeScorer` and `DeliveryJudgeScorer` (Spec 2
of the v2026-05-06 roadmap).

Two implementations:

  - `AudioJudge` — production Gemini-backed judge. Constructs a real
    `google.genai` client lazily so test environments without a
    `GEMINI_API_KEY` don't need it.
  - `StubAudioJudge` — deterministic test double; returns a
    pre-configured payload or raises a configured error. Records the
    inputs it received so tests can assert on them.

The judge interface is intentionally narrow: `judge(system, user,
audio_paths) -> dict`. Scorers compose this with their own prompts and
JSON schemas.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK_MODEL = "gemini-2.5-flash"


def _default_model() -> str:
    """Resolve the default judge model at call time.

    Defaults to the Flash tier — ~10–20× cheaper than Pro, still
    audio-capable. Override per-run via `REHEARSE_AUDIO_JUDGE_MODEL=<id>`
    (e.g. `gemini-2.5-pro`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`).
    """
    return os.environ.get("REHEARSE_AUDIO_JUDGE_MODEL", _FALLBACK_MODEL)


class AudioJudgeError(RuntimeError):
    """Raised when an audio-judge call fails or its output cannot be parsed."""


class _AudioClient(Protocol):
    """Minimal client interface — anything that can call a multimodal LLM."""

    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        audio_paths: list[Path],
    ) -> str: ...


class AudioJudge:
    """Audio-aware judge: calls a multimodal LLM with audio + prompt, returns parsed JSON.

    Pass `client=` for tests; omit for production (a real Gemini client is
    constructed on first call).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        client: _AudioClient | None = None,
    ) -> None:
        self.model = model or _default_model()
        self._client = client

    async def judge(
        self,
        *,
        system: str,
        user: str,
        audio_paths: list[Path],
    ) -> dict[str, Any]:
        client = self._client or self._lazy_default_client()
        try:
            text = await client.generate(
                model=self.model,
                system=system,
                user=user,
                audio_paths=list(audio_paths),
            )
        except Exception as exc:
            raise AudioJudgeError(f"audio judge call failed: {exc}") from exc

        return _parse_json_block(text)

    def _lazy_default_client(self) -> _AudioClient:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise AudioJudgeError(
                "GEMINI_API_KEY not set; pass `client=` for tests or set the key for live runs"
            )
        return _GeminiAudioClient(api_key=api_key)


class StubAudioJudge:
    """Deterministic stand-in for `AudioJudge`. No network."""

    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if response is None and error is None:
            response = {}
        if response is not None and error is not None:
            raise ValueError("StubAudioJudge takes either response or error, not both")
        self._response = response
        self._error = error
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_audio_paths: list[Path] = []

    async def judge(
        self,
        *,
        system: str,
        user: str,
        audio_paths: list[Path],
    ) -> dict[str, Any]:
        self.last_system = system
        self.last_user = user
        self.last_audio_paths = list(audio_paths)
        if self._error is not None:
            raise self._error
        assert self._response is not None  # narrowed by ctor invariant
        return self._response


def _parse_json_block(text: str) -> dict[str, Any]:
    """Extract the first JSON object from `text`."""
    if not text:
        raise AudioJudgeError("audio judge returned empty text")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise AudioJudgeError(f"could not parse JSON from audio judge output: {text[:300]!r}")


class _GeminiAudioClient:
    """Real Gemini client adapter. Lazy import; only used for live runs."""

    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key

    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        audio_paths: list[Path],
    ) -> str:
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional SDK
            raise AudioJudgeError("google-genai is not installed") from exc

        client = genai.Client(api_key=self._api_key)
        uploaded = []
        for p in audio_paths:
            if not Path(p).exists():
                raise AudioJudgeError(f"audio file not found: {p}")
            uploaded.append(await _maybe_await(client.aio.files.upload(file=str(p))))

        prompt = f"{system}\n\n{user}" if system else user
        contents: list[Any] = [*uploaded, prompt]

        response = await _maybe_await(
            client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=2048,
                    temperature=0.0,
                ),
            )
        )
        return getattr(response, "text", "") or ""


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
