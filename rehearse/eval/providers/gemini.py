"""Gemini audio provider.

The implementation uses optional imports so the offline harness and tests do
not require the Google SDK unless a Gemini run is requested.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rehearse.eval.providers.base import AudioInput, ProviderError, ProviderResponse

_DEFAULT_MODEL = "gemini-2.5-pro"


class GeminiAudioProvider:
    name = "gemini"
    version = "google-genai"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}
        self.model = self.model_slots.get("multimodal_hosted", _DEFAULT_MODEL)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY not set")

    async def complete(
        self,
        audio: AudioInput,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: int = 60,
    ) -> ProviderResponse:
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional SDK
            raise ProviderError("google-genai is not installed") from exc

        client = genai.Client(api_key=self.api_key)
        media_path = Path(audio.path)
        if not media_path.exists():
            raise ProviderError(f"audio/video file not found: {media_path}")

        try:
            uploaded = await _maybe_await(client.aio.files.upload(file=str(media_path)))
            response = await _maybe_await(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=[uploaded, prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - provider-specific
            raise ProviderError(f"gemini call failed: {type(exc).__name__}: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            text=getattr(response, "text", "") or "",
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            audio_duration_s=audio.duration_s,
            raw_finish_reason=_finish_reason(response),
            provider_metadata={"model": self.model},
        )


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return str(getattr(candidates[0], "finish_reason", None))
