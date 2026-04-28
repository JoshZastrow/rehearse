"""vLLM OpenAI-compatible audio provider."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from rehearse.eval.providers.base import AudioInput, ProviderError, ProviderResponse

_DEFAULT_MODEL = "gemma-4-e4b"


class VLLMAudioProvider:
    name = "vllm"
    version = "openai-compatible"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}
        self.model = self.model_slots.get("multimodal_open", _DEFAULT_MODEL)
        self.base_url = os.environ.get("VLLM_BASE_URL")
        self.api_key = os.environ.get("VLLM_API_KEY")
        if not self.base_url:
            raise ProviderError("VLLM_BASE_URL not set")
        if not self.api_key:
            raise ProviderError("VLLM_API_KEY not set")

    async def complete(
        self,
        audio: AudioInput,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: int = 60,
    ) -> ProviderResponse:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional SDK
            raise ProviderError("openai is not installed") from exc

        media_path = Path(audio.extracted_audio_path or audio.path)
        if not media_path.exists():
            raise ProviderError(f"audio/video file not found: {media_path}")

        encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_s,
        )

        try:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded,
                                    "format": _audio_format(media_path),
                                },
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:  # pragma: no cover - provider-specific
            raise ProviderError(f"vllm call failed: {type(exc).__name__}: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            text=(choice.message.content if choice and choice.message else "") or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            audio_duration_s=audio.duration_s,
            raw_finish_reason=getattr(choice, "finish_reason", None),
            provider_metadata={"model": self.model, "base_url": self.base_url},
        )


def _audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav"}:
        return suffix
    return "wav"
