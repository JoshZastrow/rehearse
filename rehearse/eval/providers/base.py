"""Provider protocol for audio-native LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response."""


@dataclass(frozen=True)
class AudioInput:
    path: Path
    duration_s: float
    extracted_audio_path: Path | None = None


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    audio_duration_s: float | None = None
    raw_finish_reason: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AudioLLMProvider(Protocol):
    name: str
    version: str

    async def complete(
        self,
        audio: AudioInput,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: int = 60,
    ) -> ProviderResponse: ...
