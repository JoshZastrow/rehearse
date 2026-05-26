"""Voice participant contracts for live-call actors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from rehearse.bus import FrameBus
from rehearse.types import ParticipantConfig, Strict


class SpeakRequest(Strict):
    """Typed payload for one deterministic speech injection."""

    text: str
    priority: Literal["normal", "interrupt"] = "normal"
    utterance_id: str | None = None


@runtime_checkable
class VoiceSpeaker(Protocol):
    """Capability to inject deterministic speech on a live call."""

    async def say(self, request: SpeakRequest) -> None:
        """Speak request.text verbatim."""
        ...


class VoiceParticipant(ABC):
    """One side of a live voice call."""

    @property
    @abstractmethod
    def config(self) -> ParticipantConfig:
        """Return stable identity for this participant."""
        ...

    @abstractmethod
    async def receive_audio(self, pcm16_16k: bytes) -> None:
        """Accept one PCM16/16kHz audio chunk from another participant."""
        ...

    @abstractmethod
    async def say(self, request: SpeakRequest) -> None:
        """Inject one deterministic utterance without an LLM round-trip."""
        ...

    @abstractmethod
    def audio_stream(self, bus: FrameBus) -> AsyncIterator[bytes]:
        """Yield PCM16/16kHz audio chunks until the call ends."""
        ...

    async def run(self, bus: FrameBus) -> None:
        """Consume audio_stream, discarding chunks (convenience for standalone use)."""
        async for _ in self.audio_stream(bus):
            pass


__all__ = [
    "ParticipantConfig",
    "SpeakRequest",
    "VoiceParticipant",
    "VoiceSpeaker",
]
