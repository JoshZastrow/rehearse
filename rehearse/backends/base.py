"""ConversationBackend protocol and PersonaSpec.

One backend owns the audio conversation loop for one live call.
It receives raw caller audio, runs inference, and publishes typed
frames to a FrameBus. Everything downstream of the bus is backend-agnostic.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, runtime_checkable

from rehearse.bus import FrameBus


class PersonaSpec(TypedDict):
    """Provider-agnostic description of a conversation persona."""

    name: str
    gender: Literal["male", "female"]
    system_prompt: str
    voice_ref: str | None


@runtime_checkable
class ConversationBackend(Protocol):
    """Own or delegate the audio conversation loop for one live call.

    Receives raw caller audio from the telephony layer.
    Emits typed frames to a FrameBus.
    The rest of the runtime is backend-agnostic.
    """

    async def __aenter__(self) -> ConversationBackend:
        """Set up pre-session resources."""
        ...

    async def __aexit__(self, *args: object) -> None:
        """Tear down connections and release resources."""
        ...

    async def start(self, session_id: str, bus: FrameBus) -> None:
        """Connect to backing services and begin processing.

        Launches any internal tasks needed to drive the audio loop.
        Returns immediately; the loop runs in the background.
        """
        ...

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        """Push one chunk of caller audio (PCM16, 16 kHz, mono)."""
        ...

    async def inject_speech(self, text: str) -> None:
        """Speak a deterministic line, bypassing the LLM.

        Fire-and-forget: returns immediately. The caller may barge in;
        backends must handle this gracefully.
        """
        ...

    async def swap_persona(self, persona: PersonaSpec) -> None:
        """Change voice and character prompt at a phase transition."""
        ...

    async def close(self) -> None:
        """Cancel internal tasks and release resources."""
        ...
