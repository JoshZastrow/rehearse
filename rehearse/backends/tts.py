"""TTSService protocol and CPU-friendly implementations.

Each implementation accepts a voice_ref at set_voice() and always
delivers PCM16 mono audio at 16 kHz — callers never need to resample.
"""

from __future__ import annotations

import struct
from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSService(Protocol):
    """Synthesize speech from text, returning PCM16 mono at 16 kHz."""

    async def synthesize(self, text: str) -> bytes:
        """Return PCM16 mono audio at 16 kHz."""
        ...

    async def set_voice(self, voice_ref: str | None) -> None:
        """Switch to a different voice.

        voice_ref is backend-specific: a named voice ("alba"), a wav path, or None
        to restore the default. Fire-and-forget — the next synthesize() call uses it.
        """
        ...


class SilenceTTSService:
    """Return silence. For local dev and frame-contract tests that need no model."""

    async def synthesize(self, text: str) -> bytes:
        # 200ms of silence at 16kHz PCM16 = 3200 bytes = 1600 int16 samples
        return struct.pack("<1600h", *([0] * 1600))

    async def set_voice(self, voice_ref: str | None) -> None:
        pass
