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


class PocketTTSService:
    """Synthesize using Pocket TTS (CPU-only, ~100 M params, no GPU required).

    Pocket TTS natively outputs 24000 Hz float32 audio. This class converts
    that tensor to int16 bytes and resamples to 16000 Hz before returning,
    so callers receive standard PCM16/16kHz matching AudioChunk.pcm16_16k.
    """

    _NATIVE_SAMPLE_RATE = 24_000  # verified: TTSModel.sample_rate == 24000
    _OUTPUT_SAMPLE_RATE = 16_000  # Rehearse AudioChunk requirement

    def __init__(self, voice_ref: str = "alba") -> None:
        self._voice_ref = voice_ref
        self._model: object = None   # pocket_tts.TTSModel — lazy
        self._voice_state: object = None

    async def _load(self) -> None:
        if self._model is None:
            from pocket_tts import TTSModel
            self._model = TTSModel.load_model()
            self._voice_state = self._model.get_state_for_audio_prompt(self._voice_ref)

    async def synthesize(self, text: str) -> bytes:
        """Return PCM16 mono audio at 16 kHz."""
        import numpy as np
        from rehearse.audio.resample import resample_pcm16
        await self._load()
        tensor = self._model.generate_audio(self._voice_state, text)
        audio_np = tensor.numpy()
        int16_np = (audio_np * 32767.0).clip(-32768, 32767).astype(np.int16)
        pcm_native = int16_np.tobytes()
        return resample_pcm16(pcm_native, src_rate=self._NATIVE_SAMPLE_RATE, dst_rate=self._OUTPUT_SAMPLE_RATE)

    async def set_voice(self, voice_ref: str | None) -> None:
        """Switch to a different voice. Takes effect on the next synthesize() call."""
        if not voice_ref or voice_ref == self._voice_ref:
            return
        self._voice_ref = voice_ref
        if self._model is not None:
            self._voice_state = self._model.get_state_for_audio_prompt(voice_ref)
