"""InteractiveASR — faster-whisper wrapper for post-hoc user transcription.

Called from InteractiveBackend after each user turn: push_audio() accumulates
raw PCM16 16 kHz chunks; transcribe_and_reset() runs Whisper and clears.
"""
from __future__ import annotations

import numpy as np
import structlog
from faster_whisper import WhisperModel

log = structlog.get_logger(__name__)


class InteractiveASR:
    """Buffer user audio and transcribe on demand."""

    def __init__(self, model_size: str = "base") -> None:
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._chunks: list[np.ndarray] = []

    def push_audio(self, pcm16_16k: bytes) -> None:
        """Accumulate one PCM16 mono 16 kHz chunk into the buffer."""
        arr = np.frombuffer(pcm16_16k, dtype=np.int16).astype(np.float32) / 32768.0
        self._chunks.append(arr)

    def transcribe_and_reset(self) -> str:
        """Transcribe all buffered audio, clear the buffer, return text."""
        if not self._chunks:
            return ""
        audio = np.concatenate(self._chunks)
        self._chunks.clear()
        segments, _ = self._model.transcribe(audio, language="en", vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        log.debug("interactive_asr.transcribed", chars=len(text))
        return text
