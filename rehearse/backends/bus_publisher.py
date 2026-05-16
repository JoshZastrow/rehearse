"""BusPublisher — translate Pipecat frame events to Rehearse FrameBus frames.

Called from PipelineBackend processors. Each on_* method receives a
Pipecat frame object and publishes the equivalent Rehearse frame to the bus.
This is the single place where Pipecat vocabulary maps to Rehearse vocabulary.
"""

from __future__ import annotations

import time
import uuid

import structlog

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, TranscriptDelta
from rehearse.types import ProsodyScores, Speaker

log = structlog.get_logger(__name__)


class BusPublisher:
    """Translate Pipecat events into Rehearse FrameBus frames."""

    def __init__(self, *, session_id: str, bus: FrameBus) -> None:
        self._session_id = session_id
        self._bus = bus

    def _uid(self) -> str:
        return uuid.uuid4().hex[:8]

    async def on_transcription(self, frame: object) -> None:
        """Pipecat TranscriptionFrame → TranscriptDelta(USER, is_final=True)."""
        text = getattr(frame, "text", "") or ""
        now = time.time()
        await self._bus.publish(TranscriptDelta(
            session_id=self._session_id,
            utterance_id=self._uid(),
            speaker=Speaker.USER,
            text=text,
            is_final=True,
            ts_start=now,
            ts_end=now,
        ))

    async def on_interim_transcription(self, frame: object) -> None:
        """Pipecat InterimTranscriptionFrame → TranscriptDelta(USER, is_final=False)."""
        text = getattr(frame, "text", "") or ""
        now = time.time()
        await self._bus.publish(TranscriptDelta(
            session_id=self._session_id,
            utterance_id=self._uid(),
            speaker=Speaker.USER,
            text=text,
            is_final=False,
            ts_start=now,
        ))

    async def on_llm_response(self, text: str, utterance_id: str) -> None:
        """LLM full response → TranscriptDelta(COACH, is_final=True)."""
        now = time.time()
        await self._bus.publish(TranscriptDelta(
            session_id=self._session_id,
            utterance_id=utterance_id,
            speaker=Speaker.COACH,
            text=text,
            is_final=True,
            ts_start=now,
            ts_end=now,
        ))

    async def on_bot_audio(self, pcm16_16k: bytes) -> None:
        """Pipecat AudioRawFrame (bot) → AudioChunk(COACH)."""
        await self._bus.publish(AudioChunk(
            session_id=self._session_id,
            speaker=Speaker.COACH,
            pcm16_16k=pcm16_16k,
            ts=time.time(),
        ))

    async def on_user_audio(self, pcm16_16k: bytes) -> None:
        """Pipecat AudioRawFrame (user) → AudioChunk(USER)."""
        await self._bus.publish(AudioChunk(
            session_id=self._session_id,
            speaker=Speaker.USER,
            pcm16_16k=pcm16_16k,
            ts=time.time(),
        ))

    async def on_prosody(self, scores: ProsodyScores, utterance_id: str) -> None:
        """ProsodyService result → ProsodyEvent(USER)."""
        now = time.time()
        await self._bus.publish(ProsodyEvent(
            session_id=self._session_id,
            utterance_id=utterance_id,
            speaker=Speaker.USER,
            scores=scores,
            ts_start=now,
            ts_end=now,
        ))

    async def on_end(self, reason: str = "hangup") -> None:
        """Pipeline EndFrame / ErrorFrame → EndOfCall."""
        await self._bus.publish(EndOfCall(
            session_id=self._session_id,
            reason=reason,  # type: ignore[arg-type]
            ts=time.time(),
        ))
