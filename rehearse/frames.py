"""Runtime frame types carried over the owned live-call bus."""

from __future__ import annotations

from typing import Literal, TypeAlias

from rehearse.types import Phase, ProsodyScores, Speaker, Strict


class AudioChunk(Strict):
    """One PCM16 mono audio chunk on the runtime bus."""

    session_id: str
    speaker: Speaker
    pcm16_16k: bytes
    ts: float


class TranscriptDelta(Strict):
    """One transcript update from Hume EVI."""

    session_id: str
    utterance_id: str
    speaker: Speaker
    text: str
    is_final: bool
    ts_start: float
    ts_end: float | None = None


class ProsodyEvent(Strict):
    """One prosody sample aligned to one utterance."""

    session_id: str
    utterance_id: str
    speaker: Speaker
    scores: ProsodyScores
    ts_start: float
    ts_end: float


class PhaseSignal(Strict):
    """One runtime phase transition event."""

    session_id: str
    from_phase: Phase | None = None
    to_phase: Phase
    reason: Literal["budget", "cue", "consent_decline"]
    ts: float


class EndOfCall(Strict):
    """A terminal event emitted by the live runtime."""

    session_id: str
    reason: Literal["hangup", "error", "budget_exceeded"]
    ts: float


Frame: TypeAlias = AudioChunk | TranscriptDelta | ProsodyEvent | PhaseSignal | EndOfCall
