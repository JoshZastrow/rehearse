"""Define the runtime events that move through the live call bus.

These models are the small shared vocabulary for the owned runtime. Twilio and
Hume adapters publish them, and writers or future phase logic consume them.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from rehearse.types import Phase, ProsodyScores, Speaker, Strict


class AudioChunk(Strict):
    """Carry one chunk of mono PCM16 audio for one speaker."""

    session_id: str
    speaker: Speaker
    pcm16_16k: bytes
    ts: float


class TranscriptDelta(Strict):
    """Carry one transcript update produced during the live call."""

    session_id: str
    utterance_id: str
    speaker: Speaker
    text: str
    is_final: bool
    ts_start: float
    ts_end: float | None = None


class ProsodyEvent(Strict):
    """Carry one prosody sample aligned to one utterance."""

    session_id: str
    utterance_id: str
    speaker: Speaker
    scores: ProsodyScores
    ts_start: float
    ts_end: float


class PhaseSignal(Strict):
    """Carry one phase transition event on the runtime bus."""

    session_id: str
    from_phase: Phase | None = None
    to_phase: Phase
    reason: Literal["budget", "cue", "consent_decline"]
    ts: float


class EndOfCall(Strict):
    """Carry the final termination reason for a live call."""

    session_id: str
    reason: Literal["hangup", "error", "budget_exceeded"]
    ts: float


Frame: TypeAlias = AudioChunk | TranscriptDelta | ProsodyEvent | PhaseSignal | EndOfCall
