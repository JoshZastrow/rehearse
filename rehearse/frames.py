"""Define the runtime events that move through the live call bus.

These models are the small shared vocabulary for the owned runtime. Twilio and
Hume adapters publish them, and writers or future phase logic consume them.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from rehearse.types import ConsentState, Phase, ProsodyScores, Speaker, Strict


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
    reason: Literal["budget", "cue", "consent_decline", "llm"]
    ts: float


class EndOfCall(Strict):
    """Carry the final termination reason for a live call."""

    session_id: str
    reason: Literal["hangup", "error", "budget_exceeded", "consent_decline"]
    ts: float


class ConsentResolved(Strict):
    """Carry the resolved consent state once the consent gate has decided."""

    session_id: str
    state: ConsentState
    ts: float


class IntakeComplete(Strict):
    """Signal that IntakeProcessor finished writing intake.json.

    PhaseProcessor awaits this before emitting INTAKE→PRACTICE so the persona
    compiler is guaranteed to read a complete intake.json. On IntakeProcessor
    failure, `error` is set and PhaseProcessor decides whether to proceed.
    """

    session_id: str
    intake_path: str
    error: str | None = None


class ProviderReady(Strict):
    """Signal that the interactive model is connected and ready to converse.

    Published by the interactive backend the moment its websocket to the provider
    is established — which, with the scale-to-zero Modal endpoint, is the end of
    the ~60s cold start (the model server only accepts the socket once it has
    finished loading + warming). The LiveKit datachannel bridge forwards it to the
    browser as ``{"type": "provider_ready"}`` so the UI can leave its warming-up
    state and reveal the live call.
    """

    session_id: str
    ts: float


class SessionStoredEvent(Strict):
    """Signal that a full-duplex session was persisted to the Modal Volume.

    Published by the interactive server's WebSocket client when the server
    emits a ``session_stored`` event. Subscribers annotate the session
    manifest with the Volume path for downstream training data access.
    """

    session_id: str
    volume_path: str  # absolute path inside the Modal Volume mount
    artifacts: list[str]  # filenames present under volume_path
    ts: float


Frame: TypeAlias = (
    AudioChunk
    | TranscriptDelta
    | ProsodyEvent
    | PhaseSignal
    | EndOfCall
    | ConsentResolved
    | IntakeComplete
    | ProviderReady
    | SessionStoredEvent
)
