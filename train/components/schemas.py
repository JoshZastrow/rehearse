"""
Pydantic schemas for Rehearse session data and annotation output.

Input schemas (session directory):
    Session          — session.json top-level descriptor
    TranscriptUtterance — one line from transcript.jsonl
    ProsodyEntry     — one line from prosody.jsonl

Output schemas (annotation result):
    AnnotationOutput — audio.json produced by annotate.py
                       {"alignments": [["word", [start_sec, end_sec], "SPEAKER_MAIN"], ...]}
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator


# ─── Session input schemas ────────────────────────────────────────────────────


class SessionIntake(BaseModel):
    session_id: str
    situation: str
    counterparty_name: Optional[str] = None
    counterparty_relationship: str
    counterparty_description: str
    stakes: str
    user_goal: str
    desired_tone: Optional[str] = None
    captured_at: datetime


class SessionPersona(BaseModel):
    session_id: str
    name: Optional[str] = None
    relationship: str
    personality_prompt: str
    hot_buttons: list[str]
    likely_reactions: list[str]
    compiled_at: datetime


class PhaseTimings(BaseModel):
    phase: str
    started_at: datetime
    ended_at: datetime
    budget_seconds: int
    overran: bool


class ArtifactPaths(BaseModel):
    transcript: str
    prosody: str
    audio: str
    timing: str
    telemetry: str
    story: str
    feedback: str


class Participant(BaseModel):
    participant_id: str
    role: str
    display_name: Optional[str] = None
    backend: str


class Session(BaseModel):
    """Top-level schema for session.json."""

    id: str
    created_at: datetime
    phone_number_hash: str
    consent: str
    intake: SessionIntake
    persona: SessionPersona
    phase_timings: list[PhaseTimings]
    persona_key: str
    artifact_paths: ArtifactPaths
    completion_status: str
    finalized_at: Optional[datetime] = None
    outcome_label: Optional[str] = None
    outcome_probe_status: Optional[str] = None
    pipeline_version: Optional[str] = None
    model_slots: dict[str, Any]
    participants: list[Participant]


# ─── Transcript schema ────────────────────────────────────────────────────────


class TranscriptUtterance(BaseModel):
    """One utterance line from transcript.jsonl."""

    session_id: str
    utterance_id: str
    ts_start: float
    ts_end: float
    speaker: str
    phase: str
    text: str
    is_interim: bool


# ─── Prosody schema ───────────────────────────────────────────────────────────


class ProsodyScores(BaseModel):
    arousal: float
    valence: float
    dominance: Optional[float] = None
    emotions: dict[str, float]
    speech_rate_wpm: Optional[float] = None
    pause_before_ms: Optional[float] = None


class ProsodyEntry(BaseModel):
    """One line from prosody.jsonl."""

    session_id: str
    utterance_id: str
    ts_start: float
    ts_end: float
    speaker: str
    source: str
    scores: ProsodyScores


# ─── Annotation output schema ─────────────────────────────────────────────────


def _parse_alignment_item(v: Any) -> tuple[str, tuple[float, float], str]:
    """Coerce a raw JSON list [word, [start, end], speaker] into a typed tuple."""
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"alignment item must be [word, [start, end], speaker], got {v!r}")
    text, ts, speaker = v
    if not isinstance(ts, (list, tuple)) or len(ts) != 2:
        raise ValueError(f"timestamp must be [start, end], got {ts!r}")
    return (str(text), (float(ts[0]), float(ts[1])), str(speaker))


AlignmentItem = Annotated[
    tuple[str, tuple[float, float], str],
    BeforeValidator(_parse_alignment_item),
]


class AnnotationOutput(BaseModel):
    """Schema for audio.json produced by annotate.py.

    Each alignment is [word_text, [start_sec, end_sec], speaker_label].

    Example:
        {"alignments": [["Hello", [0.0, 0.4], "SPEAKER_MAIN"], ...]}
    """

    alignments: list[AlignmentItem]
