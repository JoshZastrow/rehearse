"""Data contracts for session curation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class TurningPoint(BaseModel):
    """The span where the caller's stance shifts; anchor for RL credit assignment."""

    ts_start: float
    ts_end: float
    rationale: str


class SessionReview(BaseModel):
    """One curation decision for one prepared session."""

    session_id: str
    decision: Literal["approved", "rejected"]
    turning_point: TurningPoint | None = None
    rationale: str
    criteria_version: int
    audio_duration: float
    model: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
