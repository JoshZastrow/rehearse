"""Pydantic data contracts.

The single source of truth for every interface in the system. Production artifacts,
eval harness outputs, training corpora, and telemetry all use these types. If a
field exists in one context and not the other, that is a bug.

Grouped by concern:
  - Identity & enums
  - Domain (session runtime + artifacts)
  - Eval (scenarios, synthetic users, rubric)
  - Training (SFT examples, DPO preference pairs)
  - Telemetry (inference + latency instrumentation)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ───────────────────────────────────────────────────────────────────────────────
# Identity & enums
# ───────────────────────────────────────────────────────────────────────────────


class Phase(str, Enum):
    INTAKE = "intake"
    PRACTICE = "practice"
    FEEDBACK = "feedback"


class Speaker(str, Enum):
    USER = "user"
    COACH = "coach"
    CHARACTER = "character"


class ConsentState(str, Enum):
    PENDING = "pending"
    GRANTED = "granted"
    DECLINED = "declined"


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    HUME = "hume"
    OPENAI = "openai"
    LOCAL = "local"


class ProsodySource(str, Enum):
    HUME_LIVE = "hume_live"
    SCRIPTED = "scripted"
    TTS_HUME = "tts_hume"
    HUMAN_RECORDED = "human_recorded"


class FaultLabel(str, Enum):
    # Pure transcript faults
    BURY_LEDE = "bury_lede"
    OVER_JUSTIFY = "over_justify"
    DEFENSIVE_PREEMPTION = "defensive_preemption"
    MONOLOGUE = "monologue"
    MISSING_ASK = "missing_ask"
    SOFT_PAST_TENSE = "soft_past_tense"
    # Prosody-only faults
    FLAT_AFFECT = "flat_affect"
    FALSE_CONFIDENCE_PROSODY = "false_confidence_prosody"
    TRAILING_UNCERTAINTY = "trailing_uncertainty"
    RUSHED_DELIVERY = "rushed_delivery"
    CLIPPED_ANGER = "clipped_anger"
    PERFORMANCE_INCONGRUENCE = "performance_incongruence"
    FALSE_WARMTH = "false_warmth"
    ESCALATING_ANXIETY = "escalating_anxiety"
    UNDER_ENGAGEMENT = "under_engagement"
    EMOTIONAL_WHIPLASH = "emotional_whiplash"
    # Mixed faults
    SAID_VS_MEANT_GAP = "said_vs_meant_gap"
    HEDGING_SPIRAL = "hedging_spiral"


class RubricDimension(str, Enum):
    INTAKE_FIDELITY = "intake_fidelity"
    CHARACTER_PERSONA_FIDELITY = "character_persona_fidelity"
    CHARACTER_BELIEVABILITY = "character_believability"
    FAULT_RECALL = "fault_recall"
    FAULT_PRECISION = "fault_precision"
    FEEDBACK_GROUNDEDNESS = "feedback_groundedness"
    PACING_ADHERENCE = "pacing_adherence"
    INCONGRUENCE_DETECTION = "incongruence_detection"
    PROSODY_CITATION_ACCURACY = "prosody_citation_accuracy"
    USEFULNESS_HOLISTIC = "usefulness_holistic"


class ScenarioCategory(str, Enum):
    RELATIONSHIP_CONFLICT = "relationship_conflict"
    PROFESSIONAL_CONFLICT = "professional_conflict"
    CONNECTION_INFLUENCE = "connection_influence"
    VULNERABILITY = "vulnerability"
    NEGOTIATION = "negotiation"


# ───────────────────────────────────────────────────────────────────────────────
# Base model
# ───────────────────────────────────────────────────────────────────────────────


class Strict(BaseModel):
    """Forbid unknown fields; every contract is explicit."""

    model_config = ConfigDict(extra="forbid", frozen=False, use_enum_values=False)


# ───────────────────────────────────────────────────────────────────────────────
# Domain — session runtime & artifacts
# ───────────────────────────────────────────────────────────────────────────────


def _new_id() -> str:
    return uuid4().hex


class IntakeRecord(Strict):
    """Structured capture of Phase 1 intake."""

    session_id: str
    situation: str
    counterparty_name: str | None = None
    counterparty_relationship: str
    counterparty_description: str
    stakes: str
    user_goal: str
    desired_tone: str | None = None
    captured_at: datetime


class CounterpartyPersona(Strict):
    """Compiled character prompt used to drive the Phase 2 voice."""

    session_id: str
    name: str | None
    relationship: str
    personality_prompt: str
    hot_buttons: list[str] = Field(default_factory=list)
    likely_reactions: list[str] = Field(default_factory=list)
    compiled_at: datetime


class ProsodyScores(Strict):
    """Per-utterance prosody vector.

    `arousal` and `valence` are required aggregates. `emotions` carries the full
    Hume emotion vector (~48 dimensions) keyed by emotion name. `dominance` is
    optional and model-dependent.
    """

    arousal: float
    valence: float
    dominance: float | None = None
    emotions: dict[str, float] = Field(default_factory=dict)
    speech_rate_wpm: float | None = None
    pause_before_ms: float | None = None


class ProsodyFrame(Strict):
    """One prosody sample aligned to one utterance."""

    session_id: str
    utterance_id: str
    ts_start: float
    ts_end: float
    speaker: Speaker
    source: ProsodySource
    scores: ProsodyScores


class TranscriptFrame(Strict):
    """One utterance of text with timing and speaker."""

    session_id: str
    utterance_id: str
    ts_start: float
    ts_end: float
    speaker: Speaker
    phase: Phase
    text: str
    is_interim: bool = False


class PhaseTiming(Strict):
    phase: Phase
    started_at: datetime
    ended_at: datetime | None = None
    budget_seconds: int
    overran: bool = False


class OutcomeLabel(Strict):
    """Captured post-real-conversation. The sparse high-signal label."""

    captured_at: datetime
    did_it_help: bool
    notes: str | None = None


class Session(Strict):
    """Index record for a session directory.

    All artifact file paths are stored in `artifact_paths` keyed by logical name
    (e.g. 'transcript', 'prosody', 'audio', 'story', 'feedback'). The actual
    per-frame data lives in the referenced files; `Session` is the manifest.
    """

    id: str = Field(default_factory=_new_id)
    created_at: datetime
    phone_number_hash: str | None = None
    consent: ConsentState = ConsentState.PENDING
    intake: IntakeRecord | None = None
    persona: CounterpartyPersona | None = None
    phase_timings: list[PhaseTiming] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    completion_status: Literal["complete", "partial", "failed", "in_progress"] = "in_progress"
    outcome_label: OutcomeLabel | None = None
    pipeline_version: str | None = None
    model_slots: dict[str, str] = Field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────────
# Eval — scenarios, synthetic users, rubric
# ───────────────────────────────────────────────────────────────────────────────


class Counterparty(Strict):
    """Counterparty description inside an eval scenario (different from runtime
    `CounterpartyPersona` — this is ground truth input, not compiled output)."""

    name: str
    relationship: str
    personality: str
    hot_buttons: list[str] = Field(default_factory=list)
    likely_reactions: list[str] = Field(default_factory=list)


class SyntheticUserProfile(Strict):
    """Behavior profile for the synthetic user agent in eval.

    `injected_faults` are the ground-truth weaknesses this run is testing for.
    `prosody_trajectory` optionally scripts arousal/valence over utterance index
    for deterministic tier-1 prosody generation.
    """

    speaking_style: str
    injected_faults: list[FaultLabel] = Field(default_factory=list)
    prosody_baseline: ProsodyScores
    prosody_trajectory: dict[str, list[float]] | None = None


class ExampleScenario(Strict):
    """One row of the eval dataset."""

    id: str
    category: ScenarioCategory
    situation: str
    counterparty: Counterparty
    user_goal: str
    synthetic_user: SyntheticUserProfile
    ground_truth_diagnosis: list[str] = Field(default_factory=list)


class RubricScore(Strict):
    run_id: str
    example_id: str
    session_id: str
    dimension: RubricDimension
    value: float
    scorer: Literal["deterministic", "llm_judge", "human"]
    rationale: str | None = None


class EvalRun(Strict):
    id: str = Field(default_factory=_new_id)
    started_at: datetime
    completed_at: datetime | None = None
    example_ids: list[str]
    pipeline_version: str
    model_slots: dict[str, str]
    results_path: Path
    aggregate_scores: dict[RubricDimension, float] = Field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────────
# Training — SFT examples, DPO preference pairs
# ───────────────────────────────────────────────────────────────────────────────


class TrainingExample(Strict):
    """A fully-assembled session ready for SFT / critic training.

    Materialized by the training pipeline from frozen `Session` artifacts plus
    the rubric scores produced by eval. Never written from the live path.
    """

    session_id: str
    category: ScenarioCategory | None = None
    transcript: list[TranscriptFrame]
    prosody: list[ProsodyFrame]
    intake: IntakeRecord
    feedback_text: str
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    outcome_label: OutcomeLabel | None = None
    source: Literal["live", "synthetic"]


class PreferencePair(Strict):
    """One (chosen, rejected) pair for DPO on a specific dimension."""

    id: str = Field(default_factory=_new_id)
    context: str
    chosen: str
    rejected: str
    dimension: RubricDimension
    annotator: Literal["human", "critic_llm", "outcome_weighted"]
    weight: float = 1.0


# ───────────────────────────────────────────────────────────────────────────────
# Telemetry — inference logs & latency
# ───────────────────────────────────────────────────────────────────────────────


class InferenceLogEntry(Strict):
    """One model call. Emitted per LLM or TTS invocation."""

    session_id: str
    ts: datetime
    phase: Phase
    provider: ModelProvider
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int
    stop_reason: str | None = None
    error: str | None = None


class LatencyBreakdown(Strict):
    """Per-turn user-perceived latency breakdown (Phase 2 practice turns)."""

    session_id: str
    turn_id: str
    user_speech_end_ts: float
    first_model_token_ts: float | None = None
    first_tts_audio_ts: float | None = None
    roundtrip_ms: int | None = None
