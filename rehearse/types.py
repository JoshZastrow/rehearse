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
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# ───────────────────────────────────────────────────────────────────────────────
# Identity & enums
# ───────────────────────────────────────────────────────────────────────────────


class Phase(StrEnum):
    """Name the three phases of one live rehearsal call."""

    INTAKE = "intake"
    PRACTICE = "practice"
    FEEDBACK = "feedback"


class Speaker(StrEnum):
    """Name the speaker attached to one runtime utterance or audio chunk."""

    USER = "user"
    COACH = "coach"
    CHARACTER = "character"


class ConsentState(StrEnum):
    """Track whether the caller has granted consent for the session."""

    PENDING = "pending"
    GRANTED = "granted"
    DECLINED = "declined"


class ModelProvider(StrEnum):
    """Name the upstream model provider that produced one inference."""

    ANTHROPIC = "anthropic"
    HUME = "hume"
    OPENAI = "openai"
    LOCAL = "local"


class ProsodySource(StrEnum):
    """Describe where a prosody measurement came from."""

    HUME_LIVE = "hume_live"
    SCRIPTED = "scripted"
    TTS_HUME = "tts_hume"
    HUMAN_RECORDED = "human_recorded"


class FaultLabel(StrEnum):
    """List the coaching faults the product and evals care about."""

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


class RubricDimension(StrEnum):
    """Name the rubric dimensions used when scoring sessions and runs."""

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


class ScenarioCategory(StrEnum):
    """Group scenarios into broad conversation categories."""

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
    """Return a new opaque identifier for a stored object."""
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
    gender_preference: str | None = None
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
    """Store timing and budget data for one call phase."""

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


class ParticipantConfig(Strict):
    """Stable identity for one live-call participant."""

    participant_id: str
    role: Literal["caller", "coach", "observer"]
    display_name: str | None = None
    backend: str


class Session(Strict):
    """Index record for a session directory.

    All artifact file paths are stored in `artifact_paths` keyed by logical name.
    The actual per-frame data lives in the referenced files; `Session` is the
    manifest. Canonical keys (all paths relative to session_dir):

    Runtime artifacts (written during the live call):
      "transcript"      → transcript.jsonl
      "prosody"         → prosody.jsonl
      "audio"           → audio.wav  (mixed, 16kHz PCM16)
      "timing"          → timing.jsonl
      "story"           → story.md
      "feedback"        → feedback.md

    Voice training pipeline (written by offline batch jobs):
      "clips"           → pipeline/clips/clips.jsonl
      "enhanced_audio"  → pipeline/enhanced/manifest.jsonl
    """

    id: str = Field(default_factory=_new_id)
    created_at: datetime
    phone_number_hash: str | None = None
    consent: ConsentState = ConsentState.PENDING
    intake: IntakeRecord | None = None
    persona: CounterpartyPersona | None = None
    phase_timings: list[PhaseTiming] = Field(default_factory=list)
    persona_key: str = "default"
    selected_persona_id: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    completion_status: Literal["complete", "partial", "failed", "in_progress"] = "in_progress"
    finalized_at: datetime | None = None
    outcome_label: OutcomeLabel | None = None
    outcome_probe_status: Literal["pending", "asked", "captured", "skipped"] | None = None
    pipeline_version: str | None = None
    model_slots: dict[str, str] = Field(default_factory=dict)
    participants: list[ParticipantConfig] = Field(default_factory=list)


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
    """Store one scored rubric dimension for one run/example pair.

    Spec 1 (v2026-05-06 roadmap) added optional fields. All new fields
    have backwards-compatible defaults so artifacts written before this
    change still deserialize.

    Fields:
      - `modality` — what kind of signal produced the score. `"text"` for
        transcript-only judges, `"audio"` / `"audio+text"` for audio
        judges, `"timing"` for deterministic naturalness scorers, `"meta"`
        for cross-rollout meta-scorers (e.g. stability), `"aggregate"` for
        combined scores like `weighted_reward`.
      - `confidence` — judge-reported confidence ∈ [0, 1]. Optional.
      - `judge_prompt_version` — version string for the prompt or
        deterministic-threshold set that produced this score. Required for
        scores that enter training data; old scores stay valid for
        historical eval runs but cannot mix with new prompt versions.
      - `flags` — surfaces degradations (`"audio_missing"`,
        `"timing_missing"`, `"uncalibrated"`, `"partial_modality"`, etc.)
        Used by the data-card filter to exclude scores from training data.
    """

    run_id: str
    example_id: str
    session_id: str | None = None
    dimension: RubricDimension | str
    value: float
    scorer: Literal["deterministic", "llm_judge", "human"]
    rationale: str | None = None
    modality: Literal[
        "text", "audio", "audio+text", "timing", "meta", "aggregate"
    ] = "text"
    confidence: float | None = None
    judge_prompt_version: str | None = None
    flags: list[str] = Field(default_factory=list)


class EvalRun(Strict):
    """Describe one eval run and where its output artifacts were written."""

    id: str = Field(default_factory=_new_id)
    started_at: datetime
    completed_at: datetime | None = None
    example_ids: list[str]
    pipeline_version: str
    model_slots: dict[str, str]
    results_path: Path
    aggregate_scores: dict[RubricDimension | str, float] = Field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────────
# Training — CLM (coaching model) SFT/DPO types
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
# Training — Voice model (TTS fine-tuning) pipeline types
#
# These types represent the output of the offline audio pipeline that converts
# live session recordings into (audio, text) pairs for TTS fine-tuning.
#
# Pipeline order:
#   AudioRecorder (live) → audio/user/turn_*.wav   [role-switch blocks, 16kHz]
#   vad_segment  (batch) → pipeline/clips/          [per-VAD clips, 16kHz]
#   audio_enhance (batch) → pipeline/enhanced/      [denoised + 24kHz]
# ───────────────────────────────────────────────────────────────────────────────


class AudioClipRecord(Strict):
    """One VAD-segmented user clip produced by the turn-segmentation pipeline.

    Rows are written to `pipeline/clips/clips.jsonl` inside the session directory.
    All paths are relative to the session directory root.

    `timing_turn_index` is the turn_index from timing.jsonl. `start_ms` and
    `end_ms` are relative to the start of the containing `audio/user/turn_N.wav`
    file (not the session wall-clock anchor), so they can be used directly as
    byte offsets: `offset = start_ms / 1000 * 16000 * 2`.
    """

    session_id: str
    clip_index: int
    role_switch_turn: int
    timing_turn_index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    wav_path: str
    text: str | None = None
    transcript_missing: bool = False
    status: Literal["accepted", "rejected_too_short", "rejected_consent"] = "accepted"


class EnhancedClipRecord(Strict):
    """One clip after source separation and bandwidth extension to 24kHz.

    Rows are written to `pipeline/enhanced/manifest.jsonl` inside the session
    directory. All paths are relative to the session directory root.
    """

    session_id: str
    clip_index: int
    source_wav_path: str
    enhanced_wav_path: str
    duration_s: float
    sample_rate: int = 24_000
    dnsmos_ovrl: float
    status: Literal["accepted", "rejected_quality", "rejected_consent"] = "accepted"


class VoiceTrainingRecord(Strict):
    """One (audio, text) pair ready for TTS fine-tuning.

    Produced by joining accepted AudioClipRecords with accepted EnhancedClipRecords.
    This is the final output consumed by the fine-tuning job.
    `wav_path` points to the 24kHz enhanced file; all paths are relative to
    the session directory root.
    """

    session_id: str
    clip_index: int
    wav_path: str
    text: str
    duration_s: float
    dnsmos_ovrl: float
    phone_number_hash: str | None = None


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
