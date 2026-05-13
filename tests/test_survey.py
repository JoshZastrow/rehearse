"""TDD tests for the post-call survey agent.

Covers:
  - Schema: Phase.SURVEY, SurveyQuestion, SurveyResponse, SurveyRecord, Session.survey_status
  - PhaseBudgets: survey phase, configurable env vars, FEEDBACK default cut to 30s
  - classify_scale: digit and word-map classifier
  - SurveyAgent lifecycle: capture, hangup, skipped, backward-compat OutcomeLabel

Spec: docs/specs/v2026-05-13-survey-agent.md
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall, Frame, PhaseSignal, TranscriptDelta
from rehearse.participants import SpeakRequest
from rehearse.phases import PhaseBudgets
from rehearse.storage import LocalFilesystemStore
from rehearse.survey import (
    SURVEY_CLOSING_LINE,
    SURVEY_QUESTION,
    SurveyAgent,
    SurveyAgentConfig,
    classify_scale,
)
from rehearse.types import (
    ConsentState,
    Phase,
    Session,
    Speaker,
    SurveyQuestion,
    SurveyRecord,
    SurveyResponse,
)

# ---------------------------------------------------------------------------
# Schema tests — fail fast if types or enum values are missing
# ---------------------------------------------------------------------------


def test_phase_enum_has_survey() -> None:
    assert Phase.SURVEY == "survey"


def test_survey_question_schema() -> None:
    q = SurveyQuestion(
        text="How did the conversation feel?",
        response_type="binary",
        rubric_dimension="usefulness_holistic",
    )
    assert q.text == "How did the conversation feel?"
    assert q.response_type == "binary"


def test_survey_response_binary_captured() -> None:
    r = SurveyResponse(
        question_text="Was it useful?",
        response_type="binary",
        rubric_dimension="usefulness_holistic",
        captured=True,
        value=True,
        verbatim="Yes, definitely",
        captured_at=datetime.now(UTC),
    )
    assert r.captured is True
    assert r.value is True


def test_survey_response_not_captured_has_no_value() -> None:
    r = SurveyResponse(
        question_text="Was it useful?",
        response_type="binary",
        rubric_dimension="usefulness_holistic",
        captured=False,
    )
    assert r.captured is False
    assert r.value is None
    assert r.verbatim is None


def test_survey_record_schema() -> None:
    started = datetime.now(UTC)
    q = SurveyQuestion(
        text="How did it go?",
        response_type="open",
        rubric_dimension="usefulness_holistic",
    )
    r = SurveyResponse(
        question_text="How did it go?",
        response_type="open",
        rubric_dimension="usefulness_holistic",
        captured=True,
        value="Really good",
        verbatim="Really good, felt natural",
        captured_at=datetime.now(UTC),
    )
    record = SurveyRecord(
        session_id="abc123",
        generation_method="fallback",
        questions=[q],
        responses=[r],
        started_at=started,
    )
    assert record.session_id == "abc123"
    assert len(record.questions) == 1
    assert len(record.responses) == 1
    assert record.generation_method == "fallback"


def test_session_has_survey_status_field() -> None:
    s = Session(created_at=datetime.now(UTC))
    assert s.survey_status is None


def test_session_survey_status_can_be_set() -> None:
    s = Session(created_at=datetime.now(UTC))
    s.survey_status = "in_progress"
    assert s.survey_status == "in_progress"


# ---------------------------------------------------------------------------
# PhaseBudgets — survey phase and configurable env vars
# ---------------------------------------------------------------------------


def test_phase_budgets_has_survey_seconds() -> None:
    budgets = PhaseBudgets()
    assert budgets.survey_seconds == 30


def test_phase_budgets_for_phase_survey() -> None:
    budgets = PhaseBudgets()
    assert budgets.for_phase(Phase.SURVEY) == 30


def test_phase_budgets_feedback_defaults_to_30s() -> None:
    """FEEDBACK cut from 60s to 30s to make room for the SURVEY phase."""
    budgets = PhaseBudgets()
    assert budgets.feedback_seconds == 30


def test_phase_budgets_from_env_reads_survey(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SURVEY_BUDGET_SECONDS", "15")
    budgets = PhaseBudgets.from_env()
    assert budgets.survey_seconds == 15


def test_phase_budgets_from_env_reads_all_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTAKE_BUDGET_SECONDS", "10")
    monkeypatch.setenv("PRACTICE_BUDGET_SECONDS", "20")
    monkeypatch.setenv("FEEDBACK_BUDGET_SECONDS", "8")
    monkeypatch.setenv("SURVEY_BUDGET_SECONDS", "12")
    budgets = PhaseBudgets.from_env()
    assert budgets.intake_seconds == 10
    assert budgets.practice_seconds == 20
    assert budgets.feedback_seconds == 8
    assert budgets.survey_seconds == 12


def test_phase_budgets_from_env_uses_defaults_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "INTAKE_BUDGET_SECONDS",
        "PRACTICE_BUDGET_SECONDS",
        "FEEDBACK_BUDGET_SECONDS",
        "SURVEY_BUDGET_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    budgets = PhaseBudgets.from_env()
    assert budgets.intake_seconds == 60
    assert budgets.practice_seconds == 120
    assert budgets.feedback_seconds == 30
    assert budgets.survey_seconds == 30


# ---------------------------------------------------------------------------
# classify_scale — digit and word-map classifier
# ---------------------------------------------------------------------------

SCALE_DIGIT_CASES: list[tuple[str, int | None]] = [
    ("1", 1),
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("  3  ", 3),
    ("3.", 3),
    ("0", None),
    ("6", None),
    ("10", None),
    ("", None),
    ("   ", None),
]


@pytest.mark.parametrize("text,expected", SCALE_DIGIT_CASES)
def test_classify_scale_digits(text: str, expected: int | None) -> None:
    assert classify_scale(text) == expected


SCALE_WORD_CASES: list[tuple[str, int | None]] = [
    ("one", 1),
    ("terrible", 1),
    ("bad", 1),
    ("two", 2),
    ("poor", 2),
    ("three", 3),
    ("okay", 3),
    ("ok", 3),
    ("alright", 3),
    ("four", 4),
    ("good", 4),
    ("great", 5),
    ("five", 5),
    ("excellent", 5),
    ("amazing", 5),
    ("perfect", 5),
    ("fantastic", 5),
    # Unclear / ambiguous
    ("maybe three?", 3),
    ("it was great!", 5),
    ("something else entirely", None),
]


@pytest.mark.parametrize("text,expected", SCALE_WORD_CASES)
def test_classify_scale_words(text: str, expected: int | None) -> None:
    assert classify_scale(text) == expected


# ---------------------------------------------------------------------------
# SurveyAgent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def survey_store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC))
    dir_ = store.session_dir(session.id)
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "session.json").write_text(session.model_dump_json(indent=2))
    return store, session.id


def _read_manifest(store: LocalFilesystemStore, session_id: str) -> dict[str, Any]:
    path = store.session_dir(session_id) / "session.json"
    return json.loads(path.read_text())


def _survey_signal(session_id: str) -> PhaseSignal:
    return PhaseSignal(
        session_id=session_id,
        from_phase=Phase.FEEDBACK,
        to_phase=Phase.SURVEY,
        reason="budget",
        ts=1.0,
    )


def _user_frame(session_id: str, text: str, *, ts: float = 2.0) -> TranscriptDelta:
    return TranscriptDelta(
        session_id=session_id,
        utterance_id="u1",
        ts_start=ts,
        speaker=Speaker.USER,
        text=text,
        is_final=True,
    )


class _FakeSpeaker:
    """Capture spoken lines for assertion."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, request: SpeakRequest) -> None:
        self.spoken.append(request.text)


async def _run_agent_with_frames(
    session_id: str,
    store: LocalFilesystemStore,
    frames: list[Frame],
    *,
    speaker: _FakeSpeaker | None = None,
    config: SurveyAgentConfig | None = None,
) -> _FakeSpeaker:
    bus = FrameBus(session_id=session_id)
    sp = speaker or _FakeSpeaker()
    agent = SurveyAgent(session_id, store, speaker=sp, config=config)

    async def _feed() -> None:
        for frame in frames:
            await bus.publish(frame)

    subscription = bus.subscribe()
    await asyncio.gather(agent.run(subscription), _feed())
    return sp


# ---------------------------------------------------------------------------
# SurveyAgent lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_survey_agent_speaks_question_after_survey_signal(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Agent speaks the survey question when SURVEY phase signal arrives."""
    store, sid = survey_store
    frames = [
        _survey_signal(sid),
        _user_frame(sid, "yes"),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    sp = await _run_agent_with_frames(sid, store, frames)
    assert any(SURVEY_QUESTION in line for line in sp.spoken)


@pytest.mark.asyncio
async def test_survey_agent_speaks_closing_line_after_response(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, sid = survey_store
    frames = [
        _survey_signal(sid),
        _user_frame(sid, "yes"),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    sp = await _run_agent_with_frames(sid, store, frames)
    assert any(SURVEY_CLOSING_LINE in line for line in sp.spoken)


@pytest.mark.asyncio
async def test_survey_agent_captures_positive_response(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, sid = survey_store
    frames = [
        _survey_signal(sid),
        _user_frame(sid, "yes it was really helpful"),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames)
    manifest = _read_manifest(store, sid)
    assert manifest["survey_status"] == "complete"


@pytest.mark.asyncio
async def test_survey_agent_writes_survey_json(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, sid = survey_store
    frames = [
        _survey_signal(sid),
        _user_frame(sid, "yes"),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames)
    survey_path = store.session_dir(sid) / "survey.json"
    assert survey_path.exists()
    data = json.loads(survey_path.read_text())
    assert data["session_id"] == sid
    assert len(data["responses"]) >= 1
    assert data["responses"][0]["captured"] is True


@pytest.mark.asyncio
async def test_survey_agent_writes_outcome_label_for_backward_compat(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Final binary question populates OutcomeLabel so training pipeline is unaffected."""
    store, sid = survey_store
    frames = [
        _survey_signal(sid),
        _user_frame(sid, "yes"),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames)
    manifest = _read_manifest(store, sid)
    assert manifest["outcome_label"] is not None
    assert manifest["outcome_label"]["did_it_help"] is True
    assert manifest["outcome_probe_status"] == "captured"


@pytest.mark.asyncio
async def test_survey_agent_writes_negative_outcome_label(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, sid = survey_store
    frames = [
        _survey_signal(sid),
        _user_frame(sid, "no not really"),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames)
    manifest = _read_manifest(store, sid)
    assert manifest["outcome_label"]["did_it_help"] is False


@pytest.mark.asyncio
async def test_survey_agent_marks_skipped_when_no_survey_signal(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    """If SURVEY signal never arrives (e.g. consent declined), agent idles cleanly."""
    store, sid = survey_store
    frames = [EndOfCall(session_id=sid, reason="consent_decline", ts=99.0)]
    await _run_agent_with_frames(sid, store, frames)
    manifest = _read_manifest(store, sid)
    # survey_status stays None — survey phase never started
    assert manifest.get("survey_status") is None


@pytest.mark.asyncio
async def test_survey_agent_marks_partial_on_hangup_after_question_asked(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    """User hangs up after question is spoken but before answering."""
    store, sid = survey_store
    config = SurveyAgentConfig(response_timeout_seconds=60)
    frames = [
        _survey_signal(sid),
        # No user response — caller hangs up immediately
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames, config=config)
    manifest = _read_manifest(store, sid)
    assert manifest["survey_status"] in ("partial", "skipped")


@pytest.mark.asyncio
async def test_survey_agent_marks_outcome_skipped_on_hangup(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, sid = survey_store
    config = SurveyAgentConfig(response_timeout_seconds=60)
    frames = [
        _survey_signal(sid),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames, config=config)
    manifest = _read_manifest(store, sid)
    # outcome_probe_status should reflect that we couldn't capture
    assert manifest.get("outcome_probe_status") in ("skipped", None)


@pytest.mark.asyncio
async def test_survey_agent_ignores_frames_before_survey_signal(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    """User frames that arrive before the SURVEY signal are ignored."""
    store, sid = survey_store
    frames = [
        _user_frame(sid, "yes", ts=0.5),  # before SURVEY signal — should be ignored
        _survey_signal(sid),
        _user_frame(sid, "yes", ts=2.0),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames)
    manifest = _read_manifest(store, sid)
    assert manifest["survey_status"] == "complete"


@pytest.mark.asyncio
async def test_survey_agent_response_timeout_marks_skipped(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    """When no response arrives before the timeout, probe is skipped."""
    store, sid = survey_store
    config = SurveyAgentConfig(response_timeout_seconds=0)  # instant timeout
    frames = [
        _survey_signal(sid),
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames, config=config)
    manifest = _read_manifest(store, sid)
    assert manifest.get("outcome_probe_status") in ("skipped", None)


@pytest.mark.asyncio
async def test_survey_agent_is_idempotent_after_capture(
    survey_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Sending multiple user frames after capture does not overwrite the label."""
    store, sid = survey_store
    frames = [
        _survey_signal(sid),
        _user_frame(sid, "yes", ts=2.0),
        _user_frame(sid, "no wait actually no", ts=3.0),  # second frame should be ignored
        EndOfCall(session_id=sid, reason="hangup", ts=99.0),
    ]
    await _run_agent_with_frames(sid, store, frames)
    manifest = _read_manifest(store, sid)
    assert manifest["outcome_label"]["did_it_help"] is True  # first capture wins


# ---------------------------------------------------------------------------
# LLM-as-judge scorer — just verify the scorer module and function exist
# and return a RubricScore given a completed survey record.
# ---------------------------------------------------------------------------


def test_survey_judge_scorer_importable() -> None:
    from rehearse.eval.scorers.survey_judge import score_survey_response  # noqa: F401

    assert callable(score_survey_response)


def test_survey_judge_scorer_returns_rubric_score() -> None:
    from rehearse.eval.scorers.survey_judge import score_survey_response
    from rehearse.types import RubricScore

    record = SurveyRecord(
        session_id="abc",
        generation_method="fallback",
        questions=[
            SurveyQuestion(
                text="How did it go?",
                response_type="binary",
                rubric_dimension="usefulness_holistic",
            )
        ],
        responses=[
            SurveyResponse(
                question_text="How did it go?",
                response_type="binary",
                rubric_dimension="usefulness_holistic",
                captured=True,
                value=True,
                verbatim="Yes, it really helped me see the issue with my framing.",
                captured_at=datetime.now(UTC),
            )
        ],
        started_at=datetime.now(UTC),
    )
    score = score_survey_response(record, session_id="abc", run_id="test-run")
    assert isinstance(score, RubricScore)
    assert 0.0 <= score.value <= 1.0
    assert score.dimension == "survey_response_quality"
