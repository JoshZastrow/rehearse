"""TDD tests for SurveyPresenceJudge and SyntheticCaller SURVEY phase extension.

Covers:
  - SyntheticCaller: SURVEY phase prompt exists and routes correctly
  - SurveyPresenceJudge: deterministic path (no LLM) and LLM-judge path
    including rollout artifacts reading, scoring, and error fallback

Spec: docs/specs/v2026-05-13-survey-agent.md §11
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.customers.llm_customer import SyntheticCaller
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.survey_presence_judge import SurveyPresenceJudge
from rehearse.types import Phase, RubricDimension, SurveyQuestion, SurveyRecord, SurveyResponse

# ---------------------------------------------------------------------------
# SyntheticCaller — SURVEY phase
# ---------------------------------------------------------------------------


def test_synthetic_caller_survey_prompt_exists() -> None:
    """SyntheticCaller must have a system prompt for the SURVEY phase."""
    from rehearse.eval.customers.llm_customer import _PHASE_PROMPTS

    assert Phase.SURVEY in _PHASE_PROMPTS, (
        "SURVEY phase missing from _PHASE_PROMPTS — SyntheticCaller will fall "
        "back to the INTAKE prompt and give nonsensical survey responses"
    )


def test_synthetic_caller_survey_prompt_formats() -> None:
    """The SURVEY prompt must render without KeyError for standard scenario fields."""
    caller = SyntheticCaller(
        scenario={
            "situation": "Asking for a raise",
            "goal": "Get a 10% raise",
            "stakes": "Feels undervalued",
            "emotional_state": "Nervous but determined",
        }
    )
    prompt = caller._system_prompt(Phase.SURVEY)  # noqa: SLF001
    assert len(prompt) > 20
    assert "survey" in prompt.lower() or "feedback" in prompt.lower() or "feel" in prompt.lower()


def test_synthetic_caller_survey_prompt_is_distinct_from_feedback() -> None:
    """SURVEY and FEEDBACK prompts must be different strings."""
    from rehearse.eval.customers.llm_customer import _PHASE_PROMPTS

    assert _PHASE_PROMPTS[Phase.SURVEY] != _PHASE_PROMPTS[Phase.FEEDBACK], (
        "SURVEY prompt must be distinct from FEEDBACK — the survey is a "
        "structured question about the call experience, not a debrief"
    )


# ---------------------------------------------------------------------------
# SurveyPresenceJudge fixtures
# ---------------------------------------------------------------------------


_NOW = datetime.now(UTC)


def _make_example(scenario: dict[str, Any] | None = None) -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-001",
        benchmark="voice-rollout-judges",
        payload={
            "scenario": scenario
            or {
                "situation": "Asking for a raise",
                "goal": "Get a 10% raise",
                "stakes": "Feels undervalued",
            }
        },
    )


def _make_rollout(
    tmp_path: Path,
    *,
    survey_record: SurveyRecord | None = None,
    transcript_lines: list[dict[str, Any]] | None = None,
    status: str = "ok",
) -> RolloutResult:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    if survey_record is not None:
        (artifacts / "survey.json").write_text(survey_record.model_dump_json(indent=2))

    if transcript_lines is not None:
        (artifacts / "transcript.jsonl").write_text(
            "\n".join(json.dumps(line) for line in transcript_lines)
        )

    return RolloutResult(
        example_id="ex-001",
        target_name="runtime-sandbox",
        target_version="v0",
        status=status,
        started_at=_NOW,
        completed_at=_NOW,
        duration_ms=1000,
        artifacts_dir=artifacts if status == "ok" else None,
    )


def _captured_survey(*, verbatim: str = "Yes, it was really helpful") -> SurveyRecord:
    now = datetime.now(UTC)
    return SurveyRecord(
        session_id="sess-001",
        generation_method="fallback",
        questions=[
            SurveyQuestion(
                text="How did the rehearsal feel overall? Was it useful?",
                response_type="binary",
                rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
            )
        ],
        responses=[
            SurveyResponse(
                question_text="How did the rehearsal feel overall? Was it useful?",
                response_type="binary",
                rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
                captured=True,
                value=True,
                verbatim=verbatim,
                captured_at=now,
            )
        ],
        started_at=now,
        completed_at=now,
    )


def _skipped_survey() -> SurveyRecord:
    now = datetime.now(UTC)
    return SurveyRecord(
        session_id="sess-001",
        generation_method="fallback",
        questions=[
            SurveyQuestion(
                text="How did the rehearsal feel overall? Was it useful?",
                response_type="binary",
                rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
            )
        ],
        responses=[
            SurveyResponse(
                question_text="How did the rehearsal feel overall? Was it useful?",
                response_type="binary",
                rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
                captured=False,
            )
        ],
        started_at=now,
    )


def _survey_transcript(phase: str = "survey") -> list[dict[str, Any]]:
    return [
        {
            "speaker": "coach",
            "text": "Before we wrap up — how did the rehearsal feel overall? Was it useful?",
            "phase": phase,
            "ts_start": 100.0,
        },
        {
            "speaker": "user",
            "text": "Yes, it was really helpful. I feel much more confident.",
            "phase": phase,
            "ts_start": 105.0,
        },
    ]


# ---------------------------------------------------------------------------
# SurveyPresenceJudge — deterministic path (no LLM)
# ---------------------------------------------------------------------------


def test_survey_presence_judge_importable() -> None:
    from rehearse.eval.scorers.survey_presence_judge import SurveyPresenceJudge  # noqa: F401

    assert callable(SurveyPresenceJudge)


@pytest.mark.asyncio
async def test_survey_presence_judge_zero_when_rollout_failed(
    tmp_path: Path,
) -> None:
    rollout = RolloutResult(
        example_id="ex-001",
        target_name="runtime-sandbox",
        target_version="v0",
        status="error",
        started_at=_NOW,
        completed_at=_NOW,
        duration_ms=0,
        artifacts_dir=None,
    )
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert len(scores) == 1
    assert scores[0].value == 0.0
    assert scores[0].dimension == RubricDimension.SURVEY_RESPONSE_QUALITY


@pytest.mark.asyncio
async def test_survey_presence_judge_zero_when_no_survey_json(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(tmp_path)  # no survey_record written
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert len(scores) == 1
    assert scores[0].value == 0.0
    assert "no survey" in scores[0].rationale.lower()


@pytest.mark.asyncio
async def test_survey_presence_judge_low_score_when_not_captured(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_skipped_survey())
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert len(scores) == 1
    assert scores[0].value <= 0.4


@pytest.mark.asyncio
async def test_survey_presence_judge_mid_score_when_captured_bare(
    tmp_path: Path,
) -> None:
    """Bare yes/no captured (no verbatim richness) → moderate score."""
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey(verbatim="yes"))
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert 0.4 < scores[0].value <= 0.7


@pytest.mark.asyncio
async def test_survey_presence_judge_high_score_when_rich_verbatim(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(
        tmp_path,
        survey_record=_captured_survey(
            verbatim="Yes, the part about slowing down really helped me see what I was doing."
        ),
        transcript_lines=_survey_transcript(),
    )
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].value >= 0.7


@pytest.mark.asyncio
async def test_survey_presence_judge_returns_correct_dimension(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].dimension == RubricDimension.SURVEY_RESPONSE_QUALITY


@pytest.mark.asyncio
async def test_survey_presence_judge_returns_scorer_deterministic_without_llm(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].scorer == "deterministic"


# ---------------------------------------------------------------------------
# SurveyPresenceJudge — LLM-judge path (mocked client)
# ---------------------------------------------------------------------------


@dataclass
class _MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class _MockUsage:
    input_tokens: int = 10
    output_tokens: int = 20


@dataclass
class _MockResponse:
    content: list[_MockTextBlock]
    stop_reason: str = "end_turn"
    usage: _MockUsage = field(default_factory=_MockUsage)


class _MockMessages:
    """Synchronous mock for the Anthropic messages API."""

    def __init__(self, score: float = 0.85, rationale: str = "Good survey exchange.") -> None:
        self._score = score
        self._rationale = rationale

    async def create(self, **kwargs: Any) -> _MockResponse:
        payload = json.dumps(
            {
                "survey_response_quality": {
                    "score": self._score,
                    "rationale": self._rationale,
                }
            }
        )
        return _MockResponse(content=[_MockTextBlock(text=payload)])


class _MockAnthropicClient:
    def __init__(self, score: float = 0.85) -> None:
        self.messages = _MockMessages(score=score)


@pytest.mark.asyncio
async def test_survey_presence_judge_uses_llm_when_client_injected(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(
        tmp_path,
        survey_record=_captured_survey(verbatim="Yes, the pacing feedback was really useful."),
        transcript_lines=_survey_transcript(),
    )
    mock_client = _MockAnthropicClient(score=0.85)
    judge = SurveyPresenceJudge(anthropic_client=mock_client)
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].value == pytest.approx(0.85)
    assert scores[0].scorer == "llm_judge"


@pytest.mark.asyncio
async def test_survey_presence_judge_falls_back_to_deterministic_on_llm_error(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())

    class _BrokenMessages:
        async def create(self, **_: Any) -> None:
            raise RuntimeError("API unavailable")

    class _BrokenClient:
        messages = _BrokenMessages()

    judge = SurveyPresenceJudge(anthropic_client=_BrokenClient())
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    # Must not raise; falls back gracefully
    assert len(scores) == 1
    assert scores[0].value >= 0.0


@pytest.mark.asyncio
async def test_survey_presence_judge_llm_score_clamped_to_unit_interval(
    tmp_path: Path,
) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())
    mock_client = _MockAnthropicClient(score=1.5)  # out of range
    judge = SurveyPresenceJudge(anthropic_client=mock_client)
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert 0.0 <= scores[0].value <= 1.0
