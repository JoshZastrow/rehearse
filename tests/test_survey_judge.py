"""TDD tests for SurveyPresenceJudge and SyntheticCaller SURVEY phase extension.

Covers:
  - SyntheticCaller: SURVEY phase prompt exists and is distinct from FEEDBACK
  - SurveyPresenceJudge: mocked-LLM unit tests (score tiers, error fallback,
    value clamping) + one real-API smoke test that loads .env and calls Haiku

Spec: docs/specs/v2026-05-13-survey-agent.md §11
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.drivers.llm_customer import SyntheticCaller
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.llm_judge import LLMJudge
from rehearse.eval.scorers.survey_presence_judge import SurveyPresenceJudge
from rehearse.types import Phase, RubricDimension, SurveyQuestion, SurveyRecord, SurveyResponse

# ---------------------------------------------------------------------------
# Mock LLM infrastructure (used by unit tests that don't hit the real API)
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
    def __init__(self, score: float, rationale: str = "Mock rationale.") -> None:
        self._score = score
        self._rationale = rationale

    async def create(self, **_: Any) -> _MockResponse:
        payload = json.dumps(
            {"survey_response_quality": {"score": self._score, "rationale": self._rationale}}
        )
        return _MockResponse(content=[_MockTextBlock(text=payload)])


class _MockAnthropicClient:
    def __init__(self, score: float = 0.8, rationale: str = "Mock rationale.") -> None:
        self.messages = _MockMessages(score=score, rationale=rationale)


def _mock_judge(score: float) -> LLMJudge:
    """Return a LLMJudge backed by a mock client that returns `score`."""
    return LLMJudge(
        model="claude-haiku-4-5-20251001",
        client=_MockAnthropicClient(score=score),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime.now(UTC)


def _make_example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-001",
        benchmark="voice-rollout-judges",
        payload={
            "scenario": {
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
    q = SurveyQuestion(
        text="How did the rehearsal feel overall? Was it useful?",
        response_type="binary",
        rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
    )
    return SurveyRecord(
        session_id="sess-001",
        generation_method="fallback",
        questions=[q],
        responses=[
            SurveyResponse(
                question_text=q.text,
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
    q = SurveyQuestion(
        text="How did the rehearsal feel overall? Was it useful?",
        response_type="binary",
        rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
    )
    return SurveyRecord(
        session_id="sess-001",
        generation_method="fallback",
        questions=[q],
        responses=[
            SurveyResponse(
                question_text=q.text,
                response_type="binary",
                rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
                captured=False,
            )
        ],
        started_at=now,
    )


def _survey_transcript() -> list[dict[str, Any]]:
    return [
        {
            "speaker": "coach",
            "text": "Before we wrap up — how did the rehearsal feel overall? Was it useful?",
            "phase": "survey",
            "ts_start": 100.0,
        },
        {
            "speaker": "user",
            "text": "Yes, it was really helpful. I feel much more confident now.",
            "phase": "survey",
            "ts_start": 105.0,
        },
    ]


# ---------------------------------------------------------------------------
# SyntheticCaller — SURVEY phase prompt
# ---------------------------------------------------------------------------


def test_synthetic_caller_survey_prompt_exists() -> None:
    from rehearse.eval.drivers.llm_customer import _PHASE_PROMPTS

    assert Phase.SURVEY in _PHASE_PROMPTS, (
        "SURVEY phase missing from _PHASE_PROMPTS — SyntheticCaller will fall "
        "back to the INTAKE prompt and give nonsensical survey responses"
    )


def test_synthetic_caller_survey_prompt_formats() -> None:
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
    assert any(
        kw in prompt.lower() for kw in ("survey", "feedback", "feel", "useful", "experience")
    )


def test_synthetic_caller_survey_prompt_is_distinct_from_feedback() -> None:
    from rehearse.eval.drivers.llm_customer import _PHASE_PROMPTS

    assert _PHASE_PROMPTS[Phase.SURVEY] != _PHASE_PROMPTS[Phase.FEEDBACK]


# ---------------------------------------------------------------------------
# SurveyPresenceJudge — unit tests with mocked LLMJudge
# ---------------------------------------------------------------------------


def test_survey_presence_judge_importable() -> None:
    assert callable(SurveyPresenceJudge)


@pytest.mark.asyncio
async def test_survey_presence_judge_zero_when_rollout_failed(tmp_path: Path) -> None:
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
    # No LLM call needed — exits before judging
    judge = SurveyPresenceJudge(judge=_mock_judge(0.0))
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].value == 0.0
    assert scores[0].dimension == RubricDimension.SURVEY_RESPONSE_QUALITY


@pytest.mark.asyncio
async def test_survey_presence_judge_zero_when_no_survey_json(tmp_path: Path) -> None:
    rollout = _make_rollout(tmp_path)  # no survey_record written
    judge = SurveyPresenceJudge(judge=_mock_judge(0.0))
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].value == 0.0
    assert "no survey" in scores[0].rationale.lower()


@pytest.mark.asyncio
async def test_survey_presence_judge_low_score_when_not_captured(tmp_path: Path) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_skipped_survey())
    judge = SurveyPresenceJudge(judge=_mock_judge(0.1))
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].value <= 0.4


@pytest.mark.asyncio
async def test_survey_presence_judge_mid_score_when_captured_bare(tmp_path: Path) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey(verbatim="yes"))
    judge = SurveyPresenceJudge(judge=_mock_judge(0.4))
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert 0.3 <= scores[0].value <= 0.7


@pytest.mark.asyncio
async def test_survey_presence_judge_high_score_when_rich_verbatim(tmp_path: Path) -> None:
    rollout = _make_rollout(
        tmp_path,
        survey_record=_captured_survey(
            verbatim="Yes, the part about slowing down really helped me see what I was doing."
        ),
        transcript_lines=_survey_transcript(),
    )
    judge = SurveyPresenceJudge(judge=_mock_judge(0.9))
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].value >= 0.7


@pytest.mark.asyncio
async def test_survey_presence_judge_returns_correct_dimension(tmp_path: Path) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())
    judge = SurveyPresenceJudge(judge=_mock_judge(0.8))
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].dimension == RubricDimension.SURVEY_RESPONSE_QUALITY


@pytest.mark.asyncio
async def test_survey_presence_judge_scorer_is_llm_judge(tmp_path: Path) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())
    judge = SurveyPresenceJudge(judge=_mock_judge(0.8))
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert scores[0].scorer == "llm_judge"


@pytest.mark.asyncio
async def test_survey_presence_judge_falls_back_on_llm_error(tmp_path: Path) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())

    class _BrokenMessages:
        async def create(self, **_: Any) -> None:
            raise RuntimeError("API unavailable")

    class _BrokenClient:
        messages = _BrokenMessages()

    judge = SurveyPresenceJudge(anthropic_client=_BrokenClient())
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert len(scores) == 1
    assert scores[0].value >= 0.0  # must not raise


@pytest.mark.asyncio
async def test_survey_presence_judge_score_clamped_to_unit_interval(tmp_path: Path) -> None:
    rollout = _make_rollout(tmp_path, survey_record=_captured_survey())
    judge = SurveyPresenceJudge(judge=_mock_judge(1.5))  # mock returns out-of-range
    scores = await judge.score(_make_example(), rollout, run_id="r1")
    assert 0.0 <= scores[0].value <= 1.0


# ---------------------------------------------------------------------------
# Real-API smoke test — skipped unless ANTHROPIC_API_KEY is loadable
# ---------------------------------------------------------------------------


def _load_env_key() -> str | None:
    """Walk up from this file to find .env, then extract ANTHROPIC_API_KEY."""
    # Worktree root and repo root differ; walk up until .env is found.
    candidate = Path(__file__).resolve()
    for _ in range(6):
        candidate = candidate.parent
        env_path = candidate / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("ANTHROPIC_API_KEY")


_HAS_API_KEY = bool(_load_env_key())


@pytest.mark.live_api
@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_API_KEY, reason="ANTHROPIC_API_KEY not available")
async def test_survey_presence_judge_real_llm_scores_captured_response(
    tmp_path: Path,
) -> None:
    """Live call to Haiku: confirm a rich survey response scores >= 0.6."""
    from dotenv import load_dotenv

    _env = next(
        (p / ".env" for p in Path(__file__).resolve().parents if (p / ".env").exists()),
        None,
    )
    if _env:
        load_dotenv(_env, override=True)

    rollout = _make_rollout(
        tmp_path,
        survey_record=_captured_survey(
            verbatim=(
                "Yes, definitely. The coach helped me slow down and stop jumping "
                "straight to solutions. I was able to actually hear what the other "
                "person was saying before I responded."
            )
        ),
        transcript_lines=_survey_transcript(),
    )
    judge = SurveyPresenceJudge()  # uses ANTHROPIC_API_KEY from env
    scores = await judge.score(_make_example(), rollout, run_id="smoke-real")

    assert len(scores) == 1
    score = scores[0]
    print(f"\n[smoke] value={score.value:.2f}  rationale={score.rationale}")
    assert score.scorer == "llm_judge"
    assert score.dimension == RubricDimension.SURVEY_RESPONSE_QUALITY
    assert score.value >= 0.6, f"Expected >= 0.6 for a rich response, got {score.value}"


@pytest.mark.live_api
@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_API_KEY, reason="ANTHROPIC_API_KEY not available")
async def test_survey_presence_judge_real_llm_scores_skipped_survey_low(
    tmp_path: Path,
) -> None:
    """Live call to Haiku: confirm a skipped survey scores < 0.4."""
    from dotenv import load_dotenv

    _env = next(
        (p / ".env" for p in Path(__file__).resolve().parents if (p / ".env").exists()),
        None,
    )
    if _env:
        load_dotenv(_env, override=True)

    rollout = _make_rollout(tmp_path, survey_record=_skipped_survey())
    judge = SurveyPresenceJudge()
    scores = await judge.score(_make_example(), rollout, run_id="smoke-real-skipped")

    score = scores[0]
    print(f"\n[smoke] value={score.value:.2f}  rationale={score.rationale}")
    assert score.value < 0.4, f"Expected < 0.4 for a skipped survey, got {score.value}"
