"""Tests for `ContentJudgeScorer` — text-side content quality via G-Eval.

The scorer wraps a DeepEval `G-Eval` metric through the `MetricToScorer`
adapter from Spec 0. Tests exercise the wiring (correct dimension,
judge_prompt_version populated, modality="text", DeepEval test case
shape) using a stub metric — no live LLM calls.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.content_judge import ContentJudgeScorer


class _StubGEval:
    """Minimal stand-in for `deepeval.metrics.GEval`. No network."""

    def __init__(self, score: float = 0.7, reason: str = "ok") -> None:
        self.score = score
        self.reason = reason
        self.threshold = 0.5
        self.success = score >= self.threshold
        self.last_test_case: Any = None

    async def a_measure(self, test_case: Any) -> float:
        self.last_test_case = test_case
        return self.score

    def is_successful(self) -> bool:
        return self.success


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-cq-1",
        benchmark="mme-sandbox-rollout",
        payload={
            "scenario": {
                "situation": "ask manager for raise",
                "goal": "get a number commitment",
            }
        },
    )


def _rollout(tmp_path: Path) -> RolloutResult:
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"speaker": "user", "text": "I'm anxious about asking."})
        + "\n"
        + json.dumps({"speaker": "coach", "text": "What's the biggest worry?"})
        + "\n"
        + json.dumps({"speaker": "user", "text": "She'll say no."})
        + "\n"
        + json.dumps({"speaker": "coach", "text": "If she does, what would you do?"})
        + "\n"
    )
    return RolloutResult(
        example_id="ex-cq-1",
        target_name="voice-agent-sandbox",
        target_version="v0",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 8),
        duration_ms=8000,
        artifacts_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_content_judge_emits_content_quality_dimension(tmp_path: Path) -> None:
    metric = _StubGEval(score=0.73)
    scorer = ContentJudgeScorer(metric=metric)

    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")

    assert len(rows) == 1
    assert rows[0].dimension == "content_quality"
    assert rows[0].value == pytest.approx(0.73)


@pytest.mark.asyncio
async def test_content_judge_marks_modality_text(tmp_path: Path) -> None:
    scorer = ContentJudgeScorer(metric=_StubGEval())
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")
    assert rows[0].modality == "text"


@pytest.mark.asyncio
async def test_content_judge_populates_judge_prompt_version(tmp_path: Path) -> None:
    scorer = ContentJudgeScorer(metric=_StubGEval(), prompt_version="content-quality-v1")
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")
    assert rows[0].judge_prompt_version == "content-quality-v1"


@pytest.mark.asyncio
async def test_content_judge_passes_conversational_test_case(tmp_path: Path) -> None:
    metric = _StubGEval()
    scorer = ContentJudgeScorer(metric=metric)
    await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")

    from deepeval.test_case import ConversationalTestCase

    assert isinstance(metric.last_test_case, ConversationalTestCase)
    # Four transcript frames → four turns
    assert len(metric.last_test_case.turns) == 4


@pytest.mark.asyncio
async def test_content_judge_zeroes_when_rollout_failed(tmp_path: Path) -> None:
    scorer = ContentJudgeScorer(metric=_StubGEval(score=0.9))
    rollout = _rollout(tmp_path)
    rollout = rollout.model_copy(update={"status": "error", "error": "boom"})

    rows = await scorer.score(_example(), rollout, run_id="run-1")

    assert rows[0].value == 0.0
    assert "error" in (rows[0].rationale or "")


@pytest.mark.asyncio
async def test_content_judge_zeroes_when_transcript_missing(tmp_path: Path) -> None:
    scorer = ContentJudgeScorer(metric=_StubGEval(score=0.9))
    rollout = _rollout(tmp_path)
    (tmp_path / "transcript.jsonl").unlink()

    rows = await scorer.score(_example(), rollout, run_id="run-1")

    assert rows[0].value == 0.0


def test_content_judge_default_metric_is_geval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor without `metric` builds a real DeepEval G-Eval metric.

    DeepEval validates the API key during model construction, so we provide a
    stub `ANTHROPIC_API_KEY` here. The test only checks the default factory
    wires up a `GEval` instance — no API calls are made.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-stub")
    scorer = ContentJudgeScorer()
    from deepeval.metrics import GEval

    assert isinstance(scorer.metric, GEval)


def test_content_judge_exposes_dimension_attribute() -> None:
    scorer = ContentJudgeScorer(metric=_StubGEval())
    assert scorer.dimension == "content_quality"
    assert scorer.name  # any non-empty name
