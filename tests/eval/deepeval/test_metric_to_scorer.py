"""Tests for wrapping a DeepEval `BaseMetric` as a rehearse `Scorer`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.deepeval_adapter import MetricToScorer
from rehearse.eval.protocols import BenchmarkExample, RolloutResult


class _FakeMetric:
    """Minimal stand-in for a DeepEval metric. No network calls."""

    def __init__(self, score: float = 0.8, reason: str = "looked good") -> None:
        self.score = score
        self.reason = reason
        self.threshold = 0.5
        self.success = score >= self.threshold
        self.last_test_case: Any = None

    async def a_measure(self, test_case: Any) -> float:
        self.last_test_case = test_case
        return self.score

    def measure(self, test_case: Any) -> float:
        self.last_test_case = test_case
        return self.score

    def is_successful(self) -> bool:
        return self.success


class _RaisingMetric:
    score = 0.0
    reason = None
    threshold = 0.5

    async def a_measure(self, _tc: Any) -> float:
        raise RuntimeError("boom")


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-7",
        benchmark="mme-sandbox-rollout",
        payload={"scenario": {"situation": "a hard talk"}},
    )


def _rollout(artifacts_dir: Path | None = None, status: str = "ok") -> RolloutResult:
    return RolloutResult(
        example_id="ex-7",
        target_name="voice-agent-sandbox",
        target_version="v0",
        status=status,  # type: ignore[arg-type]
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=artifacts_dir,
    )


def _write_transcript(d: Path) -> None:
    (d / "transcript.jsonl").write_text(
        json.dumps({"speaker": "user", "text": "I'm anxious"})
        + "\n"
        + json.dumps({"speaker": "coach", "text": "say more about that"})
        + "\n"
    )


@pytest.mark.asyncio
async def test_metric_to_scorer_returns_one_rubric_score(tmp_path: Path) -> None:
    _write_transcript(tmp_path)
    metric = _FakeMetric(score=0.73, reason="trajectory was empathetic")
    scorer = MetricToScorer(metric, dimension="content_quality")

    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")

    assert len(rows) == 1
    row = rows[0]
    assert row.dimension == "content_quality"
    assert row.value == pytest.approx(0.73)
    assert row.rationale == "trajectory was empathetic"
    assert row.scorer == "llm_judge"
    assert row.run_id == "run-1"
    assert row.example_id == "ex-7"


@pytest.mark.asyncio
async def test_metric_to_scorer_passes_conversational_test_case_by_default(
    tmp_path: Path,
) -> None:
    _write_transcript(tmp_path)
    metric = _FakeMetric()
    scorer = MetricToScorer(metric, dimension="content_quality")

    await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")

    from deepeval.test_case import ConversationalTestCase

    assert isinstance(metric.last_test_case, ConversationalTestCase)
    assert len(metric.last_test_case.turns) == 2


@pytest.mark.asyncio
async def test_metric_to_scorer_supports_llm_test_case_kind(tmp_path: Path) -> None:
    _write_transcript(tmp_path)
    metric = _FakeMetric()
    scorer = MetricToScorer(metric, dimension="content_quality", test_case_kind="llm")

    await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")

    from deepeval.test_case import LLMTestCase

    assert isinstance(metric.last_test_case, LLMTestCase)
    assert metric.last_test_case.input == "I'm anxious"
    assert metric.last_test_case.actual_output == "say more about that"


@pytest.mark.asyncio
async def test_metric_to_scorer_zeroes_when_rollout_failed() -> None:
    metric = _FakeMetric(score=0.9)
    scorer = MetricToScorer(metric, dimension="content_quality")

    rows = await scorer.score(_example(), _rollout(status="error"), run_id="run-1")

    assert rows[0].value == 0.0
    assert "error" in (rows[0].rationale or "")


@pytest.mark.asyncio
async def test_metric_to_scorer_zeroes_when_transcript_missing(tmp_path: Path) -> None:
    metric = _FakeMetric(score=0.9)
    scorer = MetricToScorer(metric, dimension="content_quality")

    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")

    assert rows[0].value == 0.0
    assert "transcript" in (rows[0].rationale or "").lower()


@pytest.mark.asyncio
async def test_metric_to_scorer_swallows_metric_exceptions(tmp_path: Path) -> None:
    _write_transcript(tmp_path)
    scorer = MetricToScorer(_RaisingMetric(), dimension="content_quality")

    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="run-1")

    assert rows[0].value == 0.0
    assert "boom" in (rows[0].rationale or "")


def test_metric_to_scorer_rejects_invalid_test_case_kind() -> None:
    with pytest.raises(ValueError, match="test_case_kind"):
        MetricToScorer(_FakeMetric(), dimension="x", test_case_kind="bogus")


def test_metric_to_scorer_default_name_uses_metric_class() -> None:
    scorer = MetricToScorer(_FakeMetric(), dimension="content_quality")
    assert scorer.name == "deepeval:_FakeMetric"


def test_metric_to_scorer_explicit_name_overrides_default() -> None:
    scorer = MetricToScorer(_FakeMetric(), dimension="content_quality", name="custom")
    assert scorer.name == "custom"
