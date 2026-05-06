"""Tests for wrapping a rehearse `Scorer` as a DeepEval `BaseMetric`."""

from __future__ import annotations

from datetime import datetime

import pytest

from rehearse.eval.deepeval_adapter import ScorerToMetric
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore


class _FakeScorer:
    """Stub Scorer that emits a single fixed RubricScore."""

    name = "fake_scorer"
    dimension = "demo_dimension"

    def __init__(self, value: float = 0.6, rationale: str = "stub reason") -> None:
        self._value = value
        self._rationale = rationale
        self.calls = 0

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        self.calls += 1
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=self._value,
                scorer="deterministic",
                rationale=self._rationale,
            )
        ]


class _MultiDimScorer:
    """Stub Scorer that emits two RubricScores on different dimensions."""

    name = "multi"
    dimension = "primary"

    async def score(self, example, rollout, run_id):  # type: ignore[no-untyped-def]
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="primary",
                value=0.4,
                scorer="deterministic",
                rationale="primary reason",
            ),
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="secondary",
                value=0.9,
                scorer="deterministic",
                rationale="secondary reason",
            ),
        ]


def _example() -> BenchmarkExample:
    return BenchmarkExample(id="ex-9", benchmark="b", payload={})


def _rollout() -> RolloutResult:
    return RolloutResult(
        example_id="ex-9",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 1),
        duration_ms=1000,
    )


@pytest.mark.asyncio
async def test_scorer_to_metric_a_measure_round_trips() -> None:
    scorer = _FakeScorer(value=0.77, rationale="warm and grounded")
    metric = ScorerToMetric(scorer)
    metric.bind_rollout(_example(), _rollout())

    score = await metric.a_measure(test_case=None)

    assert score == pytest.approx(0.77)
    assert metric.score == pytest.approx(0.77)
    assert metric.reason == "warm and grounded"
    assert metric.is_successful() is True  # default threshold 0.5
    assert scorer.calls == 1


@pytest.mark.asyncio
async def test_scorer_to_metric_threshold_controls_success() -> None:
    scorer = _FakeScorer(value=0.3)
    metric = ScorerToMetric(scorer, threshold=0.5)
    metric.bind_rollout(_example(), _rollout())

    await metric.a_measure(test_case=None)

    assert metric.is_successful() is False


def test_scorer_to_metric_measure_runs_async_scorer_synchronously() -> None:
    scorer = _FakeScorer(value=0.42)
    metric = ScorerToMetric(scorer)
    metric.bind_rollout(_example(), _rollout())

    score = metric.measure(test_case=None)

    assert score == pytest.approx(0.42)


def test_scorer_to_metric_requires_bind_rollout() -> None:
    metric = ScorerToMetric(_FakeScorer())
    with pytest.raises(RuntimeError, match="bind_rollout"):
        metric.measure(test_case=None)


@pytest.mark.asyncio
async def test_scorer_to_metric_picks_dimension_when_scorer_is_multi() -> None:
    scorer = _MultiDimScorer()
    metric = ScorerToMetric(scorer, dimension="secondary")
    metric.bind_rollout(_example(), _rollout())

    await metric.a_measure(test_case=None)

    assert metric.score == pytest.approx(0.9)
    assert metric.reason == "secondary reason"


def test_scorer_to_metric_uses_scorer_dimension_by_default() -> None:
    metric = ScorerToMetric(_FakeScorer())
    assert metric.dimension == "demo_dimension"


def test_scorer_to_metric_requires_dimension_when_scorer_lacks_attr() -> None:
    class _BareScorer:
        name = "bare"

        async def score(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return []

    with pytest.raises(ValueError, match="dimension"):
        ScorerToMetric(_BareScorer())  # type: ignore[arg-type]


def test_scorer_to_metric_name_uses_scorer_name() -> None:
    metric = ScorerToMetric(_FakeScorer())
    assert metric.__name__ == "rehearse:fake_scorer"
