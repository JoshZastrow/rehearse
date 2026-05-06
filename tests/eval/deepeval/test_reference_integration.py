"""End-to-end reference tests for the DeepEval adapter.

Two flavors:
  1. Local — runs `deepeval.evaluate()` over our `ScorerToMetric` wrapping
     a custom deterministic scorer. No network calls; serves as the
     reference for "how to plug a rehearse Scorer into a DeepEval test
     run."
  2. Live — `GEval` against a real LLM provider via `MetricToScorer`.
     Marked `live_api`; deselected by default. Requires
     `OPENAI_API_KEY` (or another DeepEval-supported provider) in the
     environment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from rehearse.eval.deepeval_adapter import (
    MetricToScorer,
    ScorerToMetric,
    to_conversational_test_case,
)
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ref-1",
        benchmark="ref-bench",
        payload={
            "scenario": {
                "situation": "ask a colleague to switch a deadline",
                "goal": "agree on a new date without conflict",
                "counterparty_role": "peer",
            }
        },
        expected={"opening_emotion": "tense"},
    )


def _rollout(tmp_path: Path) -> RolloutResult:
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"speaker": "user", "text": "I keep putting this off."})
        + "\n"
        + json.dumps(
            {"speaker": "coach", "text": "What's the worry — the conversation, or the answer?"}
        )
        + "\n"
        + json.dumps({"speaker": "user", "text": "The answer. I think she'll say no."})
        + "\n"
        + json.dumps(
            {"speaker": "coach", "text": "OK. If she said no, what would you do next?"}
        )
        + "\n"
    )
    return RolloutResult(
        example_id="ref-1",
        target_name="voice-agent-sandbox",
        target_version="v0",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 8),
        duration_ms=8000,
        artifacts_dir=tmp_path,
    )


class _CoverageScorer:
    """Deterministic stub: scores higher when the coach asks more questions."""

    name = "question_coverage"
    dimension = "question_coverage"

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        text = (rollout.artifacts_dir / "transcript.jsonl").read_text()  # type: ignore[union-attr]
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        coach_lines = [line["text"] for line in lines if line.get("speaker") == "coach"]
        if not coach_lines:
            value = 0.0
        else:
            value = sum("?" in t for t in coach_lines) / len(coach_lines)
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=value,
                scorer="deterministic",
                rationale=f"{int(value * 100)}% of coach turns asked a question",
            )
        ]


def test_scorer_to_metric_via_deepeval_evaluate(tmp_path: Path) -> None:
    """ScorerToMetric is consumable by `deepeval.evaluate()` end-to-end.

    Demonstrates the canonical pattern: wrap a rehearse scorer, bind the
    rollout, hand the metric to DeepEval's evaluator. Note that DeepEval
    clones metrics during evaluation; read scores from the return value
    rather than the original metric reference.
    """

    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

    rollout = _rollout(tmp_path)
    example = _example()
    test_case = to_conversational_test_case(example, rollout)

    metric = ScorerToMetric(_CoverageScorer(), threshold=0.5)
    metric.bind_rollout(example, rollout)

    result = evaluate(
        test_cases=[test_case],
        metrics=[metric],
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # 2 of 2 coach turns end with "?" → score == 1.0
    assert result.test_results, "evaluate returned no test results"
    metric_data = result.test_results[0].metrics_data[0]
    assert metric_data.score == pytest.approx(1.0)
    assert metric_data.success is True


@pytest.mark.asyncio
async def test_metric_to_scorer_with_stub_metric_emits_rubric_score(tmp_path: Path) -> None:
    """MetricToScorer plus the canonical conversational adapter shape.

    No network: uses a hand-rolled metric to demonstrate the emit path.
    The live G-Eval variant is below, marked `live_api`.
    """

    class _StubMetric:
        score = 0.62
        reason = "trajectory turned defensive in turn 3"
        threshold = 0.5

        async def a_measure(self, _tc):  # type: ignore[no-untyped-def]
            return self.score

        def is_successful(self):
            return self.score >= self.threshold

    scorer = MetricToScorer(_StubMetric(), dimension="content_quality")
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="ref-run")

    assert len(rows) == 1
    assert rows[0].dimension == "content_quality"
    assert rows[0].value == pytest.approx(0.62)
    assert "defensive" in (rows[0].rationale or "")


# ─────────────────────────────────────────────────────────────────────────────
# Live API: real LLM-backed G-Eval. Skipped unless OPENAI_API_KEY is set
# AND pytest is run with `-m live_api`.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.live_api
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_geval_via_metric_to_scorer_live(tmp_path: Path) -> None:
    """G-Eval through MetricToScorer end-to-end against a real LLM."""

    from deepeval.metrics import GEval
    from deepeval.test_case import TurnParams

    metric = GEval(
        name="ContentQuality",
        criteria=(
            "Evaluate whether the coach's responses moved the user toward "
            "clearer, calmer, actionable phrasing for their conversation."
        ),
        evaluation_params=[TurnParams.CONTENT, TurnParams.ROLE],
        threshold=0.5,
    )
    scorer = MetricToScorer(metric, dimension="content_quality")
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="live-ref")

    assert len(rows) == 1
    assert 0.0 <= rows[0].value <= 1.0
    assert rows[0].dimension == "content_quality"
