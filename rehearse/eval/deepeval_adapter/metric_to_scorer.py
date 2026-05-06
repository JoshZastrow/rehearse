"""Wrap a DeepEval `BaseMetric` as a rehearse `Scorer`.

Lets DeepEval-authored metrics flow through the existing eval runner —
each rollout produces one or more `RubricScore` rows just like a custom
scorer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rehearse.eval.deepeval_adapter.test_case import (
    to_conversational_test_case,
    to_llm_test_case,
)
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

if TYPE_CHECKING:
    from deepeval.metrics import BaseMetric


class MetricToScorer:
    """Adapter: DeepEval `BaseMetric` → rehearse `Scorer`.

    The metric must accept either an `LLMTestCase` (when
    `test_case_kind="llm"`) or a `ConversationalTestCase` (when
    `test_case_kind="conversational"`). The adapter:

      1. Converts `(example, rollout)` to the requested test-case shape.
      2. Calls `metric.a_measure(test_case)` if available, else
         `metric.measure(test_case)` in a thread.
      3. Emits one `RubricScore` carrying `metric.score`,
         `scorer="llm_judge"`, and the metric's reason as rationale.

    On any failure (rollout missing transcript, metric raises) emits a
    `RubricScore(value=0.0)` with the failure reason as rationale.
    """

    def __init__(
        self,
        metric: BaseMetric,
        *,
        dimension: str,
        name: str | None = None,
        test_case_kind: str = "conversational",
    ) -> None:
        if test_case_kind not in {"conversational", "llm"}:
            raise ValueError(
                f"test_case_kind must be 'conversational' or 'llm', got {test_case_kind!r}"
            )
        self._metric = metric
        self.dimension = dimension
        self.name = name or f"deepeval:{type(metric).__name__}"
        self._kind = test_case_kind

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        if rollout.status != "ok":
            return [self._zero(example, run_id, f"rollout {rollout.status}")]

        try:
            test_case = (
                to_conversational_test_case(example, rollout)
                if self._kind == "conversational"
                else to_llm_test_case(example, rollout)
            )
        except ValueError as exc:
            return [self._zero(example, run_id, str(exc))]

        try:
            await self._metric.a_measure(test_case)
        except Exception as exc:  # noqa: BLE001 — adapter must not crash the run
            return [self._zero(example, run_id, f"metric {self.name} crashed: {exc}")]

        score = float(getattr(self._metric, "score", 0.0) or 0.0)
        rationale = getattr(self._metric, "reason", None)

        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=score,
                scorer="llm_judge",
                rationale=rationale,
            )
        ]

    def _zero(self, example: BenchmarkExample, run_id: str, rationale: str) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=0.0,
            scorer="llm_judge",
            rationale=rationale,
        )
