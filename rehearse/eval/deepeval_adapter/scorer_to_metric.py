"""Wrap a rehearse `Scorer` as a DeepEval `BaseMetric`.

Lets custom scorers (audio judges, naturalness, deterministic) participate
in DeepEval-driven test runs (`evaluate(...)`, `assert_test(...)`).

The metric needs `score`, `reason`, `success`, `is_successful()` to plug
into the DeepEval evaluator. We compute them by:

  1. Reconstructing a `(BenchmarkExample, RolloutResult)` pair from the
     test case (this is approximate — DeepEval test cases don't carry
     full rollout artifacts; we pass the originals through).
  2. Calling the rehearse `Scorer.score(...)`.
  3. Selecting the `RubricScore` whose `dimension` matches and using its
     `value` as the DeepEval `score`.

For metrics where the test case alone is insufficient (e.g. needs audio
files), pass the originating `RolloutResult` via `bind_rollout(rollout)`
before calling `measure`. The DeepEval evaluator does not expose a way
to thread custom state through arbitrary metrics, so this is the
escape hatch.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from deepeval.metrics import BaseConversationalMetric

from rehearse.eval.protocols import BenchmarkExample, RolloutResult

if TYPE_CHECKING:
    from rehearse.eval.protocols import Scorer


class ScorerToMetric(BaseConversationalMetric):
    """Adapter: rehearse `Scorer` → DeepEval `BaseConversationalMetric`.

    Coaching trajectories are inherently multi-turn, so this adapter
    produces a *conversational* metric — usable with
    `ConversationalTestCase` and DeepEval's `evaluate()` runner. For
    single-turn evals, write a `BaseMetric` subclass directly; we don't
    have a rehearse use case for that today.

    Usage:
        scorer = MMERecognitionScorer()
        metric = ScorerToMetric(scorer, dimension="mme_recognition_accuracy")
        metric.bind_rollout(example, rollout)
        metric.measure(test_case)  # returns scorer's value as float
    """

    def __init__(
        self,
        scorer: Scorer,
        *,
        dimension: str | None = None,
        threshold: float = 0.5,
        run_id: str = "deepeval-runtime",
    ) -> None:
        self.scorer = scorer  # public so DeepEval's metric-cloning sees it
        self.dimension = dimension or getattr(scorer, "dimension", None)
        if self.dimension is None:
            raise ValueError(
                f"scorer {scorer!r} has no `dimension` attr; pass dimension= explicitly"
            )
        self.threshold = threshold
        self.run_id = run_id

        # `bind_rollout` stores the originating (example, rollout) pair on
        # the scorer instance — that way, if DeepEval clones this metric
        # for parallel execution, the clone shares the scorer reference and
        # picks up the same bound state.
        # (Local attrs would be lost during DeepEval's metric cloning.)

        # BaseConversationalMetric attributes (set defaults so DeepEval's
        # evaluator doesn't choke on missing fields).
        self.score: float = 0.0
        self.reason: str | None = None
        self.success: bool = False
        self.error: str | None = None
        self.evaluation_cost: float | None = None
        # Disable DeepEval's parallel execution; we manage concurrency
        # ourselves at the runner level. This also sidesteps DeepEval's
        # metric-reconstruction step which can lose attributes that aren't
        # `__init__` parameters.
        self.async_mode = False

    @property
    def __name__(self) -> str:
        return f"rehearse:{getattr(self.scorer, 'name', type(self.scorer).__name__)}"

    def bind_rollout(self, example: BenchmarkExample, rollout: RolloutResult) -> None:
        """Provide the original (example, rollout) pair before `measure`.

        Required for scorers that need rollout artifacts the DeepEval test
        case doesn't carry (audio, timing, sandbox state). Bound state is
        stored on the scorer object so it survives DeepEval's metric
        cloning during parallel evaluation.
        """
        # Store on the scorer (shared reference across metric clones).
        self.scorer._rehearse_bound_example = example  # type: ignore[attr-defined]
        self.scorer._rehearse_bound_rollout = rollout  # type: ignore[attr-defined]

    def _bound_pair(self) -> tuple[BenchmarkExample, RolloutResult]:
        ex = getattr(self.scorer, "_rehearse_bound_example", None)
        ro = getattr(self.scorer, "_rehearse_bound_rollout", None)
        if ex is None or ro is None:
            raise RuntimeError(
                "ScorerToMetric called before bind_rollout(...). "
                "Custom rehearse scorers need the original RolloutResult."
            )
        return ex, ro

    def measure(self, test_case: Any) -> float:
        example, rollout = self._bound_pair()
        results = asyncio.run(self.scorer.score(example, rollout, self.run_id))
        return self._update_state(results)

    async def a_measure(self, test_case: Any) -> float:
        example, rollout = self._bound_pair()
        results = await self.scorer.score(example, rollout, self.run_id)
        return self._update_state(results)

    def is_successful(self) -> bool:
        return self.success

    def _update_state(self, results: list) -> float:
        # Pick the score for our dimension; fall back to first.
        match = next((r for r in results if str(r.dimension) == self.dimension), None)
        if match is None and results:
            match = results[0]

        if match is None:
            self.score = 0.0
            self.reason = "scorer returned no RubricScore rows"
            self.success = False
            return 0.0

        self.score = float(match.value)
        self.reason = match.rationale
        self.success = self.score >= self.threshold
        return self.score
