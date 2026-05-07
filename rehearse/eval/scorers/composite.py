"""`CompositeScorer` — runs per-dim scorers and aggregates in one `.score()` call.

The runner protocol calls `Scorer.score(example, rollout, run_id)` per scorer
and collects the rows. `AggregateScorer.aggregate(...)` is a different shape
(takes per-dim rows as input), and the runner doesn't currently wire it in.

This composite plugs the gap: it conforms to the standard `Scorer` protocol,
internally calls each child scorer, then folds the results through an
`AggregateScorer`. Returns per-dim rows + the aggregate row in one shot.
"""

from __future__ import annotations

from rehearse.eval.protocols import BenchmarkExample, RolloutResult, Scorer
from rehearse.eval.scorers.aggregate import AggregateScorer
from rehearse.types import RubricScore


class CompositeScorer:
    name = "composite"
    dimension = "weighted_reward"

    def __init__(
        self,
        *,
        children: list[Scorer],
        aggregator: AggregateScorer,
    ) -> None:
        self._children = list(children)
        self._aggregator = aggregator

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        per_dim: list[RubricScore] = []
        for child in self._children:
            try:
                rows = await child.score(example, rollout, run_id)
            except Exception as exc:
                rows = [
                    RubricScore(
                        run_id=run_id,
                        example_id=example.id,
                        dimension=child.dimension,
                        value=0.0,
                        scorer="deterministic",
                        rationale=f"scorer {child.name} crashed: {exc}",
                    )
                ]
            per_dim.extend(rows)
        agg_rows = await self._aggregator.aggregate(example, rollout, per_dim, run_id)
        return per_dim + list(agg_rows)
