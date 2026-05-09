"""`CompositeScorer` — runs per-dim scorers and aggregates in one `.score()` call.

The runner protocol calls `Scorer.score(example, rollout, run_id)` per scorer
and collects the rows. `AggregateScorer.aggregate(...)` is a different shape
(takes per-dim rows as input), and the runner doesn't currently wire it in.

This composite plugs the gap: it conforms to the standard `Scorer` protocol,
internally calls each child scorer, then folds the results through an
`AggregateScorer`. Returns per-dim rows + the aggregate row in one shot.

Streaming: when the runner passes a `publish` callable, each child dimension
is published as soon as the child returns it, rather than waiting for the
whole composite to complete. The aggregate is published last.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

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
        publish: Callable[[RubricScore], None] | None = None,
    ) -> list[RubricScore]:
        per_dim: list[RubricScore] = []
        for child in self._children:
            try:
                rows = await _call_scorer(child, example, rollout, run_id)
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
            if publish is not None:
                for row in rows:
                    publish(row)
        agg_rows = list(
            await self._aggregator.aggregate(example, rollout, per_dim, run_id)
        )
        if publish is not None:
            for row in agg_rows:
                publish(row)
        return per_dim + agg_rows


async def _call_scorer(
    scorer: Scorer,
    example: BenchmarkExample,
    rollout: RolloutResult,
    run_id: str,
) -> list[RubricScore]:
    """Call `scorer.score(...)`, tolerating both old (3-arg) and new signatures.

    Composite children don't see the runner's `publish`; they stream via the
    composite's per-row publish above. This wrapper exists so we can extend
    child scorers later without breaking older signatures.
    """
    return await scorer.score(example, rollout, run_id)


def supports_publish(scorer: Scorer) -> bool:
    """Return True if `scorer.score` accepts a `publish` keyword argument."""
    try:
        sig = inspect.signature(scorer.score)
    except (TypeError, ValueError):
        return False
    return "publish" in sig.parameters
