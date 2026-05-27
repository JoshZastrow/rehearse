"""Noop eval — dataset + scorer for smoke tests."""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.datasets.noop import NoopDataset
from rehearse.eval.protocols import BenchmarkExample, RolloutResult, Scorer
from rehearse.types import RubricScore


class _PassthroughScorer:
    name = "passthrough"
    dimension = "noop_score"

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        value = 1.0 if rollout.status == "ok" else 0.0
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=value,
                scorer="deterministic",
            )
        ]


class NoopEval:
    name = "noop"
    version = "v0"
    supported_environments = frozenset({"echo"})
    preferred_environment = "echo"

    def __init__(self) -> None:
        self.dataset = NoopDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        return [_PassthroughScorer()]

    def rollout_timeout_s(self) -> int:
        return 5
