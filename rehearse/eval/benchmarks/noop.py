"""Noop benchmark — two synthetic examples, one trivial scorer.

Exists to exercise the harness end-to-end without touching any model API.
Used by tests and by `rehearse-eval run --benchmark noop --target echo` as a
smoke test.
"""

from __future__ import annotations

from collections.abc import Iterable

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


class NoopBenchmark:
    name = "noop"
    version = "v0"
    supported_targets = frozenset({"echo"})
    preferred_target = "echo"

    def load(self) -> Iterable[BenchmarkExample]:
        return [
            BenchmarkExample(
                id=f"noop-{i:03d}",
                benchmark=self.name,
                payload={"echo": f"hello-{i}"},
                expected={"echo": f"hello-{i}"},
            )
            for i in range(2)
        ]

    def scoring_plan(self) -> list[Scorer]:
        return [_PassthroughScorer()]

    def rollout_timeout_s(self) -> int:
        return 5
