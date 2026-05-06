"""Protocol conformance tests for `MetaScorer`.

A meta-scorer operates on a *list* of `RolloutResult`s sharing an
`example_id`, after per-trajectory scorers have run. It's the substrate
for stability scoring (Spec 8); Spec 1 only lands the protocol.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from rehearse.eval.protocols import (
    BenchmarkExample,
    MetaScorer,
    RolloutResult,
)
from rehearse.types import RubricScore


def _example() -> BenchmarkExample:
    return BenchmarkExample(id="ex-1", benchmark="b", payload={})


def _rollout(idx: int) -> RolloutResult:
    return RolloutResult(
        example_id="ex-1",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, idx),
        completed_at=datetime(2026, 5, 6, 12, 0, idx + 1),
        duration_ms=1000,
    )


class _StubMeta:
    """Minimal MetaScorer implementation; structurally typed."""

    name = "stub_meta"

    async def score_meta(
        self,
        example: BenchmarkExample,
        rollouts: list[RolloutResult],
        per_rollout_scores: list[list[RubricScore]],
        run_id: str,
    ) -> list[RubricScore]:
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="meta_demo",
                value=float(len(rollouts)),
                scorer="deterministic",
                modality="meta",
            )
        ]


def test_meta_scorer_protocol_accepts_stub_implementation() -> None:
    assert isinstance(_StubMeta(), MetaScorer)


@pytest.mark.asyncio
async def test_meta_scorer_receives_grouped_rollouts() -> None:
    meta = _StubMeta()
    rollouts = [_rollout(i) for i in range(3)]
    per_rollout: list[list[RubricScore]] = [[] for _ in rollouts]

    rows = await meta.score_meta(_example(), rollouts, per_rollout, "run-1")

    assert len(rows) == 1
    assert rows[0].dimension == "meta_demo"
    assert rows[0].value == pytest.approx(3.0)
    assert rows[0].modality == "meta"


def test_object_without_score_meta_does_not_satisfy_protocol() -> None:
    class _NotMeta:
        name = "fake"

    assert not isinstance(_NotMeta(), MetaScorer)
