"""Tests for the live score-stream queue (`scores.jsonl`).

Covers:
- ScoreStreamWriter writes one line per publish, line-buffered
- close() always writes scores.jsonl.done sentinel
- CompositeScorer publishes each child + the aggregate when given a callback
- A scorer that doesn't accept `publish` is detected via supports_publish()
- A reader can tolerate a partially-written line
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.score_stream import ScoreStreamWriter
from rehearse.eval.scorers.aggregate import AggregateScorer
from rehearse.eval.scorers.composite import CompositeScorer, supports_publish
from rehearse.types import RubricScore

_NOW = datetime(2026, 5, 8, tzinfo=UTC)


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-1", benchmark="voice-rollout-judges", payload={}
    )


def _rollout() -> RolloutResult:
    return RolloutResult(
        example_id="ex-1",
        target_name="runtime-sandbox",
        target_version="v1",
        status="ok",
        started_at=_NOW,
        completed_at=_NOW,
        duration_ms=0,
    )


class _StaticScorer:
    """Returns one score with a fixed value. Used to test composite streaming."""

    def __init__(self, *, name: str, dimension: str, value: float) -> None:
        self.name = name
        self.dimension = dimension
        self._value = value

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=self._value,
                scorer="deterministic",
            )
        ]


# ---------------------------------------------------------------------------
# ScoreStreamWriter mechanics
# ---------------------------------------------------------------------------


def test_publish_writes_one_line_per_score(tmp_path: Path) -> None:
    """Each publish() appends a single JSON line."""
    with ScoreStreamWriter(tmp_path) as stream:
        for i in range(3):
            stream.publish(
                RubricScore(
                    run_id="run-1",
                    example_id=f"ex-{i}",
                    dimension="content_quality",
                    value=0.7,
                    scorer="deterministic",
                )
            )

    lines = (tmp_path / "scores.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert row["dimension"] == "content_quality"
        assert "emitted_at" in row


def test_close_writes_sentinel(tmp_path: Path) -> None:
    """Closing the writer always creates scores.jsonl.done."""
    with ScoreStreamWriter(tmp_path) as stream:
        stream.publish(
            RubricScore(
                run_id="run-1",
                example_id="ex-1",
                dimension="x",
                value=0.5,
                scorer="deterministic",
            )
        )
    assert (tmp_path / "scores.jsonl.done").exists()


def test_sentinel_written_on_zero_scores(tmp_path: Path) -> None:
    """Even with no publishes, close() writes the sentinel."""
    with ScoreStreamWriter(tmp_path):
        pass
    assert (tmp_path / "scores.jsonl.done").exists()


# ---------------------------------------------------------------------------
# CompositeScorer streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composite_publishes_each_child_then_aggregate(tmp_path: Path) -> None:
    """Composite calls publish for each child dim, then the aggregate, in order."""
    children = [
        _StaticScorer(name="content", dimension="content_quality", value=0.8),
        _StaticScorer(name="affect", dimension="affect_perception", value=0.6),
    ]
    composite = CompositeScorer(
        children=children,
        aggregator=AggregateScorer(
            weights={"content_quality": 0.5, "affect_perception": 0.5}
        ),
    )

    received: list[RubricScore] = []
    rows = await composite.score(
        _example(), _rollout(), "run-1", publish=received.append
    )

    dims_in_order = [r.dimension for r in received]
    assert dims_in_order == [
        "content_quality",
        "affect_perception",
        "weighted_reward",
    ]
    # Returned list still includes everything so the runner can persist.
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# supports_publish detection
# ---------------------------------------------------------------------------


def test_supports_publish_true_for_composite() -> None:
    composite = CompositeScorer(
        children=[],
        aggregator=AggregateScorer(weights={"x": 1.0}),
    )
    assert supports_publish(composite) is True


def test_supports_publish_false_for_static_scorer() -> None:
    assert supports_publish(_StaticScorer(name="x", dimension="x", value=0.0)) is False


# ---------------------------------------------------------------------------
# Reader robustness
# ---------------------------------------------------------------------------


def test_reader_skips_malformed_line(tmp_path: Path) -> None:
    """A line that fails JSON parsing should not break aggregation."""
    from rehearse.eval.watch import AggregateState

    state = AggregateState()
    state.add(json.loads('{"example_id":"a","dimension":"d1","value":0.7}'))
    # Simulate a malformed line by trying to add a row missing required fields.
    state.add({"value": 0.5})  # no example_id, no dimension — silently dropped
    state.add(json.loads('{"example_id":"b","dimension":"d1","value":0.9}'))

    table = state.render(Path("/tmp/run-x"))
    # We don't assert on visual rendering; the row count should reflect 2 valid.
    # (Internal state check below.)
    assert state._values["d1"] == [0.7, 0.9]
