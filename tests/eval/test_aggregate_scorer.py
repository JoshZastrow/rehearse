"""Tests for `AggregateScorer` — the pure weighted-reward aggregator.

Composes per-dimension `RubricScore` rows produced by other scorers in
the same scoring plan and emits a `weighted_reward` row plus an optional
on-disk `judge.json` provenance artifact.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.aggregate import AggregateScorer
from rehearse.types import RubricScore


def _example(payload: dict | None = None) -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-agg-1", benchmark="b", payload=payload or {}
    )


def _rollout(artifacts_dir: Path | None = None) -> RolloutResult:
    return RolloutResult(
        example_id="ex-agg-1",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=artifacts_dir,
    )


def _score(dimension: str, value: float, **extra) -> RubricScore:
    return RubricScore(
        run_id="run-1",
        example_id="ex-agg-1",
        dimension=dimension,
        value=value,
        scorer="llm_judge",
        **extra,
    )


@pytest.mark.asyncio
async def test_aggregate_emits_weighted_reward() -> None:
    scorer = AggregateScorer(weights={"content_quality": 1.0})
    inputs = [_score("content_quality", 0.7)]

    rows = await scorer.aggregate(_example(), _rollout(), inputs, run_id="run-1")

    assert len(rows) == 1
    assert rows[0].dimension == "weighted_reward"
    assert rows[0].value == pytest.approx(0.7)
    assert rows[0].modality == "aggregate"


@pytest.mark.asyncio
async def test_aggregate_computes_weighted_average() -> None:
    scorer = AggregateScorer(
        weights={"affect_perception": 0.35, "delivery_quality": 0.30, "content_quality": 0.35}
    )
    inputs = [
        _score("affect_perception", 0.8),
        _score("delivery_quality", 0.6),
        _score("content_quality", 0.7),
    ]

    rows = await scorer.aggregate(_example(), _rollout(), inputs, run_id="run-1")

    expected = 0.35 * 0.8 + 0.30 * 0.6 + 0.35 * 0.7
    assert rows[0].value == pytest.approx(expected)


@pytest.mark.asyncio
async def test_aggregate_payload_weights_override_defaults() -> None:
    """rubric_weights on the example payload beat the scorer's defaults."""
    scorer = AggregateScorer(
        weights={"content_quality": 0.5, "affect_perception": 0.5}
    )
    payload = {"rubric_weights": {"content_quality": 1.0, "affect_perception": 0.0}}
    inputs = [
        _score("content_quality", 1.0),
        _score("affect_perception", 0.0),
    ]

    rows = await scorer.aggregate(_example(payload=payload), _rollout(), inputs, run_id="run-1")

    assert rows[0].value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_aggregate_renormalizes_when_dimension_missing() -> None:
    """If a configured dimension wasn't scored, weights renormalize over the rest."""
    scorer = AggregateScorer(
        weights={"content_quality": 0.5, "delivery_quality": 0.5}
    )
    inputs = [_score("content_quality", 0.6)]  # delivery_quality missing

    rows = await scorer.aggregate(_example(), _rollout(), inputs, run_id="run-1")

    # weights renormalize: content_quality → 1.0
    assert rows[0].value == pytest.approx(0.6)
    assert "partial_modality" in rows[0].flags


@pytest.mark.asyncio
async def test_aggregate_returns_zero_with_flag_when_no_inputs() -> None:
    scorer = AggregateScorer(weights={"content_quality": 1.0})

    rows = await scorer.aggregate(_example(), _rollout(), [], run_id="run-1")

    assert rows[0].value == 0.0
    assert "no_inputs" in rows[0].flags


@pytest.mark.asyncio
async def test_aggregate_writes_judge_json_to_artifacts_dir(tmp_path: Path) -> None:
    scorer = AggregateScorer(
        weights={"content_quality": 0.6, "affect_perception": 0.4}
    )
    inputs = [
        _score("content_quality", 0.7, judge_prompt_version="content-v1"),
        _score("affect_perception", 0.5, judge_prompt_version="affect-v1"),
    ]
    rollout = _rollout(artifacts_dir=tmp_path)

    await scorer.aggregate(_example(), rollout, inputs, run_id="run-1")

    judge_path = tmp_path / "judge.json"
    assert judge_path.exists()
    artifact = json.loads(judge_path.read_text())
    assert artifact["weighted_reward"] == pytest.approx(0.6 * 0.7 + 0.4 * 0.5)
    assert "weights" in artifact
    assert artifact["weights"]["content_quality"] == pytest.approx(0.6)
    assert artifact["judge_prompt_versions"]["content_quality"] == "content-v1"


@pytest.mark.asyncio
async def test_aggregate_skips_artifact_when_no_artifacts_dir() -> None:
    scorer = AggregateScorer(weights={"content_quality": 1.0})
    inputs = [_score("content_quality", 0.5)]
    # No artifacts_dir on rollout → no judge.json written, no crash
    rows = await scorer.aggregate(_example(), _rollout(), inputs, run_id="run-1")
    assert rows[0].value == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_aggregate_emits_aggregate_modality_and_version() -> None:
    scorer = AggregateScorer(
        weights={"content_quality": 1.0}, version="aggregate-v1"
    )
    inputs = [_score("content_quality", 0.42)]

    rows = await scorer.aggregate(_example(), _rollout(), inputs, run_id="run-1")

    assert rows[0].modality == "aggregate"
    assert rows[0].judge_prompt_version == "aggregate-v1"
    assert rows[0].scorer == "deterministic"  # pure arithmetic
