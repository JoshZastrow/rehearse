"""Tests for `StabilityScorer` — Spec 8 of the v2026-05-06 eval roadmap.

Diagnostic meta-scorer over `repetitions > 1` rollouts of the same example.
Per-dimension stability = `1 - normalized_stddev(values)`, clipped to [0, 1].
"""

from __future__ import annotations

from datetime import datetime

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.stability import StabilityScorer
from rehearse.types import RubricScore


def _example() -> BenchmarkExample:
    return BenchmarkExample(id="ex-1", benchmark="b", payload={})


def _rollout(idx: int, *, temperature: float | None = None) -> RolloutResult:
    payload: dict | None = {"temperature": temperature} if temperature is not None else None
    return RolloutResult(
        example_id="ex-1",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, idx),
        completed_at=datetime(2026, 5, 6, 12, 0, idx + 1),
        duration_ms=1000,
        payload=payload,
    )


def _score(dim: str, value: float) -> RubricScore:
    return RubricScore(
        run_id="run-1",
        example_id="ex-1",
        dimension=dim,
        value=value,
        scorer="llm_judge",
    )


async def test_identical_scores_yield_perfect_stability() -> None:
    rollouts = [_rollout(i) for i in range(3)]
    per_rollout = [[_score("content_quality", 0.7)] for _ in rollouts]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    assert len(rows) == 1
    row = rows[0]
    assert row.dimension == "stability.content_quality"
    assert row.value == pytest.approx(1.0)
    assert row.modality == "meta"
    assert row.scorer == "deterministic"
    assert "stability_unmeasurable" not in row.flags


async def test_high_variance_yields_low_stability() -> None:
    rollouts = [_rollout(i) for i in range(3)]
    per_rollout = [
        [_score("content_quality", 0.1)],
        [_score("content_quality", 0.5)],
        [_score("content_quality", 0.9)],
    ]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    assert rows[0].dimension == "stability.content_quality"
    # mean=0.5, stddev≈0.327, normalized≈0.654, stability≈0.346
    assert 0.30 <= rows[0].value <= 0.40


async def test_per_dimension_rows_emitted() -> None:
    rollouts = [_rollout(i) for i in range(2)]
    per_rollout = [
        [_score("content_quality", 0.7), _score("delivery_quality", 0.5)],
        [_score("content_quality", 0.7), _score("delivery_quality", 0.6)],
    ]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    dims = {r.dimension for r in rows}
    assert dims == {"stability.content_quality", "stability.delivery_quality"}


async def test_single_rollout_emits_unmeasurable_flag() -> None:
    rollouts = [_rollout(0)]
    per_rollout = [[_score("content_quality", 0.7)]]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    assert len(rows) == 1
    row = rows[0]
    assert "stability_unmeasurable" in row.flags
    assert row.value == 0.0


async def test_mixed_temperature_flag_when_temps_differ() -> None:
    rollouts = [_rollout(0, temperature=0.7), _rollout(1, temperature=1.0)]
    per_rollout = [[_score("content_quality", 0.7)], [_score("content_quality", 0.7)]]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    assert "mixed_temperature" in rows[0].flags


async def test_uniform_temperature_no_mixed_flag() -> None:
    rollouts = [_rollout(0, temperature=0.7), _rollout(1, temperature=0.7)]
    per_rollout = [[_score("content_quality", 0.7)], [_score("content_quality", 0.7)]]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    assert "mixed_temperature" not in rows[0].flags


async def test_zero_mean_uses_raw_stddev() -> None:
    """When all scores are 0, normalized_stddev would divide by zero; we
    fall back to raw stddev so identical-zero scores still report 1.0."""
    rollouts = [_rollout(i) for i in range(3)]
    per_rollout = [[_score("content_quality", 0.0)] for _ in rollouts]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    assert rows[0].value == pytest.approx(1.0)


async def test_meta_modality_and_judge_prompt_version() -> None:
    rollouts = [_rollout(i) for i in range(2)]
    per_rollout = [[_score("content_quality", 0.5)] for _ in rollouts]

    rows = await StabilityScorer().score_meta(_example(), rollouts, per_rollout, "run-1")

    assert rows[0].modality == "meta"
    assert rows[0].judge_prompt_version == "stability-v1"


def test_protocol_conformance() -> None:
    from rehearse.eval.protocols import MetaScorer

    assert isinstance(StabilityScorer(), MetaScorer)
