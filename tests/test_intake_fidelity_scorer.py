"""Tests for IntakeFidelityScorer (Phase 4c)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.intake_fidelity import IntakeFidelityScorer
from rehearse.types import RubricScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = "run-test-001"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _example(expected: dict[str, Any]) -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-001",
        benchmark="voice-rollout-judges",
        payload={},
        expected=expected,
    )


def _rollout(artifacts_dir: Path | None) -> RolloutResult:
    return RolloutResult(
        example_id="ex-001",
        target_name="runtime-sandbox",
        target_version="v1",
        status="ok",
        started_at=_NOW,
        completed_at=_NOW,
        duration_ms=0,
        artifacts_dir=artifacts_dir,
    )


def _write_intake(run_dir: Path, **fields: Any) -> None:
    data = {
        "counterparty_relationship": "manager",
        "stakes": "The user wants a better outcome",
        "user_goal": "Handle the conversation clearly and confidently.",
        **fields,
    }
    (run_dir / "intake.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Match → high score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_returns_high_score(tmp_path: Path) -> None:
    """All expected fields present in intake.json → score ≥ 0.8."""
    _write_intake(tmp_path, counterparty_relationship="manager")
    scorer = IntakeFidelityScorer()
    scores = await scorer.score(
        _example({"intake_relationship": "manager"}),
        _rollout(tmp_path),
        _RUN_ID,
    )
    assert len(scores) == 1
    assert scores[0].value >= 0.8
    assert "intake_missing" not in scores[0].flags


# ---------------------------------------------------------------------------
# Mismatch → low score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mismatch_returns_low_score(tmp_path: Path) -> None:
    """When intake.json has a different relationship, score ≤ 0.3."""
    _write_intake(tmp_path, counterparty_relationship="counterparty")
    scorer = IntakeFidelityScorer()
    scores = await scorer.score(
        _example({"intake_relationship": "manager"}),
        _rollout(tmp_path),
        _RUN_ID,
    )
    assert len(scores) == 1
    assert scores[0].value <= 0.3


# ---------------------------------------------------------------------------
# No expected.intake_* fields → empty list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_expected_fields_returns_empty(tmp_path: Path) -> None:
    """When example has no intake_* expected fields, scorer returns no scores."""
    _write_intake(tmp_path)
    scorer = IntakeFidelityScorer()
    scores = await scorer.score(
        _example({"opening_emotion": "anxious"}),  # unrelated expected field
        _rollout(tmp_path),
        _RUN_ID,
    )
    assert scores == []


# ---------------------------------------------------------------------------
# Missing intake.json → RubricScore with flags=["intake_missing"]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_intake_json_returns_flagged_score(tmp_path: Path) -> None:
    """Missing intake.json → flagged score, not a raised exception."""
    # tmp_path exists but has no intake.json
    scorer = IntakeFidelityScorer()
    scores = await scorer.score(
        _example({"intake_relationship": "manager"}),
        _rollout(tmp_path),
        _RUN_ID,
    )
    assert len(scores) == 1
    assert scores[0].value == 0.0
    assert "intake_missing" in scores[0].flags
    assert scores[0].rationale is not None


# ---------------------------------------------------------------------------
# Partial match (2/3 fields) → intermediate score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_match_intermediate_score(tmp_path: Path) -> None:
    """2/3 expected fields matching → score ≈ 0.67."""
    _write_intake(
        tmp_path,
        counterparty_relationship="manager",
        stakes="The user wants a better outcome",
        user_goal="Handle the conversation clearly.",
    )
    scorer = IntakeFidelityScorer()
    scores = await scorer.score(
        _example({
            "intake_relationship": "manager",        # matches
            "intake_stakes": "better outcome",       # matches
            "intake_user_goal": "negotiate salary",  # doesn't match
        }),
        _rollout(tmp_path),
        _RUN_ID,
    )
    assert len(scores) == 1
    assert 0.5 < scores[0].value < 0.9  # 2/3


# ---------------------------------------------------------------------------
# Regression: monkeypatching _infer_relationship to return "counterparty"
# causes intake_fidelity to drop for s02/s03/s04 seed rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regression_infer_relationship_fallback_drops_score(tmp_path: Path) -> None:
    """If _infer_relationship always returns 'counterparty', score drops for manager row."""
    # Simulate what happens when _infer_relationship reverts to always returning "counterparty"
    _write_intake(tmp_path, counterparty_relationship="counterparty")
    scorer = IntakeFidelityScorer()
    scores = await scorer.score(
        _example({"intake_relationship": "manager"}),  # s02 expects "manager"
        _rollout(tmp_path),
        _RUN_ID,
    )
    assert len(scores) == 1
    assert scores[0].value < 0.5, (
        "intake_fidelity must drop when _infer_relationship produces 'counterparty' "
        "instead of the expected 'manager'"
    )
