"""Unit tests for the GPU-cost → credit formula (rehearse/billing/cost.py).

Asserts the worked table in the plan and the guard rails (missing/negative
finalized_at). Also replays `session_credits()` over any real session manifests
present under `sessions/` so the formula stays valid against production artifacts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rehearse.billing.cost import (
    A10G_USD_PER_HOUR,
    MARKUP,
    USD_PER_CREDIT,
    session_credits,
)
from rehearse.types import ConsentState, Session


def _session(seconds: float) -> Session:
    created = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)
    return Session(
        created_at=created,
        finalized_at=created + timedelta(seconds=seconds),
        phone_number_hash="webrtc",
        consent=ConsentState.PENDING,
    )


@pytest.mark.parametrize(
    "minutes,expected_credits",
    [(2, 6.2), (5, 15.6), (10, 31.2)],
)
def test_worked_table(minutes: float, expected_credits: float) -> None:
    """The plan's Part 2 table: 2/5/10-minute sessions → ~6.2/15.6/31.2 credits."""
    cost = session_credits(_session(minutes * 60))
    assert cost.gpu_seconds == pytest.approx(minutes * 60)
    assert cost.credits == pytest.approx(expected_credits, abs=0.1)


def test_formula_components() -> None:
    """Each intermediate value follows the documented formula exactly."""
    cost = session_credits(_session(300))
    raw = 300 * (A10G_USD_PER_HOUR / 3600)
    assert cost.raw_cost_usd == pytest.approx(raw)
    assert cost.billed_usd == pytest.approx(raw * MARKUP)
    assert cost.credits == pytest.approx(raw * MARKUP / USD_PER_CREDIT)


def test_credits_per_minute_rate() -> None:
    """Sanity: ~3.12 credits per minute (0.0519 credits/sec)."""
    cost = session_credits(_session(60))
    assert cost.credits == pytest.approx(3.12, abs=0.02)


def test_zero_length_session_is_free() -> None:
    cost = session_credits(_session(0))
    assert cost.gpu_seconds == 0.0
    assert cost.credits == 0.0


def test_missing_finalized_at_raises() -> None:
    created = datetime.now(UTC)
    session = Session(created_at=created, consent=ConsentState.PENDING)
    assert session.finalized_at is None
    with pytest.raises(ValueError, match="no finalized_at"):
        session_credits(session)


def test_negative_duration_raises() -> None:
    created = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)
    session = Session(
        created_at=created,
        finalized_at=created - timedelta(seconds=5),
        consent=ConsentState.PENDING,
    )
    with pytest.raises(ValueError, match="precedes"):
        session_credits(session)


def test_replays_over_real_sessions() -> None:
    """`session_credits` runs cleanly on any finalized manifest under sessions/."""
    sessions_root = Path(__file__).parent.parent / "sessions"
    if not sessions_root.exists():
        pytest.skip("no sessions/ directory")

    checked = 0
    for manifest in sessions_root.glob("*/session.json"):
        raw = json.loads(manifest.read_text())
        if raw.get("finalized_at") is None:
            continue
        session = Session.model_validate(raw)
        cost = session_credits(session)
        assert cost.credits >= 0.0
        assert cost.gpu_seconds >= 0.0
        checked += 1

    if checked == 0:
        pytest.skip("no finalized session manifests to replay")
