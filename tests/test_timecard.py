"""Verify the per-turn time card built from a live session manifest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rehearse.agents.timecard import (
    HARD_CAP_SECONDS,
    TimeCard,
    build_time_card,
    render_time_card,
)
from rehearse.types import ConsentState, Phase, PhaseTiming, Session

_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)


def _session(*timings: PhaseTiming) -> Session:
    return Session(
        created_at=_T0,
        consent=ConsentState.GRANTED,
        phase_timings=list(timings),
    )


def test_build_time_card_intake_start():
    session = _session(
        PhaseTiming(phase=Phase.INTAKE, started_at=_T0, budget_seconds=60),
    )
    card = build_time_card(session, now=_T0, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.phase == Phase.INTAKE
    assert card.seconds_elapsed_in_phase == 0
    assert card.seconds_remaining_in_phase == 60
    assert card.seconds_remaining_in_call == 300
    assert 15 <= card.word_budget_this_turn <= 80


def test_build_time_card_practice_midway():
    started = _T0 + timedelta(seconds=60)
    session = _session(
        PhaseTiming(
            phase=Phase.INTAKE, started_at=_T0, ended_at=started, budget_seconds=60
        ),
        PhaseTiming(phase=Phase.PRACTICE, started_at=started, budget_seconds=180),
    )
    now = started + timedelta(seconds=90)
    card = build_time_card(session, now=now, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.phase == Phase.PRACTICE
    assert card.seconds_elapsed_in_phase == 90
    assert card.seconds_remaining_in_phase == 90
    assert card.seconds_remaining_in_call == 150


def test_build_time_card_clamps_word_budget_floor():
    started = _T0
    session = _session(
        PhaseTiming(phase=Phase.PRACTICE, started_at=started, budget_seconds=180),
    )
    now = started + timedelta(seconds=178)
    card = build_time_card(session, now=now, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.word_budget_this_turn == 15


def test_build_time_card_after_hard_cap():
    started = _T0
    session = _session(
        PhaseTiming(phase=Phase.FEEDBACK, started_at=started, budget_seconds=60),
    )
    now = started + timedelta(seconds=400)
    card = build_time_card(session, now=now, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.seconds_remaining_in_call == 0
    assert card.seconds_remaining_in_phase == 0


def test_render_time_card_contains_phase_remaining_and_words():
    card = TimeCard(
        phase=Phase.PRACTICE,
        seconds_elapsed_in_phase=42,
        seconds_remaining_in_phase=138,
        seconds_remaining_in_call=198,
        phase_budget_seconds=180,
        word_budget_this_turn=36,
    )
    rendered = render_time_card(card)
    assert "practice" in rendered.lower()
    assert "2:18" in rendered
    assert "3:18" in rendered
    assert "36" in rendered


def test_build_time_card_raises_when_no_open_phase():
    import pytest as _pytest
    started = _T0
    session = _session(
        PhaseTiming(
            phase=Phase.FEEDBACK,
            started_at=started,
            ended_at=started + timedelta(seconds=60),
            budget_seconds=60,
        ),
    )
    with _pytest.raises(ValueError, match="no open phase"):
        build_time_card(session, now=started + timedelta(seconds=70))
