"""Build and render a per-turn time card injected into the CLM system payload.

The card lets the language model see which phase the call is in, how much time
remains in the phase and the call, and how many words it should aim to speak
on the current turn. It is recomputed for every CLM request from the session
manifest's phase timings plus the current clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rehearse.types import Phase, Session

HARD_CAP_SECONDS: int = 300
"""Hume's per-call model-provider cap. Calls must end before this elapses."""

_AVG_TURN_SECONDS: float = 15.0
_WORDS_PER_SECOND: float = 160.0 / 60.0
_WORD_BUDGET_FLOOR: int = 15
_WORD_BUDGET_CEIL: int = 80

_MODEL_SHARE: dict[Phase, float] = {
    Phase.INTAKE: 0.50,
    Phase.PRACTICE: 0.30,
    Phase.FEEDBACK: 0.70,
}


@dataclass(frozen=True)
class TimeCard:
    """Snapshot of where the call is in time, computed once per CLM turn."""

    phase: Phase
    seconds_elapsed_in_phase: int
    seconds_remaining_in_phase: int
    seconds_remaining_in_call: int
    phase_budget_seconds: int
    word_budget_this_turn: int


def build_time_card(
    session: Session,
    *,
    now: datetime,
    hard_cap_seconds: int = HARD_CAP_SECONDS,
) -> TimeCard:
    """Compute the live time card for one CLM turn."""
    open_timing = _open_phase_timing(session)
    elapsed_in_phase = max(0, int((now - open_timing.started_at).total_seconds()))
    remaining_in_phase = max(0, open_timing.budget_seconds - elapsed_in_phase)
    call_started_at = session.phase_timings[0].started_at
    elapsed_in_call = max(0, int((now - call_started_at).total_seconds()))
    remaining_in_call = max(0, hard_cap_seconds - elapsed_in_call)
    return TimeCard(
        phase=open_timing.phase,
        seconds_elapsed_in_phase=elapsed_in_phase,
        seconds_remaining_in_phase=remaining_in_phase,
        seconds_remaining_in_call=remaining_in_call,
        phase_budget_seconds=open_timing.budget_seconds,
        word_budget_this_turn=_word_budget(open_timing.phase, remaining_in_phase),
    )


def render_time_card(card: TimeCard) -> str:
    """Return the system-block text the model sees for this turn."""
    return (
        "Live timing\n"
        f"- Phase: {card.phase.value} ({_mmss(card.phase_budget_seconds)} budget)\n"
        f"- Elapsed in phase: {_mmss(card.seconds_elapsed_in_phase)}\n"
        f"- Remaining in phase: {_mmss(card.seconds_remaining_in_phase)}\n"
        f"- Remaining in call: {_mmss(card.seconds_remaining_in_call)} (hard cap)\n"
        f"- Target length for THIS reply: ~{card.word_budget_this_turn} words\n"
        "Speak only as long as the target. When phase time is nearly up, "
        "land the current beat in one closing sentence."
    )


def _word_budget(phase: Phase, remaining_in_phase: int) -> int:
    share = _MODEL_SHARE.get(phase, 0.5)
    expected_turns_left = max(1.0, remaining_in_phase / _AVG_TURN_SECONDS)
    raw = (remaining_in_phase * share * _WORDS_PER_SECOND) / expected_turns_left
    return max(_WORD_BUDGET_FLOOR, min(_WORD_BUDGET_CEIL, int(round(raw))))


def _open_phase_timing(session: Session):
    if not session.phase_timings:
        raise ValueError("session has no phase_timings; PhaseProcessor.bootstrap not run")
    for timing in reversed(session.phase_timings):
        if timing.ended_at is None:
            return timing
    return session.phase_timings[-1]


def _mmss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
