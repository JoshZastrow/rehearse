"""Track live runtime phase timing and publish phase-transition signals.

This file owns the runtime's simple three-phase controller. It updates the
session manifest as the call moves through intake, practice, and feedback, and
it emits `PhaseSignal` frames on the shared bus when a transition happens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall, Frame, PhaseSignal, TranscriptDelta
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, PhaseTiming, Session, Speaker


@dataclass(frozen=True)
class PhaseBudgets:
    """Store the live time budget for each phase in seconds."""

    intake_seconds: int = 60
    practice_seconds: int = 180
    feedback_seconds: int = 60

    def for_phase(self, phase: Phase) -> int:
        """Return the configured time budget for one phase."""
        if phase == Phase.PRACTICE:
            return self.practice_seconds
        if phase == Phase.FEEDBACK:
            return self.feedback_seconds
        return self.intake_seconds


class PhaseProcessor:
    """Drive the runtime's phase state from live timing and transcript cues."""

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        bus: FrameBus,
        *,
        budgets: PhaseBudgets | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        """Store the session id, manifest store, bus, and timing dependencies."""
        self._session_id = session_id
        self._store = store
        self._bus = bus
        self._budgets = budgets or PhaseBudgets()
        self._clock = clock
        self._current_phase = Phase.INTAKE
        self._phase_started_at: datetime | None = None
        self._final_user_turns = 0

    @property
    def current_phase(self) -> Phase:
        """Return the phase the live call is currently in."""
        return self._current_phase

    async def bootstrap(self) -> None:
        """Ensure the session manifest has an open intake timing row."""
        session = await self._store.update_session(self._session_id, self._bootstrap_session)
        open_timing = session.phase_timings[-1]
        self._current_phase = open_timing.phase
        self._phase_started_at = open_timing.started_at

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume frames, advance phases, and close the final phase on exit."""
        if self._phase_started_at is None:
            await self.bootstrap()
        async for frame in frames:
            await self._maybe_advance_for_budget()
            if (
                isinstance(frame, TranscriptDelta)
                and frame.speaker == Speaker.USER
                and frame.is_final
            ):
                self._final_user_turns += 1
                await self._maybe_advance_for_cue(frame)
            elif isinstance(frame, EndOfCall):
                break
        await self._close_current_phase()

    def _bootstrap_session(self, session: Session) -> Session:
        """Insert an intake timing row if the session does not have one yet."""
        if session.phase_timings:
            return session
        started_at = self._clock()
        session.phase_timings.append(
            PhaseTiming(
                phase=Phase.INTAKE,
                started_at=started_at,
                budget_seconds=self._budgets.intake_seconds,
            )
        )
        return session

    async def _maybe_advance_for_budget(self) -> None:
        """Advance the phase when the active phase has exhausted its time budget."""
        if self._phase_started_at is None or self._current_phase == Phase.FEEDBACK:
            return
        budget = timedelta(seconds=self._budgets.for_phase(self._current_phase))
        if self._clock() - self._phase_started_at < budget:
            return
        next_phase = Phase.PRACTICE if self._current_phase == Phase.INTAKE else Phase.FEEDBACK
        await self._transition(next_phase, reason="budget")

    async def _maybe_advance_for_cue(self, frame: TranscriptDelta) -> None:
        """Advance the phase when a simple transcript cue says the user is ready."""
        text = frame.text.lower()
        if self._current_phase == Phase.INTAKE:
            if self._final_user_turns >= 2 or any(cue in text for cue in _INTAKE_READY_CUES):
                await self._transition(Phase.PRACTICE, reason="cue")
            return
        if self._current_phase == Phase.PRACTICE:
            if self._final_user_turns >= 5 or any(cue in text for cue in _FEEDBACK_READY_CUES):
                await self._transition(Phase.FEEDBACK, reason="cue")

    async def _transition(self, to_phase: Phase, *, reason: str) -> None:
        """Move to a new phase, persist the manifest change, and emit a signal."""
        if to_phase == self._current_phase:
            return
        now = self._clock()
        from_phase = self._current_phase
        await self._store.update_session(
            self._session_id,
            lambda session: _apply_phase_transition(
                session,
                from_phase=from_phase,
                to_phase=to_phase,
                at=now,
                budgets=self._budgets,
            ),
        )
        self._current_phase = to_phase
        self._phase_started_at = now
        self._final_user_turns = 0
        await self._bus.publish(
            PhaseSignal(
                session_id=self._session_id,
                from_phase=from_phase,
                to_phase=to_phase,
                reason=reason,
                ts=now.timestamp(),
            )
        )

    async def _close_current_phase(self) -> None:
        """Mark the active phase ended in the manifest when the call stops."""
        now = self._clock()
        await self._store.update_session(
            self._session_id,
            lambda session: _close_open_phase(session, ended_at=now),
        )


def _apply_phase_transition(
    session: Session,
    *,
    from_phase: Phase,
    to_phase: Phase,
    at: datetime,
    budgets: PhaseBudgets,
) -> Session:
    """Close the previous phase row and append the next active phase row."""
    session = _close_open_phase(session, ended_at=at)
    session.phase_timings.append(
        PhaseTiming(
            phase=to_phase,
            started_at=at,
            budget_seconds=budgets.for_phase(to_phase),
        )
    )
    return session


def _close_open_phase(session: Session, *, ended_at: datetime) -> Session:
    """Fill the end timestamp on the last open phase row if needed."""
    if not session.phase_timings:
        return session
    current = session.phase_timings[-1]
    if current.ended_at is not None:
        return session
    current.ended_at = ended_at
    budget = timedelta(seconds=current.budget_seconds)
    current.overran = ended_at - current.started_at > budget
    return session


_INTAKE_READY_CUES = frozenset({"let's practice", "roleplay", "start the conversation", "try it"})
_FEEDBACK_READY_CUES = frozenset({"feedback", "debrief", "how did that go", "what should i change"})
