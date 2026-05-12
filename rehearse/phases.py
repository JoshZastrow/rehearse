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
from rehearse.frames import EndOfCall, Frame, IntakeComplete, PhaseSignal, TranscriptDelta
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Phase, PhaseTiming, Session, Speaker


@dataclass(frozen=True)
class PhaseBudgets:
    """Store the live time budget and minimum dwell for each phase in seconds.

    `*_min_dwell_seconds` is the floor a phase must run for before a cue-driven
    transition is allowed. Budget-driven transitions ignore the floor — they
    fire only after the full budget has elapsed anyway.
    """

    intake_seconds: int = 60
    practice_seconds: int = 120
    feedback_seconds: int = 60
    intake_min_dwell_seconds: int = 30
    practice_min_dwell_seconds: int = 90

    def for_phase(self, phase: Phase) -> int:
        """Return the configured time budget for one phase."""
        if phase == Phase.PRACTICE:
            return self.practice_seconds
        if phase == Phase.FEEDBACK:
            return self.feedback_seconds
        return self.intake_seconds

    def min_dwell_for(self, phase: Phase) -> int:
        """Return the minimum dwell-time floor before a cue can leave a phase."""
        if phase == Phase.PRACTICE:
            return self.practice_min_dwell_seconds
        return self.intake_min_dwell_seconds


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
        consent_getter: Callable[[], ConsentState] | None = None,
        wait_for_intake_complete: bool = False,
    ) -> None:
        """Store the session id, manifest store, bus, and timing dependencies."""
        self._session_id = session_id
        self._store = store
        self._bus = bus
        self._budgets = budgets or PhaseBudgets()
        self._clock = clock
        self._consent_getter = consent_getter or (lambda: ConsentState.GRANTED)
        self._current_phase = Phase.INTAKE
        self._phase_started_at: datetime | None = None
        self._final_user_turns = 0
        # When True, INTAKE→PRACTICE transition is gated on receiving IntakeComplete.
        # Used by RuntimeHost to guarantee intake.json is written before persona compile.
        self._wait_for_intake_complete = wait_for_intake_complete
        self._intake_complete_received = not wait_for_intake_complete

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
                # Don't count consent-phase turns toward intake progress.
                # Otherwise "Yup" (consent grant) plus the user's first real
                # answer trips the n>=2 cue before intake content lands.
                if self._consent_getter() == ConsentState.GRANTED:
                    self._final_user_turns += 1
                    await self._maybe_advance_for_cue(frame)
            elif isinstance(frame, IntakeComplete):
                self._intake_complete_received = True
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
        if (
            self._current_phase == Phase.INTAKE
            and self._consent_getter() != ConsentState.GRANTED
        ):
            return
        if self._current_phase == Phase.INTAKE and not self._intake_complete_received:
            return
        budget = timedelta(seconds=self._budgets.for_phase(self._current_phase))
        if self._clock() - self._phase_started_at < budget:
            return
        next_phase = Phase.PRACTICE if self._current_phase == Phase.INTAKE else Phase.FEEDBACK
        await self._transition(next_phase, reason="budget")

    async def _maybe_advance_for_cue(self, frame: TranscriptDelta) -> None:
        """Advance the phase when a transcript cue plus minimum dwell are met."""
        if not self._min_dwell_elapsed():
            return
        text = frame.text.lower()
        if self._current_phase == Phase.INTAKE:
            if self._consent_getter() != ConsentState.GRANTED:
                return
            if not self._intake_complete_received:
                return
            if self._final_user_turns >= 2 and _matches_any(text, _INTAKE_READY_CUES):
                await self._transition(Phase.PRACTICE, reason="cue")
            return
        if self._current_phase == Phase.PRACTICE:
            if self._final_user_turns >= 5 or _matches_any(text, _FEEDBACK_READY_CUES):
                await self._transition(Phase.FEEDBACK, reason="cue")

    def _min_dwell_elapsed(self) -> bool:
        """Return True if the active phase has run past its minimum dwell floor."""
        if self._phase_started_at is None:
            return False
        floor = self._budgets.min_dwell_for(self._current_phase)
        if floor <= 0:
            return True
        return self._clock() - self._phase_started_at >= timedelta(seconds=floor)

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


_INTAKE_READY_CUES: frozenset[str] = frozenset(
    {
        "let's practice",
        "let's roleplay",
        "let's try it",
        "start the conversation",
        "i'm ready",
        "ready to practice",
        "ready to roleplay",
        "begin the scene",
    }
)

# Phrasal cues only — bare "feedback" matches every job-feedback / performance-
# review situation in intake content and trips a premature transition.
_FEEDBACK_READY_CUES: frozenset[str] = frozenset(
    {
        "give me feedback",
        "your feedback",
        "some feedback",
        "how did i do",
        "how did that go",
        "what should i change",
        "what would you change",
        "let's debrief",
        "stop the scene",
        "pause the scene",
        "out of scene",
        "step out",
    }
)


def _matches_any(text: str, cues: frozenset[str]) -> bool:
    """Return True if any cue phrase appears as a substring of the lowered text."""
    return any(cue in text for cue in cues)
