"""Capture intake structure and compile a practice persona during the live call.

This file observes transcript and phase events on the runtime bus, keeps a
simple deterministic intake record up to date during Phase 1, and compiles a
session-specific character persona when the call enters practice.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime

from rehearse.frames import Frame, PhaseSignal, TranscriptDelta
from rehearse.personas import build_intake_record, compile_character
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, Session, Speaker


class IntakeProcessor:
    """Persist structured intake and compiled persona from live transcript events."""

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        *,
        phase_getter: Callable[[], Phase],
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        """Store the session id, manifest store, and current-phase callback."""
        self._session_id = session_id
        self._store = store
        self._phase_getter = phase_getter
        self._clock = clock
        self._phase = Phase.INTAKE
        self._user_turns: list[str] = []

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume bus frames and update intake/persona artifacts as phases change."""
        async for frame in frames:
            if (
                isinstance(frame, TranscriptDelta)
                and frame.speaker == Speaker.USER
                and frame.is_final
                and self._phase == Phase.INTAKE
            ):
                self._user_turns.append(frame.text.strip())
                await self._persist_intake()
                if len(self._user_turns) >= 2:
                    await self._compile_persona()
            elif isinstance(frame, PhaseSignal):
                self._phase = frame.to_phase
                if frame.to_phase == Phase.PRACTICE:
                    await self._compile_persona()
        if self._user_turns and (
            self._phase != Phase.INTAKE or self._phase_getter() != Phase.INTAKE
        ):
            await self._compile_persona()

    async def _persist_intake(self) -> None:
        """Write the latest intake guess into the session manifest."""
        if not self._user_turns:
            return
        captured_at = self._clock()
        await self._store.update_session(
            self._session_id,
            lambda session: _apply_intake(
                session,
                build_intake_record(
                    session_id=self._session_id,
                    user_turns=self._user_turns,
                    captured_at=captured_at,
                ),
            ),
        )

    async def _compile_persona(self) -> None:
        """Compile and persist the character persona from the stored intake."""
        compiled_at = self._clock()
        await self._store.update_session(
            self._session_id,
            lambda session: _apply_persona(session, compiled_at=compiled_at),
        )


def _apply_intake(session: Session, intake) -> Session:
    """Set the current intake record on the stored session manifest."""
    session.intake = intake
    return session


def _apply_persona(session: Session, *, compiled_at: datetime) -> Session:
    """Compile and store the session persona when intake data exists."""
    if session.intake is not None:
        session.persona = compile_character(session.intake, compiled_at=compiled_at)
    return session
