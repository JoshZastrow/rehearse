"""LLM-driven phase transitions for the live call runtime.

`MeetingPhaseProcessor` is a drop-in replacement for `PhaseProcessor`. It
subscribes to the shared `FrameBus`, fires a Haiku-class classifier on each
final user transcript turn (after the min dwell floor), and emits `PhaseSignal`
when the classifier calls the `transition_phase` tool.

Spec: docs/specs/v2026-05-18-meeting-phase-processor.md
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta
from typing import Any

import structlog

from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall, Frame, PhaseSignal, TranscriptDelta
from rehearse.phases.phases import PhaseBudgets, _apply_phase_transition, _close_open_phase
from rehearse.session.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, PhaseTiming, Session, Speaker

log = structlog.get_logger(__name__)

TRANSITION_PHASE_TOOL: dict = {
    "name": "transition_phase",
    "description": (
        "Move the conversation to the next phase when the current phase goals "
        "are complete. Call this only when the evidence in the transcript is clear. "
        "When in doubt, do not call this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "enum": ["practice", "feedback", "survey"],
                "description": "The target phase.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One sentence explaining what in the transcript signals "
                    "the current phase is complete."
                ),
            },
        },
        "required": ["to", "reason"],
    },
}

_PHASE_GOALS: dict[Phase, str] = {
    Phase.INTAKE: (
        "Goal: collect the user's scenario, counterparty, and stakes. "
        "Complete when the user has named their situation and is ready to rehearse."
    ),
    Phase.PRACTICE: (
        "Goal: run the roleplay. "
        "Complete when the user has finished at least one exchange and signals they are done."
    ),
    Phase.FEEDBACK: (
        "Goal: debrief and coach. "
        "Complete when the user has received feedback and signals they are done."
    ),
}


class MeetingPhaseProcessor:
    """Drive phase transitions via LLM tool calls instead of heuristic cues.

    Same external interface as `PhaseProcessor`: call `bootstrap()` once, then
    `run(bus.subscribe())` as an asyncio task. Emits `PhaseSignal` frames on the
    shared bus when the classifier calls `transition_phase`.
    """

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        bus: FrameBus,
        *,
        budgets: PhaseBudgets | None = None,
        clock: Callable[[], datetime] = utcnow,
        llm_client: Any,
        model: str = "claude-haiku-4-5-20251001",
        context_turns: int = 10,
    ) -> None:
        self._session_id = session_id
        self._store = store
        self._bus = bus
        self._budgets = budgets or PhaseBudgets()
        self._clock = clock
        self._llm_client = llm_client
        self._model = model
        self._context_turns = context_turns
        self._current_phase = Phase.INTAKE
        self._phase_started_at: datetime | None = None
        self._window: list[dict[str, str]] = []

    @property
    def current_phase(self) -> Phase:
        return self._current_phase

    async def bootstrap(self) -> None:
        """Ensure the session manifest has an open intake timing row."""
        session = await self._store.update_session(
            self._session_id, self._bootstrap_session
        )
        open_timing = session.phase_timings[-1]
        self._current_phase = open_timing.phase
        self._phase_started_at = open_timing.started_at

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume frames, classify on user turns, close phase on exit."""
        if self._phase_started_at is None:
            await self.bootstrap()
        async for frame in frames:
            if isinstance(frame, TranscriptDelta) and frame.is_final:
                role = "user" if frame.speaker == Speaker.USER else "assistant"
                self._window.append({"role": role, "content": frame.text})
                if frame.speaker == Speaker.USER:
                    await self._maybe_classify()
            elif isinstance(frame, EndOfCall):
                break
        await self._close_current_phase()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bootstrap_session(self, session: Session) -> Session:
        if session.phase_timings:
            return session
        session.phase_timings.append(
            PhaseTiming(
                phase=Phase.INTAKE,
                started_at=self._clock(),
                budget_seconds=self._budgets.intake_seconds,
            )
        )
        return session

    def _min_dwell_elapsed(self) -> bool:
        if self._phase_started_at is None:
            return False
        floor = self._budgets.min_dwell_for(self._current_phase)
        if floor <= 0:
            return True
        return self._clock() - self._phase_started_at >= timedelta(seconds=floor)

    async def _maybe_classify(self) -> None:
        if not self._min_dwell_elapsed():
            return
        window = self._window[-self._context_turns:]
        try:
            response = await self._llm_client.messages.create(
                model=self._model,
                max_tokens=256,
                system=self._build_system_prompt(),
                messages=window,
                tools=[TRANSITION_PHASE_TOOL],
                tool_choice={"type": "auto"},
            )
        except Exception as exc:
            log.warning(
                "meeting_phase_processor.classify_failed",
                session_id=self._session_id,
                error=str(exc),
            )
            return
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and block.name == "transition_phase"
            ):
                to_phase = Phase(block.input["to"])
                reason = block.input["reason"]
                await self._transition(to_phase, reason=reason)
                break

    def _build_system_prompt(self) -> str:
        goal = _PHASE_GOALS.get(self._current_phase, "")
        return (
            f"You are a phase classifier for a voice coaching session. "
            f"Current phase: {self._current_phase.value}. {goal} "
            f"Call transition_phase only when the evidence in the transcript clearly "
            f"shows the current phase is complete. Be conservative — when in doubt, "
            f"do not transition."
        )

    async def _transition(self, to_phase: Phase, *, reason: str) -> None:
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
        log.info(
            "meeting_phase_processor.transition",
            session_id=self._session_id,
            from_phase=from_phase.value,
            to_phase=to_phase.value,
            reason=reason,
        )
        await self._bus.publish(
            PhaseSignal(
                session_id=self._session_id,
                from_phase=from_phase,
                to_phase=to_phase,
                reason="llm",
                ts=now.timestamp(),
            )
        )
        await self._store.append(
            self._session_id,
            "telemetry.jsonl",
            json.dumps(
                {
                    "event": "phase_transition",
                    "from": from_phase.value,
                    "to": to_phase.value,
                    "reason": reason,
                    "ts": now.isoformat(),
                }
            ),
        )

    async def _close_current_phase(self) -> None:
        now = self._clock()
        await self._store.update_session(
            self._session_id,
            lambda session: _close_open_phase(session, ended_at=now),
        )


__all__ = ["MeetingPhaseProcessor", "TRANSITION_PHASE_TOOL"]
