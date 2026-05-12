"""Record intake situations to caller memory after each completed intake phase.

Subscribes to the FrameBus and writes the caller's situation to memory
when IntakeComplete fires, so future calls can surface prior topics.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import structlog

from rehearse.frames import EndOfCall, Frame, IntakeComplete
from rehearse.memory import CallerMemory

log = structlog.get_logger(__name__)


class IntakeMemoryRecorder:
    """Write the intake situation to caller memory after each completed intake."""

    def __init__(
        self,
        session_id: str,
        caller_hash: str | None,
        memory: CallerMemory,
    ) -> None:
        self._session_id = session_id
        self._caller_hash = caller_hash
        self._memory = memory

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume bus frames and record the situation on IntakeComplete."""
        async for frame in frames:
            if isinstance(frame, EndOfCall):
                return
            if not isinstance(frame, IntakeComplete):
                continue
            if not self._caller_hash:
                return
            situation = self._extract_situation(frame)
            if situation:
                await self._memory.record_intake(self._caller_hash, situation)
                log.info(
                    "intake_memory.recorded",
                    session_id=self._session_id,
                    situation=situation[:60],
                )
            return  # only fires once per call

    def _extract_situation(self, frame: IntakeComplete) -> str | None:
        """Read situation from the intake.json artifact path."""
        if frame.error or not frame.intake_path:
            return None
        try:
            data = json.loads(open(frame.intake_path).read())
            return data.get("situation", "").strip() or None
        except Exception as exc:
            log.warning(
                "intake_memory.read_failed",
                session_id=self._session_id,
                error=str(exc),
            )
            return None
