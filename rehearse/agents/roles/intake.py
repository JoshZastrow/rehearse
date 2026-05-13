"""IntakeCoachAgent — drives the intake phase of a Rehearse call."""

from __future__ import annotations

from rehearse.agents.roles.base import wrap_memory_context
from rehearse.memory_manager import MemoryManager
from rehearse.personas import coach_system_prompt
from rehearse.types import Session


class IntakeCoachAgent:
    """Intake coach with cross-session topic recall."""

    name = "intake_coach"

    _RECALL_QUERY = (
        "What topics has this caller practiced before? "
        "What situations did they work on and what patterns or challenges came up?"
    )

    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    async def recall(self, session: Session) -> str:
        caller_hash = session.phone_number_hash or ""
        if not caller_hash:
            return ""
        return await self._memory.prefetch(caller_hash, self._RECALL_QUERY)

    def system_prompt(self, session: Session, memory_context: str = "") -> str:
        base = coach_system_prompt()
        if memory_context:
            base = f"{base}\n\n{wrap_memory_context(memory_context)}"
        return base

    async def after_turn(
        self, session: Session, user_text: str, agent_text: str
    ) -> None:
        pass  # IntakeMemoryRecorder writes via IntakeComplete frame
