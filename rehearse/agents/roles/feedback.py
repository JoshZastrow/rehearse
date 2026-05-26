"""FeedbackCoachAgent — delivers end-of-call coaching feedback."""

from __future__ import annotations

from rehearse.agents.roles.base import wrap_memory_context
from rehearse.memory.memory_manager import MemoryManager
from rehearse.personas import feedback_coach_system_prompt
from rehearse.types import Session


class FeedbackCoachAgent:
    """Feedback coach with cross-session growth recall."""

    name = "feedback_coach"

    _RECALL_QUERY = (
        "What growth has this caller shown across sessions? "
        "What feedback landed well for them? What patterns persist?"
    )

    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    async def recall(self, session: Session) -> str:
        caller_hash = session.phone_number_hash or ""
        if not caller_hash:
            return ""
        return await self._memory.prefetch(caller_hash, self._RECALL_QUERY)

    def system_prompt(self, session: Session, memory_context: str = "") -> str:
        base = feedback_coach_system_prompt()
        if memory_context:
            base = f"{base}\n\n{wrap_memory_context(memory_context)}"
        return base

    async def after_turn(
        self, session: Session, user_text: str, agent_text: str
    ) -> None:
        # store_session() is called by telephony.py at call end.
        # The Honcho Deriver / Hindsight indexer extracts feedback observations
        # from the stored transcript automatically.
        pass
