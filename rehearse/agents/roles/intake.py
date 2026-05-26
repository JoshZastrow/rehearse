"""IntakeCoachAgent — drives the intake phase of a Rehearse call."""

from __future__ import annotations

from rehearse.agents.roles.base import wrap_memory_context
from rehearse.memory.memory_manager import MemoryManager
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

        prior_context = await self._memory.prefetch(caller_hash, self._RECALL_QUERY)

        # Gate the gender question based on per-topic preference.
        intake = session.intake
        topic = intake.topic_category if intake else None
        if topic:
            pref = await self._memory.get_agent_preference(caller_hash, topic)
            if pref is not None:
                gender_note = (
                    f"This caller's preference for {topic} topics is already known: {pref}. "
                    "Do NOT ask about gender preference."
                )
            else:
                gender_note = (
                    f"This caller has no stored preference for {topic} topics. "
                    "After confirming their situation, ask: "
                    "'One more thing — would you prefer to practice with a male or female voice?'"
                )
            return f"{prior_context}\n\n{gender_note}".strip() if prior_context else gender_note

        return prior_context

    def system_prompt(self, session: Session, memory_context: str = "") -> str:
        base = coach_system_prompt()
        if memory_context:
            base = f"{base}\n\n{wrap_memory_context(memory_context)}"
        return base

    async def after_turn(
        self, session: Session, user_text: str, agent_text: str
    ) -> None:
        pass  # IntakeMemoryRecorder writes via IntakeComplete frame
