"""CharacterAgent — plays the counterparty during the practice phase."""

from __future__ import annotations

from rehearse.agents.roles.base import wrap_memory_context
from rehearse.memory_manager import MemoryManager
from rehearse.personas import character_system_prompt
from rehearse.types import Session


class CharacterAgent:
    """Counterparty character. Phase 1: no cross-session recall."""

    name = "character"

    _RECALL_QUERY = (
        "What do you know about how this caller communicates under pressure? "
        "What triggers or patterns have you observed in their rehearsal sessions?"
    )

    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    async def recall(self, session: Session) -> str:
        # Phase 1: no cross-session recall for the character role.
        # Phase 2: uncomment to give the character behavioral context.
        # return await self._memory.prefetch(session.phone_number_hash or "", self._RECALL_QUERY)
        return ""

    def system_prompt(self, session: Session, memory_context: str = "") -> str:
        persona = session.persona if session else None
        base = character_system_prompt(persona or "Be the other person in the conversation.")
        if memory_context:
            base = f"{base}\n\n{wrap_memory_context(memory_context)}"
        return base

    async def after_turn(
        self, session: Session, user_text: str, agent_text: str
    ) -> None:
        pass


class MaleCharacterAgent(CharacterAgent):
    """Male-gendered practice partner. Voice set via session_settings at transition."""

    name = "male_character"

    def system_prompt(self, session: Session, memory_context: str = "") -> str:
        persona = session.persona if session else None
        base = character_system_prompt(persona or "Be the other person in the conversation.")
        gender_prefix = (
            "You are playing a male character. Use a male name (e.g. Alex, Marcus, or James) "
            "and male pronouns. "
        )
        base = gender_prefix + base
        if memory_context:
            base = f"{base}\n\n{wrap_memory_context(memory_context)}"
        return base


class FemaleCharacterAgent(CharacterAgent):
    """Female-gendered practice partner. Voice set via session_settings at transition."""

    name = "female_character"

    def system_prompt(self, session: Session, memory_context: str = "") -> str:
        persona = session.persona if session else None
        base = character_system_prompt(persona or "Be the other person in the conversation.")
        gender_prefix = (
            "You are playing a female character. Use a female name (e.g. Sarah, Maya, or Elena) "
            "and female pronouns. "
        )
        base = gender_prefix + base
        if memory_context:
            base = f"{base}\n\n{wrap_memory_context(memory_context)}"
        return base
