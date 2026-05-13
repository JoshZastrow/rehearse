"""AgentRouter protocol and PhaseRouter implementation.

The router maps a session (+ optional hint) to the right RehearseAgent.
PhaseRouter is the v1 implementation — it reads the current phase from
session.phase_timings and returns the corresponding agent. Future routers
can use intake artifacts, caller history, or other signals.
"""

from __future__ import annotations

from typing import Any, Protocol

from rehearse.agents.registry import AgentRegistry
from rehearse.agents.roles.base import RehearseAgent
from rehearse.types import Phase, Session


class AgentRouter(Protocol):
    """Choose the right agent for the current session turn."""

    async def route(
        self,
        session: Session,
        *,
        role_hint: str | None = None,
        artifact: Any | None = None,
    ) -> RehearseAgent:
        """Return the agent that should handle this CLM turn.

        role_hint: optional role string from Hume's ?role= query param.
        artifact: optional structured output from the previous phase.
        """
        ...


class PhaseRouter:
    """Route by current session phase. Direct replacement for _resolve_role()."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    async def route(
        self,
        session: Session,
        *,
        role_hint: str | None = None,
        artifact: Any | None = None,
    ) -> RehearseAgent:
        phase = _current_phase(session)
        if phase == Phase.PRACTICE:
            return self._registry.get("character")
        if phase == Phase.FEEDBACK:
            return self._registry.get("feedback_coach")
        return self._registry.get("intake_coach")


class PersonaAwareRouter:
    """Routes practice phase to a gender-specific character agent.

    At PRACTICE phase, reads the session's selected_persona_id to choose
    MaleCharacterAgent or FemaleCharacterAgent. Falls back to PhaseRouter
    for all other phases.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._phase_router = PhaseRouter(registry)

    async def route(
        self,
        session: Session,
        *,
        role_hint: str | None = None,
        artifact: Any | None = None,
    ) -> RehearseAgent:
        phase = _current_phase(session)

        if phase == Phase.PRACTICE:
            persona_id = getattr(session, "selected_persona_id", None)
            if persona_id:
                agent = self._registry.get(persona_id)
                if agent and agent.name in ("male_character", "female_character"):
                    return agent
            # Fall back: check selected_persona_id gender convention
            if persona_id and "male" in persona_id:
                return self._registry.get("male_character")
            return self._registry.get("female_character")

        return await self._phase_router.route(session, role_hint=role_hint, artifact=artifact)


def _current_phase(session: Session) -> Phase:
    """Return the currently active phase from session.phase_timings."""
    for timing in reversed(session.phase_timings):
        if timing.ended_at is None:
            return timing.phase
    return Phase.INTAKE
