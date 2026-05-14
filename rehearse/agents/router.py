"""AgentRouter protocol, PhaseRouter, PersonaAwareRouter, and IntakeAwareRouter.

The router maps a session (+ optional hint) to the right RehearseAgent.
PhaseRouter is the v1 implementation — it reads the current phase from
session.phase_timings and returns the corresponding agent. Future routers
can use intake artifacts, caller history, or other signals.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, Protocol

import structlog

from rehearse.agents.registry import AgentRegistry
from rehearse.agents.roles.base import RehearseAgent
from rehearse.types import Phase, Session

log = structlog.get_logger(__name__)


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


class IntakeAwareRouter:
    """Routes practice phase using per-topic gender preference from caller memory.

    At PRACTICE phase:
      1. If intake.topic_category is already set, use it directly (no LLM call).
      2. Otherwise, run classify_topic(intake.situation) to determine the topic.
      3. Look up get_agent_preference(caller_hash, topic) from memory.
      4. Return MaleCharacterAgent if preference is "male", else FemaleCharacterAgent.

    Any error in classification or memory lookup falls back to FemaleCharacterAgent.
    Non-PRACTICE phases delegate to PhaseRouter.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        memory: Any,
        *,
        llm_client: Any,
        _classify: Callable[..., Coroutine[Any, Any, str]] | None = None,
    ) -> None:
        self._registry = registry
        self._memory = memory
        self._llm = llm_client
        self._phase_router = PhaseRouter(registry)
        # Injectable classifier for testing; defaults to classify_topic
        self._classify = _classify

    async def route(
        self,
        session: Session,
        *,
        role_hint: str | None = None,
        artifact: Any | None = None,
    ) -> RehearseAgent:
        phase = _current_phase(session)

        if phase != Phase.PRACTICE:
            return await self._phase_router.route(session, role_hint=role_hint, artifact=artifact)

        try:
            intake = session.intake
            if not intake or not intake.situation:
                return self._registry.get("female_character")

            # Use cached topic if already classified during intake phase.
            topic = intake.topic_category
            if not topic:
                classify = self._classify or _default_classify
                topic = await classify(intake.situation, client=self._llm)

            pref = await self._memory.get_agent_preference(
                session.phone_number_hash or "", topic
            )
            if pref == "male":
                return self._registry.get("male_character")
            return self._registry.get("female_character")

        except Exception as exc:
            log.warning("intake_aware_router.fallback", error=str(exc))
            return self._registry.get("female_character")


async def _default_classify(situation: str, *, client: Any, model: str = "") -> str:
    from rehearse.agents.topic import classify_topic

    return await classify_topic(situation, client=client)
