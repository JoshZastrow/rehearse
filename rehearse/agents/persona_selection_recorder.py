"""Run persona routing after intake completes and persist the selection.

Subscribes to IntakeComplete. At that point, we know:
- The intake situation (from intake.json)
- The caller's gender preference (from CallerMemory)

Runs PersonaRoutingAgent to pick the best persona, then writes
session.selected_persona_id to the session manifest so PersonaSwapCoordinator
can read it at the intake→practice transition.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import structlog

from rehearse.frames import EndOfCall, Frame, IntakeComplete
from rehearse.memory import CallerMemory
from rehearse.storage import LocalFilesystemStore

log = structlog.get_logger(__name__)


class PersonaSelectionRecorder:
    """Pick and persist a persona after IntakeComplete fires."""

    def __init__(
        self,
        session_id: str,
        caller_hash: str | None,
        memory: CallerMemory,
        store: LocalFilesystemStore,
        *,
        routing_agent: object | None = None,
    ) -> None:
        self._session_id = session_id
        self._caller_hash = caller_hash
        self._memory = memory
        self._store = store
        self._routing_agent = routing_agent

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        async for frame in frames:
            if isinstance(frame, EndOfCall):
                return
            if not isinstance(frame, IntakeComplete):
                continue
            await self._handle(frame)
            return

    async def _handle(self, frame: IntakeComplete) -> None:
        if not self._caller_hash:
            return
        try:
            # 1. Get gender preference from memory
            gender = await self._memory.get_gender_preference(self._caller_hash)

            # 2. Get situation from intake artifact
            situation = self._extract_situation(frame)

            # 3. Run routing agent if available, else use default for gender
            persona_id = await self._select_persona(situation or "", gender)
            if not persona_id:
                return

            # 4. Persist selected_persona_id to session manifest
            await self._store.update_session(
                self._session_id,
                lambda s: _set_persona_id(s, persona_id),
            )

            # 5. If gender preference is new, store it from the persona
            if gender is None:
                from rehearse.personas.registry import PERSONA_REGISTRY, PersonaRegistry
                registry = PersonaRegistry(PERSONA_REGISTRY)
                persona = registry.get(persona_id)
                if persona:
                    await self._memory.record_gender_preference(self._caller_hash, persona.gender)

            log.info(
                "persona_selection.recorded",
                session_id=self._session_id,
                persona_id=persona_id,
            )
        except Exception as exc:
            log.warning("persona_selection.failed", session_id=self._session_id, error=str(exc))

    async def _select_persona(self, situation: str, gender: str | None) -> str | None:
        if self._routing_agent is not None:
            try:
                persona = await self._routing_agent.select(
                    transcript=situation,
                    gender_hint=gender,
                )
                return persona.id
            except Exception as exc:
                log.warning("persona_selection.routing_agent_failed", error=str(exc))

        # Fallback: use gender directly
        from rehearse.personas.registry import PERSONA_REGISTRY, PersonaRegistry
        registry = PersonaRegistry(PERSONA_REGISTRY)
        resolved_gender = gender if gender in ("male", "female") else "female"
        return registry.default(gender=resolved_gender).id

    def _extract_situation(self, frame: IntakeComplete) -> str | None:
        if frame.error or not frame.intake_path:
            return None
        try:
            data = json.loads(open(frame.intake_path).read())
            return data.get("situation", "").strip() or None
        except Exception:
            return None


def _set_persona_id(session, persona_id: str):
    session.selected_persona_id = persona_id
    return session
