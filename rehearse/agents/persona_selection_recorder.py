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

            # 2. Get situation and intake details from artifact
            situation = self._extract_situation(frame)
            intake_gender, topic = self._extract_intake_fields(frame)

            # 3. Run routing agent if available, else use default for gender
            resolved_gender = intake_gender or gender
            persona_id = await self._select_persona(situation or "", resolved_gender)
            if not persona_id:
                return

            # 4. Persist selected_persona_id to session manifest
            await self._store.update_session(
                self._session_id,
                lambda s: _set_persona_id(s, persona_id),
            )

            # 5. Record per-topic preference when we have both topic and gender
            if intake_gender and topic:
                await self._memory.record_agent_preference(
                    self._caller_hash, topic, intake_gender
                )

            # 6. Also record global gender preference if new (backward compat)
            if gender is None and resolved_gender:
                await self._memory.record_gender_preference(self._caller_hash, resolved_gender)

            log.info(
                "persona_selection.recorded",
                session_id=self._session_id,
                persona_id=persona_id,
                topic=topic,
                gender=resolved_gender,
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

    def _extract_intake_fields(self, frame: IntakeComplete) -> tuple[str | None, str | None]:
        """Return (gender_preference, topic_category) from the intake artifact."""
        if frame.error or not frame.intake_path:
            return None, None
        try:
            data = json.loads(open(frame.intake_path).read())
            gender = data.get("gender_preference") or None
            topic = data.get("topic_category") or None
            return gender, topic
        except Exception:
            return None, None


def _set_persona_id(session, persona_id: str):
    session.selected_persona_id = persona_id
    return session
