"""Speak a short bridge utterance whenever the live phase changes.

Subscribes to the runtime `FrameBus` and watches for `PhaseSignal` frames.
On `to_phase=PRACTICE` or `to_phase=FEEDBACK`, the coordinator asks Hume
to speak one short, deterministic line via `assistant_input`. The bridge
is fixed copy (not LLM-generated) so transition latency stays predictable
and the user audibly hears the role change.

`PhaseSignal(to_phase=INTAKE)` is ignored: intake is the call's starting
phase and there is no role transition to mark.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Final

import structlog

from rehearse.frames import EndOfCall, Frame, PhaseSignal
from rehearse.participants import SpeakRequest, VoiceSpeaker
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, Session

_GENDER_BRIDGE = "Just a moment while I connect you with your practice partner."

log = structlog.get_logger(__name__)

_DEFAULT_BRIDGES: Final[Mapping[Phase, str]] = {
    Phase.PRACTICE: "Okay, let's run it. I'll be {relationship} now.",
    Phase.FEEDBACK: "Let's pause the scene. Quick reflection.",
}

_FALLBACK_RELATIONSHIP: Final[str] = "the other person"


class PersonaSwapCoordinator:
    """Speak a short bridge utterance whenever the live phase changes."""

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        speaker: VoiceSpeaker,
        *,
        bridges: Mapping[Phase, str] | None = None,
        backend: Any | None = None,
    ) -> None:
        """Bind the session id, manifest store, speak function, and bridge copy."""
        self._session_id = session_id
        self._store = store
        self._speaker = speaker
        self._bridges = bridges or _DEFAULT_BRIDGES
        self._backend = backend

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume bus frames and speak bridge lines on phase transitions."""
        async for frame in frames:
            if isinstance(frame, EndOfCall):
                return
            if not isinstance(frame, PhaseSignal):
                continue
            template = self._bridges.get(frame.to_phase)
            if template is None:
                continue

            # At the intake→practice transition, send session_settings to
            # swap the voice if a persona was selected for this session.
            if frame.to_phase == Phase.PRACTICE and self._backend is not None:
                await self._swap_voice_for_session()

            text = await self._render(template)
            try:
                await self._speaker.say(SpeakRequest(text=text))
            except Exception as exc:
                log.warning(
                    "persona_swap.speak_failed",
                    session_id=self._session_id,
                    to_phase=frame.to_phase.value,
                    error=str(exc),
                )
                continue
            log.info(
                "persona_swap.bridge",
                session_id=self._session_id,
                to_phase=frame.to_phase.value,
                reason=frame.reason,
            )

    async def _swap_voice_for_session(self) -> None:
        """Build a PersonaSpec from the session manifest and call backend.swap_persona()."""
        try:
            payload = await self._store.read(self._session_id, "session.json")
            session = Session.model_validate_json(payload)
            persona_id = getattr(session, "selected_persona_id", None)
            if not persona_id:
                return
            from rehearse.backends.base import PersonaSpec
            from rehearse.personas.registry import PERSONA_REGISTRY, PersonaRegistry
            from rehearse.services.hume_configs import resolve_voice_id
            registry = PersonaRegistry(PERSONA_REGISTRY)
            persona = registry.get(persona_id)
            if persona is None:
                return
            voice_id = resolve_voice_id(persona.voice_name)
            spec: PersonaSpec = {
                "name": persona.display_name or persona_id,
                "gender": persona.gender,
                "system_prompt": persona.system_prompt_template or "",
                "voice_ref": voice_id,
            }
            await self._backend.swap_persona(spec)
            log.info(
                "persona_swap.voice_swapped",
                session_id=self._session_id,
                persona_id=persona_id,
            )
        except Exception as exc:
            log.warning("persona_swap.voice_swap_failed", session_id=self._session_id, error=str(exc))

    async def _render(self, template: str) -> str:
        """Interpolate `{relationship}` from the session manifest, if used."""
        if "{relationship}" not in template:
            return template
        relationship = await self._lookup_relationship()
        return template.replace("{relationship}", relationship)

    async def _lookup_relationship(self) -> str:
        """Return the counterparty relationship from the session, or a fallback."""
        try:
            payload = await self._store.read(self._session_id, "session.json")
        except FileNotFoundError:
            return _FALLBACK_RELATIONSHIP
        try:
            session = Session.model_validate_json(payload)
        except Exception:
            return _FALLBACK_RELATIONSHIP
        if session.persona is None:
            return _FALLBACK_RELATIONSHIP
        relationship = (session.persona.relationship or "").strip()
        return relationship or _FALLBACK_RELATIONSHIP
