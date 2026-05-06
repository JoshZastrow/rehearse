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

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Final

import structlog

from rehearse.frames import EndOfCall, Frame, PhaseSignal
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, Session

log = structlog.get_logger(__name__)

Speak = Callable[[str], Awaitable[None]]

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
        speak: Speak,
        *,
        bridges: Mapping[Phase, str] | None = None,
    ) -> None:
        """Bind the session id, manifest store, speak function, and bridge copy."""
        self._session_id = session_id
        self._store = store
        self._speak = speak
        self._bridges = bridges or _DEFAULT_BRIDGES

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
            text = await self._render(template)
            try:
                await self._speak(text)
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
