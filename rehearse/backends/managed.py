"""ManagedBackend — wraps HumeEVIClient behind the ConversationBackend protocol.

No new logic. Thin delegation layer. The external Hume EVI service owns
the audio loop; this class translates between the ConversationBackend API
and the HumeEVIClient API.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import structlog

from rehearse.backends.base import PersonaSpec
from rehearse.bus import FrameBus
from rehearse.participants import SpeakRequest

log = structlog.get_logger(__name__)


class ManagedBackend:
    """ConversationBackend backed by the managed Hume EVI service."""

    def __init__(self, *, api_key: str, config_id: str) -> None:
        self._api_key = api_key
        self._config_id = config_id
        self._client: Any = None   # HumeEVIClient | None — created lazily in start()
        self._task: asyncio.Task | None = None
        self._session_id = ""

    async def __aenter__(self) -> ManagedBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self, session_id: str, bus: FrameBus) -> None:
        """Create the HumeEVIClient, open the websocket, and launch the event loop."""
        self._session_id = session_id
        if self._client is None:
            from rehearse.services.hume_evi import HumeEVIClient
            self._client = HumeEVIClient(
                api_key=self._api_key,
                config_id=self._config_id,
                session_id=session_id,
            )
        await self._client.__aenter__()
        # run_event_loop() reads Hume events and publishes Rehearse frames
        self._task = asyncio.create_task(
            self._client.run_event_loop(bus),
            name=f"managed-backend-{session_id}",
        )
        log.info("managed_backend.started", session_id=session_id)

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        """Forward caller audio to the Hume service."""
        await self._client.send_audio(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        """Speak a deterministic line through Hume assistant input."""
        await self._client.say(text)

    async def swap_persona(self, persona: PersonaSpec) -> None:
        """Change voice and system prompt via Hume session_settings."""
        await self._client.send_session_settings(
            voice_id=persona["voice_ref"] or "",
            system_prompt=persona["system_prompt"],
        )
        log.info(
            "managed_backend.persona_swapped",
            session_id=self._session_id,
            persona_name=persona["name"],
        )

    async def say(self, request: SpeakRequest) -> None:
        """Satisfy VoiceSpeaker protocol used by consent gate and outcome probe."""
        await self.inject_speech(request.text)

    async def close(self) -> None:
        """Cancel the internal task and close the Hume websocket."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
        log.info("managed_backend.closed", session_id=self._session_id)
