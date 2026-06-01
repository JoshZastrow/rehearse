"""InteractiveBackend — ConversationBackend that proxies calls to the Modal interactive server.

Connects to INTERACTIVE_MODAL_ENDPOINT via WebSocket. Sends caller PCM16
chunks as binary frames; receives provider audio and JSON events back.
Translates each to the appropriate FrameBus event.

No model loading or GPU code runs locally — all inference is on Modal.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING

import structlog
import websockets

from rehearse.frames import AudioChunk, EndOfCall, SessionStoredEvent, TranscriptDelta
from rehearse.types import Speaker

if TYPE_CHECKING:
    from rehearse.bus import FrameBus

log = structlog.get_logger(__name__)


class InteractiveBackend:
    """ConversationBackend that proxies to the deployed Modal Moshi server."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._session_id = ""
        self._bus: FrameBus | None = None
        self._task: asyncio.Task | None = None
        self._send_q: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def __aenter__(self) -> InteractiveBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self, session_id: str, bus: FrameBus) -> None:
        self._session_id = session_id
        self._bus = bus
        self._task = asyncio.create_task(
            self._run(), name=f"interactive-{session_id}"
        )
        log.info("interactive_backend.started", session_id=session_id, endpoint=self._endpoint)

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        await self._send_q.put(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        log.warning("interactive_backend.inject_speech_not_supported", text=text[:50])

    async def swap_persona(self, persona: object) -> None:
        log.warning("interactive_backend.swap_persona_not_supported")

    async def close(self) -> None:
        await self._send_q.put(None)  # sentinel to stop sender
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._task.cancel()
        log.info("interactive_backend.closed", session_id=self._session_id)

    async def _run(self) -> None:
        try:
            async with websockets.connect(self._endpoint, open_timeout=30.0) as ws:
                await ws.send(json.dumps({
                    "type": "start",
                    "session_id": self._session_id,
                }))

                sender = asyncio.create_task(self._sender(ws))
                try:
                    async for msg in ws:
                        if isinstance(msg, bytes):
                            await self._publish(AudioChunk(
                                session_id=self._session_id,
                                speaker=Speaker.COACH,
                                pcm16_16k=msg,
                                ts=time.time(),
                            ))
                        else:
                            await self._handle_event(json.loads(msg))
                finally:
                    sender.cancel()
        except Exception as exc:
            log.error("interactive_backend.connection_error", error=str(exc))
            await self._publish(EndOfCall(
                session_id=self._session_id,
                reason="error",
                ts=time.time(),
            ))

    async def _sender(self, ws: websockets.WebSocketClientProtocol) -> None:
        while True:
            chunk = await self._send_q.get()
            if chunk is None:
                break
            await ws.send(chunk)

    async def _handle_event(self, event: dict) -> None:
        t = event.get("type")
        now = time.time()

        if t == "transcript":
            await self._publish(TranscriptDelta(
                session_id=self._session_id,
                utterance_id=event.get("utterance_id", str(uuid.uuid4())),
                speaker=Speaker.COACH,
                text=event.get("text", ""),
                is_final=event.get("is_final", False),
                ts_start=now,
                ts_end=now if event.get("is_final") else None,
            ))

        elif t == "session_stored":
            await self._publish(SessionStoredEvent(
                session_id=self._session_id,
                volume_path=event.get("volume_path", ""),
                artifacts=event.get("artifacts", []),
                ts=now,
            ))

        elif t == "end_of_call":
            await self._publish(EndOfCall(
                session_id=self._session_id,
                reason=event.get("reason", "hangup"),  # type: ignore[arg-type]
                ts=now,
            ))

    async def _publish(self, frame: object) -> None:
        if self._bus is not None:
            await self._bus.publish(frame)  # type: ignore[arg-type]
