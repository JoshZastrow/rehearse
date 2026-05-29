"""ModalInteractiveBackend — ConversationBackend that proxies audio to a Modal GPU server.

Wire protocol (binary WebSocket):
  Client → server: raw PCM16 16 kHz bytes (audio) or UTF-8 JSON (control)
  Server → client: raw PCM16 16 kHz bytes (audio) or UTF-8 JSON (events)

JSON frames from server:
  {"type": "transcript", "utterance_id": "...", "speaker": "coach"|"user",
   "text": "...", "is_final": true}
  {"type": "prosody", "utterance_id": "...", "speaker": "user",
   "arousal": 0.0, "valence": 0.0}
  {"type": "end_of_call", "reason": "hangup"}
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING

import structlog
import websockets
import websockets.exceptions

from rehearse.backends.base import PersonaSpec
from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, TranscriptDelta
from rehearse.types import ProsodyScores, Speaker

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

log = structlog.get_logger(__name__)


class ModalInteractiveBackend:
    """ConversationBackend that proxies audio to a Modal-hosted WebSocket inference server."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._ws: ClientConnection | None = None
        self._recv_task: asyncio.Task | None = None
        self._session_id = ""
        self._bus: FrameBus | None = None

    async def __aenter__(self) -> ModalInteractiveBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self, session_id: str, bus: FrameBus) -> None:
        self._session_id = session_id
        self._bus = bus
        self._ws = await websockets.connect(self._endpoint)
        await self._ws.send(json.dumps({"type": "start", "session_id": session_id}))
        self._recv_task = asyncio.create_task(self._recv_loop(), name=f"modal-recv-{session_id}")
        log.info("modal_interactive_backend.started", session_id=session_id, endpoint=self._endpoint)

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        if self._ws:
            await self._ws.send(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        if self._ws:
            await self._ws.send(json.dumps({"type": "inject", "text": text}))

    async def swap_persona(self, persona: PersonaSpec) -> None:
        if self._ws:
            await self._ws.send(json.dumps({"type": "swap_persona", **persona}))

    async def close(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        log.info("modal_interactive_backend.closed", session_id=self._session_id)

    async def _recv_loop(self) -> None:
        if not self._ws or not self._bus:
            return
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    await self._bus.publish(AudioChunk(
                        session_id=self._session_id,
                        speaker=Speaker.COACH,
                        pcm16_16k=msg,
                        ts=time.time(),
                    ))
                else:
                    await self._handle_event(json.loads(msg))
        except websockets.exceptions.ConnectionClosed:
            log.warning("modal_interactive_backend.connection_closed", session_id=self._session_id)
            await self._bus.publish(EndOfCall(
                session_id=self._session_id,
                reason="error",
                ts=time.time(),
            ))

    async def _handle_event(self, data: dict) -> None:
        assert self._bus is not None
        match data.get("type"):
            case "transcript":
                speaker = Speaker.COACH if data["speaker"] == "coach" else Speaker.USER
                await self._bus.publish(TranscriptDelta(
                    session_id=self._session_id,
                    utterance_id=data.get("utterance_id", str(uuid.uuid4())),
                    speaker=speaker,
                    text=data["text"],
                    is_final=data.get("is_final", True),
                    ts_start=time.time(),
                ))
            case "prosody":
                speaker = Speaker.COACH if data["speaker"] == "coach" else Speaker.USER
                await self._bus.publish(ProsodyEvent(
                    session_id=self._session_id,
                    utterance_id=data.get("utterance_id", str(uuid.uuid4())),
                    speaker=speaker,
                    scores=ProsodyScores(
                        arousal=data.get("arousal", 0.0),
                        valence=data.get("valence", 0.0),
                    ),
                    ts_start=time.time(),
                    ts_end=time.time(),
                ))
            case "end_of_call":
                await self._bus.publish(EndOfCall(
                    session_id=self._session_id,
                    reason=data.get("reason", "hangup"),
                    ts=time.time(),
                ))
            case unknown:
                log.warning("modal_interactive_backend.unknown_event", type=unknown)
