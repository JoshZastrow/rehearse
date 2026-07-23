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
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, ProviderReady, TranscriptDelta
from rehearse.types import ProsodyScores, Speaker

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection
    from rehearse.storage import ArtifactStore

log = structlog.get_logger(__name__)

# Maps wire labels to Speaker. Both legacy ("coach"/"user") and parameterized
# ("provider"/"caller") labels are handled. Unknown labels default to GUIDE.
_SPEAKER_MAP: dict[str, Speaker] = {
    "coach": Speaker.GUIDE,
    "provider": Speaker.GUIDE,
    "user": Speaker.USER,
    "caller": Speaker.USER,
}


class ModalInteractiveBackend:
    """ConversationBackend that proxies audio to a Modal-hosted WebSocket inference server."""

    def __init__(self, endpoint: str, store: ArtifactStore | None = None) -> None:
        self._endpoint = endpoint
        self._store = store
        self._ws: ClientConnection | None = None
        self._recv_task: asyncio.Task | None = None
        self._session_id = ""
        self._bus: FrameBus | None = None
        self._hangup_logged = False

    async def __aenter__(self) -> ModalInteractiveBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    _CONNECT_ATTEMPTS = 30   # 30 × 5s = 150s total — covers a worst-case cold GPU
    _CONNECT_RETRY_DELAY = 5.0  # seconds between retries

    async def start(self, session_id: str, bus: FrameBus) -> None:
        self._session_id = session_id
        self._bus = bus
        last_exc: Exception | None = None
        for attempt in range(self._CONNECT_ATTEMPTS):
            try:
                self._ws = await websockets.connect(self._endpoint)
                break
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "modal_interactive_backend.connect_attempt_failed",
                    session_id=session_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < self._CONNECT_ATTEMPTS - 1:
                    await asyncio.sleep(self._CONNECT_RETRY_DELAY)
        else:
            log.error(
                "modal_interactive_backend.connect_failed",
                session_id=session_id,
                endpoint=self._endpoint,
                error=str(last_exc),
            )
            await bus.publish(EndOfCall(session_id=session_id, reason="error", ts=time.time()))
            return
        await self._ws.send(json.dumps({"type": "start", "session_id": session_id}))
        self._recv_task = asyncio.create_task(self._recv_loop(), name=f"modal-recv-{session_id}")
        # The socket only opens once the model server has finished loading, so a
        # successful connect is the end of the scale-to-zero cold start: tell the
        # UI the provider is ready to talk (forwarded to the browser as
        # {"type": "provider_ready"} by the LiveKit datachannel bridge).
        await bus.publish(ProviderReady(session_id=session_id, ts=time.time()))
        log.info("modal_interactive_backend.started", session_id=session_id, endpoint=self._endpoint)

    async def _send(self, payload: bytes | str) -> None:
        """Send if connected; drop silently after the server hangs up.

        The server ends a session on its idle timeout and closes the socket;
        the recv loop publishes EndOfCall for that. Audio already in flight
        must not crash the session pump — hangup mid-send is normal telephony.
        """
        if not self._ws:
            return
        try:
            await self._ws.send(payload)
        except websockets.exceptions.ConnectionClosed:
            if not self._hangup_logged:
                self._hangup_logged = True
                log.info(
                    "modal_interactive_backend.send_after_close_dropped",
                    session_id=self._session_id,
                )

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        await self._send(pcm16_16k)

    async def say(self, request: object) -> None:
        """Speaker protocol: delegates to inject_speech."""
        await self.inject_speech(getattr(request, "text", str(request)))

    async def inject_speech(self, text: str) -> None:
        await self._send(json.dumps({"type": "inject", "text": text}))

    async def swap_persona(self, persona: PersonaSpec) -> None:
        await self._send(json.dumps({"type": "swap_persona", **persona}))

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
                        speaker=Speaker.GUIDE,
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
                speaker = _SPEAKER_MAP.get(data.get("speaker", ""), Speaker.GUIDE)
                await self._bus.publish(TranscriptDelta(
                    session_id=self._session_id,
                    utterance_id=data.get("utterance_id", str(uuid.uuid4())),
                    speaker=speaker,
                    text=data["text"],
                    is_final=data.get("is_final", True),
                    ts_start=time.time(),
                ))
            case "prosody":
                speaker = _SPEAKER_MAP.get(data.get("speaker", ""), Speaker.GUIDE)
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
            case "profile":
                if self._store and self._session_id:
                    await self._store.append(self._session_id, "profile.jsonl", json.dumps(data))
                log.info("modal_interactive_backend.profile", **{k: v for k, v in data.items() if k != "type"})
            case unknown:
                log.warning("modal_interactive_backend.unknown_event", type=unknown)
