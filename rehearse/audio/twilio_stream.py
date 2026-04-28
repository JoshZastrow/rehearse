"""Read and write Twilio Media Streams websocket messages.

This file hides the Twilio-specific websocket event format from the rest of the
runtime. It performs the handshake, decodes inbound audio, and encodes outbound
assistant audio back into Twilio's expected media events.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

from fastapi import WebSocket

from rehearse.audio.mulaw import decode_mulaw, encode_pcm16
from rehearse.audio.resample import downsample_16k_to_8k, upsample_8k_to_16k


class TwilioStream:
    """Wrap a Twilio Media Streams websocket connection."""

    def __init__(self, ws: WebSocket) -> None:
        """Store the websocket used for one live Twilio stream."""
        self._ws = ws
        self._start: dict[str, Any] | None = None
        self._stream_sid: str | None = None
        self._session_id: str | None = None

    async def __aenter__(self) -> TwilioStream:
        """Wait for the Twilio start event and return a ready stream wrapper."""
        while True:
            event = await self._ws.receive_json()
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            if kind == "connected":
                continue
            if kind != "start":
                continue
            start = event.get("start")
            if not isinstance(start, dict):
                raise ValueError("twilio start event missing start payload")
            self._start = start
            self._stream_sid = _require_str(start, "streamSid")
            custom = start.get("customParameters")
            if isinstance(custom, dict):
                self._session_id = custom.get("session_id")
            if self._session_id is None:
                self._session_id = start.get("callSid") or self._stream_sid
            return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the async context manager for the stream wrapper."""
        return None

    @property
    def session_id(self) -> str:
        """Return the session id attached to this Twilio stream."""
        if self._session_id is None:
            raise RuntimeError("TwilioStream not connected")
        return self._session_id

    @property
    def stream_sid(self) -> str:
        """Return Twilio's stream SID for this websocket session."""
        if self._stream_sid is None:
            raise RuntimeError("TwilioStream not connected")
        return self._stream_sid

    async def inbound(self) -> AsyncIterator[bytes]:
        """Yield inbound user audio as PCM16 mono 16kHz chunks."""

        while True:
            event = await self._ws.receive_json()
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            if kind == "stop":
                return
            if kind != "media":
                continue
            media = event.get("media")
            if not isinstance(media, dict):
                continue
            payload = media.get("payload")
            if not isinstance(payload, str):
                continue
            try:
                mulaw = base64.b64decode(payload, validate=True)
            except (ValueError, TypeError):
                continue
            pcm8k = decode_mulaw(mulaw)
            yield upsample_8k_to_16k(pcm8k)

    async def send(self, pcm16_16k: bytes) -> None:
        """Send assistant PCM16 audio back to Twilio as a media event."""

        pcm8k = downsample_16k_to_8k(pcm16_16k)
        mulaw = encode_pcm16(pcm8k)
        payload = base64.b64encode(mulaw).decode("ascii")
        await self._ws.send_json(
            {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload},
            }
        )

    async def send_mark(self, name: str) -> None:
        """Send a Twilio mark event and return when it is queued."""

        await self._ws.send_json(
            {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": name},
            }
        )


def _require_str(data: dict[str, Any], key: str) -> str:
    """Return one required string field from a Twilio start payload."""
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"twilio start event missing {key}")
    return value
