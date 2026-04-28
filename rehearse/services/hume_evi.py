"""Bridge Hume EVI websocket events into runtime frames.

This file wraps the Hume realtime chat socket used during a live call. It sends
user audio into Hume, converts Hume events into runtime frames, and handles a
small reconnect policy for transient websocket failures.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import wave
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from hume.client import AsyncHumeClient
from hume.empathic_voice.types.audio_input import AudioInput

from rehearse.audio.resample import resample_pcm16
from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, TranscriptDelta
from rehearse.types import ProsodyScores, Speaker


class HumeEVIClient:
    """Bridge a Hume realtime chat socket into runtime frames."""

    def __init__(
        self,
        *,
        api_key: str,
        config_id: str,
        bus: FrameBus,
        session_id: str,
        connect_fn: Callable[..., Any] | None = None,
        reconnect_backoff_s: float = 0.1,
    ) -> None:
        """Store connection settings and test seams for one Hume session."""
        self._api_key = api_key
        self._config_id = config_id
        self._bus = bus
        self._session_id = session_id
        self._connect_fn = (
            connect_fn or AsyncHumeClient(api_key=api_key).empathic_voice.chat.connect
        )
        self._reconnect_backoff_s = reconnect_backoff_s
        self._stack: AsyncExitStack | None = None
        self._socket: Any = None
        self._started_at = time.monotonic()
        self._utterance_counter = 0

    async def __aenter__(self) -> HumeEVIClient:
        """Open the Hume websocket connection and return the adapter."""
        await self._connect()
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Close any open Hume websocket resources."""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._socket = None

    async def send_audio(self, pcm16_16k: bytes) -> None:
        """Send one chunk of user PCM16 audio into Hume."""

        if self._socket is None:
            raise RuntimeError("HumeEVIClient not connected")
        payload = base64.b64encode(pcm16_16k).decode("ascii")
        await self._socket.send_audio_input(AudioInput(data=payload))

    async def run_event_loop(self) -> None:
        """Read Hume events until the socket closes and publish runtime frames."""

        attempts = 0
        while True:
            try:
                assert self._socket is not None
                async for event in self._socket:
                    await self._handle_event(event)
                return
            except Exception:
                attempts += 1
                if attempts > 1:
                    await self._bus.publish(
                        EndOfCall(
                            session_id=self._session_id,
                            reason="error",
                            ts=self._elapsed_s(),
                        )
                    )
                    return
                await asyncio.sleep(self._reconnect_backoff_s)
                await self._reconnect()

    async def swap_config(self, config_id: str, system_prompt: str | None = None) -> None:
        """Swap the active Hume config during a call when that feature exists."""

        raise NotImplementedError(
            f"live config swap not implemented yet for {config_id} / {system_prompt!r}"
        )

    async def _connect(self) -> None:
        """Open a fresh Hume chat websocket and store the socket object."""
        self._stack = AsyncExitStack()
        self._socket = await self._stack.enter_async_context(
            self._connect_fn(
                config_id=self._config_id,
                api_key=self._api_key,
                session_settings={
                    "custom_session_id": self._session_id,
                    "audio": {
                        "channels": 1,
                        "encoding": "linear16",
                        "sample_rate": 16_000,
                    },
                },
            )
        )

    async def _reconnect(self) -> None:
        """Close the old socket and open a new Hume websocket connection."""
        if self._stack is not None:
            await self._stack.aclose()
        await self._connect()

    async def _handle_event(self, event: Any) -> None:
        """Dispatch one Hume event to the correct runtime-frame handler."""
        event_type = getattr(event, "type", None)
        if event_type == "audio_output":
            await self._publish_audio_output(event)
            return
        if event_type == "user_message":
            await self._publish_user_message(event)
            return
        if event_type == "assistant_message":
            await self._publish_assistant_message(event)
            return
        if event_type == "assistant_prosody":
            return
        if event_type == "user_interruption":
            return
        if event_type == "error":
            raise RuntimeError(getattr(event, "message", "hume websocket error"))

    async def _publish_audio_output(self, event: Any) -> None:
        """Convert one Hume audio chunk into a runtime audio frame."""
        wav_bytes = base64.b64decode(event.data)
        pcm48k = _decode_wav_pcm16(wav_bytes)
        pcm16k = resample_pcm16(pcm48k, src_rate=48_000, dst_rate=16_000)
        await self._bus.publish(
            AudioChunk(
                session_id=self._session_id,
                speaker=Speaker.COACH,
                pcm16_16k=pcm16k,
                ts=self._elapsed_s(),
            )
        )

    async def _publish_user_message(self, event: Any) -> None:
        """Publish transcript and prosody frames for one user utterance."""
        utterance_id = self._new_utterance_id("user")
        text = getattr(getattr(event, "message", None), "content", "") or ""
        begin_ms = float(getattr(getattr(event, "time", None), "begin", 0))
        end_ms = float(getattr(getattr(event, "time", None), "end", 0))
        await self._bus.publish(
            TranscriptDelta(
                session_id=self._session_id,
                utterance_id=utterance_id,
                speaker=Speaker.USER,
                text=text,
                is_final=not bool(getattr(event, "interim", False)),
                ts_start=begin_ms / 1000.0,
                ts_end=end_ms / 1000.0,
            )
        )

        scores = _extract_scores(getattr(getattr(event, "models", None), "prosody", None))
        await self._bus.publish(
            ProsodyEvent(
                session_id=self._session_id,
                utterance_id=utterance_id,
                speaker=Speaker.USER,
                scores=scores,
                ts_start=begin_ms / 1000.0,
                ts_end=end_ms / 1000.0,
            )
        )

    async def _publish_assistant_message(self, event: Any) -> None:
        """Publish one assistant transcript frame from a Hume message event."""
        text = getattr(getattr(event, "message", None), "content", "") or ""
        now = self._elapsed_s()
        await self._bus.publish(
            TranscriptDelta(
                session_id=self._session_id,
                utterance_id=self._new_utterance_id("assistant"),
                speaker=Speaker.COACH,
                text=text,
                is_final=True,
                ts_start=now,
                ts_end=now,
            )
        )

    def _elapsed_s(self) -> float:
        """Return seconds elapsed since this Hume session started."""
        return time.monotonic() - self._started_at

    def _new_utterance_id(self, prefix: str) -> str:
        """Return a simple unique utterance id for the current session."""
        self._utterance_counter += 1
        return f"{prefix}-{self._utterance_counter}"


def _decode_wav_pcm16(wav_bytes: bytes) -> bytes:
    """Read PCM16 frame bytes out of a WAV payload."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.readframes(wav_file.getnframes())


def _extract_scores(prosody: Any) -> ProsodyScores:
    """Convert Hume prosody scores into the runtime `ProsodyScores` model."""
    scores_obj = getattr(prosody, "scores", None)
    if scores_obj is None:
        return ProsodyScores(arousal=0.0, valence=0.0, emotions={})
    emotions = {
        key: float(value)
        for key, value in scores_obj.model_dump(exclude_none=True).items()
        if isinstance(value, (int, float))
    }
    return ProsodyScores(arousal=0.0, valence=0.0, emotions=emotions)
