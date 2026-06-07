"""LiveKit room audio adapter — mirrors twilio_stream.py.

Provides two stream implementations behind the same duck-typed interface:
  - LiveKitRoomStream: production, backed by livekit-rtc (imported lazily)
  - FakeRoomStream: test, backed by an asyncio.Queue

And LiveKitCallerParticipant, the VoiceParticipant implementation backed by
either stream — directly analogous to TwilioCallerParticipant.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk
from rehearse.audio.participants import ParticipantConfig, SpeakRequest, VoiceParticipant
from rehearse.types import Speaker


# ---------------------------------------------------------------------------
# Production stream — requires livekit-rtc (lazy import)
# ---------------------------------------------------------------------------

class LiveKitRoomStream:
    """Wrap a LiveKit room's audio I/O for the call runtime.

    Yields inbound caller audio at 16 kHz PCM16 (resampled by AudioStream).
    Sends outbound provider audio as a published LocalAudioTrack.
    Forwards DataChannel JSON messages to all room participants.

    rtc is imported lazily so this module is importable in envs without
    livekit-rtc (e.g. CI running the hermetic Tier-A test).
    """

    def __init__(self) -> None:
        self._frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._audio_source = None
        self._room = None

    async def setup(self, room: object, audio_source: object) -> None:
        """Wire room event listener and audio source after room.connect()."""
        from livekit import rtc  # noqa: PLC0415

        self._room = room
        self._audio_source = audio_source

        @room.on("track_subscribed")  # type: ignore[union-attr]
        def _on_track(track: object, pub: object, participant: object) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:  # type: ignore[union-attr]
                asyncio.ensure_future(self._drain_track(track))

        # Race safety: handle tracks already present when agent joins.
        for p in room.remote_participants.values():  # type: ignore[union-attr]
            for pub in p.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    asyncio.ensure_future(self._drain_track(pub.track))

    async def _drain_track(self, track: object) -> None:
        from livekit import rtc  # noqa: PLC0415

        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        async for event in stream:
            await self._frame_queue.put(bytes(event.frame.data))
        await self._frame_queue.put(None)  # sentinel: track ended

    async def inbound(self) -> AsyncIterator[bytes]:
        """Yield inbound caller audio as PCM16 mono 16 kHz chunks."""
        while True:
            frame = await self._frame_queue.get()
            if frame is None:
                return
            yield frame

    async def send(self, pcm16_16k: bytes) -> None:
        """Publish provider PCM audio as the agent's audio track."""
        if self._audio_source is None or not pcm16_16k:
            return
        from livekit import rtc  # noqa: PLC0415

        n = len(pcm16_16k) // 2
        lk_frame = rtc.AudioFrame(
            data=pcm16_16k,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=n,
        )
        await self._audio_source.capture_frame(lk_frame)

    async def publish_data(self, data: bytes) -> None:
        """Send a DataChannel message to all room participants."""
        if self._room is None:
            return
        await self._room.local_participant.publish_data(data, reliable=True)  # type: ignore[union-attr]

    def close(self) -> None:
        """Signal end of inbound stream."""
        self._frame_queue.put_nowait(None)


# ---------------------------------------------------------------------------
# Fake stream — no external deps; for hermetic tests
# ---------------------------------------------------------------------------

class FakeRoomStream:
    """In-process fake for tests.

    Yields frames from the queue provided at construction; captures
    send() calls to received_audio and publish_data() calls to
    datachannel_messages.
    """

    def __init__(self, frame_queue: asyncio.Queue[bytes | None]) -> None:
        self._queue = frame_queue
        self.received_audio: list[bytes] = []
        self.datachannel_messages: list[dict] = []

    async def inbound(self) -> AsyncIterator[bytes]:
        """Yield PCM frames from the queue until a None sentinel.

        Sleeps 40 ms per frame so tests that depend on timing (e.g. a backend
        that sleeps before publishing frames) have time to run.
        """
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            await asyncio.sleep(0.04)
            yield frame

    async def send(self, pcm16_16k: bytes) -> None:
        self.received_audio.append(pcm16_16k)

    async def publish_data(self, data: bytes) -> None:
        try:
            self.datachannel_messages.append(json.loads(data))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VoiceParticipant — analogous to TwilioCallerParticipant
# ---------------------------------------------------------------------------

class LiveKitCallerParticipant(VoiceParticipant):
    """VoiceParticipant backed by a LiveKit room stream (real or fake).

    audio_stream() yields caller PCM and publishes AudioChunk(USER) frames.
    receive_audio() pushes provider PCM back to the room.
    """

    def __init__(self, stream: object, session_id: str) -> None:
        self._stream = stream
        self._session_id = session_id

    @property
    def config(self) -> ParticipantConfig:
        return ParticipantConfig(
            participant_id=self._session_id,
            role="caller",
            backend="livekit_stream",
        )

    async def receive_audio(self, pcm16_16k: bytes) -> None:
        await self._stream.send(pcm16_16k)  # type: ignore[union-attr]

    async def say(self, request: SpeakRequest) -> None:
        return None

    async def audio_stream(self, bus: FrameBus) -> AsyncIterator[bytes]:
        async for chunk in self._stream.inbound():  # type: ignore[union-attr]
            await bus.publish(
                AudioChunk(
                    session_id=self._session_id,
                    speaker=Speaker.USER,
                    pcm16_16k=chunk,
                    ts=0.0,
                )
            )
            yield chunk

    async def run(self, bus: FrameBus) -> None:
        async for _ in self.audio_stream(bus):
            continue
