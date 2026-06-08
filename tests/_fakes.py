"""Shared test fakes for the Rehearse test suite.

Provides:
  _ScriptedCoachBackend  — GPU-free ConversationBackend (lifted from test_call_server_e2e.py)
  FakeRoomStream         — In-process LiveKit room stream for hermetic tests
  FakeRoom               — Minimal rtc.Room stand-in for agent.serve_session() tests
"""

from __future__ import annotations

import asyncio
import struct

from rehearse.frames import AudioChunk, ProsodyEvent, TranscriptDelta
from rehearse.types import ProsodyScores, Speaker


class FakeRoom:
    """Minimal stand-in for livekit rtc.Room — enough for agent.serve_session().

    Exposes a truthy ``remote_participants`` (so the participant-wait loop exits)
    and records ``disconnect()`` so tests can assert the agent cleaned up.
    """

    def __init__(self, *, participants: int = 1) -> None:
        self.remote_participants = {f"p{i}": object() for i in range(participants)}
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True


class _ScriptedCoachBackend:
    """Fake ConversationBackend that emits a short scripted coach turn.

    Publishes a coach transcript line plus several frames of coach audio so
    AudioRecorder produces a non-trivial audio.wav and a per-role turn WAV.
    Returns from start() immediately so the session drives to completion off
    the caller's audio stream.

    Important: start() sleeps 0.2 s before publishing so the artifact writers
    (which each run several await store-registration calls) have time to
    subscribe to the bus before frames are emitted.
    """

    # 20 ms of mono PCM16 @ 16 kHz = 320 samples = 640 bytes.
    _COACH_FRAME = struct.pack("<320h", *([1200, -1200] * 160))

    def __init__(self) -> None:
        self.received_caller_audio: list[bytes] = []
        self._session_id = ""

    async def __aenter__(self) -> "_ScriptedCoachBackend":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def start(self, session_id: str, bus: object) -> None:
        self._session_id = session_id
        await asyncio.sleep(0.2)  # let writers subscribe before burst

        await bus.publish(  # type: ignore[union-attr]
            TranscriptDelta(
                session_id=session_id,
                utterance_id="coach-1",
                speaker=Speaker.COACH,
                text="Hello, let's begin your rehearsal.",
                is_final=True,
                ts_start=0.0,
                ts_end=0.4,
            )
        )
        await bus.publish(  # type: ignore[union-attr]
            ProsodyEvent(
                session_id=session_id,
                utterance_id="coach-1",
                speaker=Speaker.COACH,
                scores=ProsodyScores(arousal=0.4, valence=0.3, emotions={"calm": 0.6}),
                ts_start=0.0,
                ts_end=0.4,
            )
        )
        for _ in range(8):
            await bus.publish(  # type: ignore[union-attr]
                AudioChunk(
                    session_id=session_id,
                    speaker=Speaker.COACH,
                    pcm16_16k=self._COACH_FRAME,
                    ts=0.0,
                )
            )
        await bus.publish(  # type: ignore[union-attr]
            TranscriptDelta(
                session_id=session_id,
                utterance_id="user-1",
                speaker=Speaker.USER,
                text="Okay, I'm ready.",
                is_final=True,
                ts_start=0.5,
                ts_end=0.9,
            )
        )

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        self.received_caller_audio.append(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        return None

    async def swap_persona(self, persona: object) -> None:
        return None

    async def say(self, request: object) -> None:
        return None

    async def close(self) -> None:
        return None
