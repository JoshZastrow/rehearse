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

    async def __aenter__(self) -> _ScriptedCoachBackend:
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


class LocalTtsProviderBackend(_ScriptedCoachBackend):
    """Scripted provider backend whose audio is synthesized locally with Pocket TTS.

    Same deterministic transcript/prosody turns as the scripted base backend, but
    the provider audio is real speech generated on CPU by Pocket TTS instead of a
    canned square wave. Used by the browser-driven hermetic e2e
    (tests/e2e/runner.py) so the artifacts contain audible provider speech an
    engineer can play back.

    Pocket TTS (`load_model`, `generate_audio`) runs in a worker thread so the
    asyncio event loop is never blocked. The model + voice load once per instance.

    Note: ``Speaker.PROVIDER``/``Speaker.CALLER`` are the current vocabulary; they
    serialize to the canonical ``"coach"``/``"user"`` (see rehearse.types.Speaker),
    so on-disk artifacts still land under ``audio/coach`` etc.
    """

    _PROVIDER_TEXT = (
        "Hello, let's begin your rehearsal. "
        "Take a breath and tell me what's on your mind."
    )
    _VOICE = "alba"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._voice_state = None

    def _synth_frames(self) -> list[bytes]:
        """Load the model (first call) and return 20 ms PCM16 @ 16 kHz frames.

        Runs synchronously — call via asyncio.to_thread.
        """
        import numpy as np  # noqa: PLC0415
        from pocket_tts import TTSModel  # noqa: PLC0415

        if self._model is None:
            self._model = TTSModel.load_model()
            self._voice_state = self._model.get_state_for_audio_prompt(self._VOICE)

        audio = self._model.generate_audio(self._voice_state, self._PROVIDER_TEXT)
        samples = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)

        # Resample model sample-rate -> 16 kHz via linear interpolation.
        src_rate = int(self._model.sample_rate)
        if src_rate != 16000 and samples.size:
            n_out = int(round(samples.size * 16000 / src_rate))
            src_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
            dst_x = np.linspace(0.0, 1.0, n_out, endpoint=False)
            samples = np.interp(dst_x, src_x, samples).astype(np.float32)

        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

        # Slice into 20 ms frames (320 samples = 640 bytes); pad the tail.
        frame_bytes = 640
        frames = [pcm16[i : i + frame_bytes] for i in range(0, len(pcm16), frame_bytes)]
        if frames and len(frames[-1]) < frame_bytes:
            frames[-1] = frames[-1] + b"\x00" * (frame_bytes - len(frames[-1]))
        return frames or [b"\x00" * frame_bytes]

    async def start(self, session_id: str, bus: object) -> None:
        self._session_id = session_id

        # Synthesize before the burst (model load + generate happen off-loop).
        provider_frames = await asyncio.to_thread(self._synth_frames)

        await asyncio.sleep(0.2)  # let writers subscribe before burst

        await bus.publish(  # type: ignore[union-attr]
            TranscriptDelta(
                session_id=session_id,
                utterance_id="provider-1",
                speaker=Speaker.PROVIDER,
                text=self._PROVIDER_TEXT,
                is_final=True,
                ts_start=0.0,
                ts_end=float(len(provider_frames) * 0.02),
            )
        )
        await bus.publish(  # type: ignore[union-attr]
            ProsodyEvent(
                session_id=session_id,
                utterance_id="provider-1",
                speaker=Speaker.PROVIDER,
                scores=ProsodyScores(arousal=0.4, valence=0.3, emotions={"calm": 0.6}),
                ts_start=0.0,
                ts_end=float(len(provider_frames) * 0.02),
            )
        )
        for frame in provider_frames:
            await bus.publish(  # type: ignore[union-attr]
                AudioChunk(
                    session_id=session_id,
                    speaker=Speaker.PROVIDER,
                    pcm16_16k=frame,
                    ts=0.0,
                )
            )
        await bus.publish(  # type: ignore[union-attr]
            TranscriptDelta(
                session_id=session_id,
                utterance_id="caller-1",
                speaker=Speaker.CALLER,
                text="Okay, I'm ready.",
                is_final=True,
                ts_start=0.5,
                ts_end=0.9,
            )
        )
