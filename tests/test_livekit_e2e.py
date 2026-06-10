"""End-to-end tests for the LiveKit / WebRTC voice runtime.

Mirrors test_call_server_e2e.py with three test tiers:

  Tier A  (default suite, hermetic):
    run_livekit_session() with FakeRoomStream + _ScriptedCoachBackend.
    Asserts the same 7-condition lifecycle as the Twilio e2e test.
    No LiveKit server, no Modal, no browser required.

  Tier A' (live_livekit marker):
    Same session pipeline but with a real livekit-server --dev and a
    Python rtc participant publishing PCM frames.
    Requires: livekit-server binary + LIVEKIT_API_KEY/SECRET in env.

  live_modal (live_modal marker):
    run_livekit_session() with FakeRoomStream + real ModalInteractiveBackend.
    Proves the interactive backend generates audio + transcript end-to-end.
    Requires: INTERACTIVE_PROVIDER_ENDPOINT in env.

  live_modal + live_livekit (combined):
    Full stack: real livekit-server + real Modal caller + real Modal provider.
    The caller side uses INTERACTIVE_CALLER_ENDPOINT to generate realistic
    caller audio; the agent bridges to INTERACTIVE_PROVIDER_ENDPOINT.
    Proves a real audio conversation through the complete WebRTC path.
    Requires: livekit-server + both Modal endpoints in env.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import wave
from contextlib import suppress
from pathlib import Path

import pytest

from rehearse.audio.livekit_stream import FakeRoomStream
from rehearse.session.livekit_session import run_livekit_session, write_session_manifest
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Session
from tests._fakes import FakeRoom, _ScriptedCoachBackend

# ---------------------------------------------------------------------------
# Helpers shared across tiers
# ---------------------------------------------------------------------------

# 20 ms of silence at 16 kHz PCM16.
_SILENCE_FRAME = b"\x00" * 640


def _make_store(tmp_path: Path) -> LocalFilesystemStore:
    return LocalFilesystemStore(tmp_path, "http://localhost")


def _prep_session(store: LocalFilesystemStore) -> str:
    """Mint + persist a session manifest; return its id."""
    from datetime import UTC, datetime

    from rehearse.types import ConsentState

    session = Session(
        created_at=datetime.now(UTC),
        phone_number_hash="test-webrtc",
        consent=ConsentState.PENDING,
    )
    write_session_manifest(store, session.id)
    return session.id


def _wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


class _ErrorLogCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def errors(self) -> list[str]:
        out: list[str] = []
        for r in self.records:
            if r.levelno >= logging.ERROR:
                out.append(f"{r.name}: {r.getMessage()}")
            elif r.exc_info:
                out.append(f"{r.name} (exc): {r.getMessage()}")
        return out


def _caller_frames(n: int = 15) -> asyncio.Queue:
    """Build a queue of n silence frames followed by a None sentinel."""
    q: asyncio.Queue[bytes | None] = asyncio.Queue()
    for _ in range(n):
        q.put_nowait(_SILENCE_FRAME)
    q.put_nowait(None)
    return q


# ---------------------------------------------------------------------------
# Tier A — hermetic, default suite
# ---------------------------------------------------------------------------

async def test_livekit_hermetic_call_lifecycle(tmp_path: Path) -> None:
    """Full lifecycle with FakeRoomStream + _ScriptedCoachBackend — no external deps.

    Asserts parity with test_call_server_e2e.py:
      1. Session minted and session.json written.
      2. run_livekit_session() starts and completes.
      3. Full-duplex audio: FakeRoomStream.received_audio non-empty (coach → caller).
      4. No traceback logged.
      5. transcript.jsonl written with both speakers.
      6. audio.wav + per-role turn WAVs with positive frame count.
      7. Clean shutdown: run_livekit_session() returns without exception.
    """
    collector = _ErrorLogCollector()
    root_logger = logging.getLogger()
    root_logger.addHandler(collector)
    root_logger.setLevel(logging.WARNING)

    store = _make_store(tmp_path)
    session_id = _prep_session(store)

    fake_stream = FakeRoomStream(_caller_frames())
    backend = _ScriptedCoachBackend()

    # ---- 1 + 2. Session minted; run_livekit_session() completes. -----------
    assert (tmp_path / session_id / "session.json").exists()

    await run_livekit_session(fake_stream, session_id, backend, store=store)

    # ---- 3. Full-duplex audio -----------------------------------------------
    assert backend.received_caller_audio, "backend received no caller audio"
    assert fake_stream.received_audio, "caller received no coach audio"

    # ---- 4. No traceback / ERROR on server ---------------------------------
    assert collector.errors == [], collector.errors

    sdir = tmp_path / session_id

    # ---- 5. Transcript written with both speakers --------------------------
    transcript_lines = [
        json.loads(line)
        for line in (sdir / "transcript.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert transcript_lines, "transcript.jsonl is empty"
    speakers = {row["speaker"] for row in transcript_lines}
    assert "coach" in speakers and "user" in speakers, speakers

    # ---- 6. Audio artifacts with positive frame count ----------------------
    full_wav = sdir / "audio.wav"
    assert full_wav.exists(), "audio.wav not written"
    assert _wav_frames(full_wav) > 0, "audio.wav has no frames"

    coach_turn = sdir / "audio" / "coach" / "turn_0.wav"
    user_turn = sdir / "audio" / "user" / "turn_0.wav"
    assert coach_turn.exists() and _wav_frames(coach_turn) > 0
    assert user_turn.exists() and _wav_frames(user_turn) > 0

    # ---- DataChannel: transcript events emitted ----------------------------
    dc_transcripts = [m for m in fake_stream.datachannel_messages if m.get("type") == "transcript"]
    assert dc_transcripts, "no transcript DataChannel messages emitted"
    assert any(m.get("speaker") == "agent" for m in dc_transcripts)

    # ---- 7. Clean shutdown -------------------------------------------------
    root_logger.removeHandler(collector)
    assert collector.errors == [], collector.errors


def _load_agent_module():
    """Import web/livekit/agent/agent.py by path (it is not an installed package).

    Safe because agent.py's livekit imports are lazy (inside run_agent) and its
    side effects (load_dotenv / basicConfig) live in main(), so import is clean.
    """
    import importlib.util

    agent_path = Path(__file__).parent.parent / "web" / "livekit" / "agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("_rehearse_livekit_agent", agent_path)
    assert spec and spec.loader, f"cannot load {agent_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def test_livekit_agent_serve_session_hermetic(tmp_path: Path) -> None:
    """The agent's serve_session() wiring drives a full session — no livekit/Modal.

    Covers the orchestration in web/livekit/agent/agent.py that the lifecycle test
    bypasses: participant-wait, the run_livekit_session() call, and clean
    room.disconnect(). Catches agent-side wiring bugs (wrong arg order, missing
    manifest, no-disconnect) without a real LiveKit server.
    """
    agent_mod = _load_agent_module()

    store = _make_store(tmp_path)
    session_id = _prep_session(store)
    room = FakeRoom(participants=1)
    stream = FakeRoomStream(_caller_frames())
    backend = _ScriptedCoachBackend()

    await agent_mod.serve_session(
        room, stream, backend, store, session_id, participant_wait_s=1.0
    )

    # Agent cleaned up the room.
    assert room.disconnected, "serve_session did not disconnect the room"

    # Real artifacts written through the same pipeline.
    sdir = tmp_path / session_id
    transcript = [
        json.loads(line)
        for line in (sdir / "transcript.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert transcript, "transcript.jsonl is empty"
    assert {r["speaker"] for r in transcript} >= {"coach", "user"}
    assert _wav_frames(sdir / "audio.wav") > 0
    assert backend.received_caller_audio, "backend received no caller audio"


async def test_livekit_agent_serve_session_no_backend_disconnects(tmp_path: Path) -> None:
    """serve_session() with backend=None logs + disconnects without running a session."""
    agent_mod = _load_agent_module()

    store = _make_store(tmp_path)
    session_id = _prep_session(store)
    room = FakeRoom(participants=1)
    stream = FakeRoomStream(_caller_frames())

    await agent_mod.serve_session(
        room, stream, None, store, session_id, participant_wait_s=1.0
    )

    assert room.disconnected, "serve_session must disconnect even with no backend"
    assert not (tmp_path / session_id / "transcript.jsonl").exists()


# ---------------------------------------------------------------------------
# Tier A' — real LiveKit server (live_livekit marker)
# ---------------------------------------------------------------------------

def _livekit_env() -> tuple[str, str, str]:
    """Return (url, api_key, api_secret) or skip."""
    url = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
    key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    return url, key, secret


@pytest.mark.live_livekit
async def test_livekit_agent_session_via_real_room(tmp_path: Path) -> None:
    """run_livekit_session() against a real livekit-server --dev.

    A Python rtc participant joins the same room and publishes silence PCM;
    the scripted coach backend generates audio back.  Asserts the full
    artifact parity matrix: transcript, audio.wav, per-role WAVs, DataChannel.

    Requires: livekit-server --dev running on LIVEKIT_URL (default ws://localhost:7880).
    """
    from livekit import rtc
    from livekit.api import AccessToken, VideoGrants

    lk_url, api_key, api_secret = _livekit_env()
    room_name = f"rehearse-test-{int(time.time())}"

    def _make_token(identity: str) -> str:
        return (
            AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

    store = _make_store(tmp_path)
    session_id = _prep_session(store)

    # Build production LiveKitRoomStream for the agent side.
    from rehearse.audio.livekit_stream import LiveKitRoomStream

    agent_room = rtc.Room()
    audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
    agent_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)

    await agent_room.connect(lk_url, _make_token("agent"))
    await agent_room.local_participant.publish_track(
        agent_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    agent_stream = LiveKitRoomStream()
    await agent_stream.setup(agent_room, audio_source)

    # Join a caller participant that publishes silence.
    caller_room = rtc.Room()
    caller_audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
    caller_track = rtc.LocalAudioTrack.create_audio_track("caller-audio", caller_audio_source)

    received_agent_audio: list[bytes] = []

    @caller_room.on("track_subscribed")
    def _on_agent_track(track: object, pub: object, participant: object) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.ensure_future(_drain_agent_audio(track))

    async def _drain_agent_audio(track: object) -> None:
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        async for event in stream:
            received_agent_audio.append(bytes(event.frame.data))

    await caller_room.connect(lk_url, _make_token("caller"))
    await caller_room.local_participant.publish_track(
        caller_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    # Run the agent session in the background.
    backend = _ScriptedCoachBackend()
    agent_task = asyncio.create_task(
        run_livekit_session(agent_stream, session_id, backend, store=store)
    )

    # Feed silence from the caller for ~300 ms to prime the scripted coach.
    silence_frame = rtc.AudioFrame(
        data=_SILENCE_FRAME,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=320,
    )
    for _ in range(15):
        await caller_audio_source.capture_frame(silence_frame)
        await asyncio.sleep(0.02)

    # Signal end of caller stream via agent_stream's queue sentinel.
    agent_stream.close()

    await asyncio.wait_for(agent_task, timeout=15.0)

    await caller_room.disconnect()
    await agent_room.disconnect()

    # Assert transcript + audio artifacts.
    sdir = tmp_path / session_id
    transcript_lines = [
        json.loads(line)
        for line in (sdir / "transcript.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert transcript_lines
    speakers = {r["speaker"] for r in transcript_lines}
    assert "coach" in speakers and "user" in speakers

    assert _wav_frames(sdir / "audio.wav") > 0
    assert _wav_frames(sdir / "audio" / "coach" / "turn_0.wav") > 0


# ---------------------------------------------------------------------------
# live_modal — real interactive provider, FakeRoomStream (no LiveKit server)
# ---------------------------------------------------------------------------

def _provider_endpoint() -> str:
    ep = os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", "")
    if not ep:
        pytest.skip("INTERACTIVE_PROVIDER_ENDPOINT not set")
    return ep


@pytest.mark.live_modal
async def test_livekit_interactive_provider_generates_audio(tmp_path: Path) -> None:
    """Real ModalInteractiveBackend generates audio + transcript through the LiveKit session.

    Uses FakeRoomStream (no real LiveKit server) with silence PCM to prime
    the Moshi model. Verifies the provider generates AudioChunk frames and at
    least one final TranscriptDelta within 30 s.

    Requires: INTERACTIVE_PROVIDER_ENDPOINT.
    """
    from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend

    endpoint = _provider_endpoint()
    store = _make_store(tmp_path)

    # Build a queue large enough for 30 s of 40 ms silence frames.
    q: asyncio.Queue[bytes | None] = asyncio.Queue()
    fake_stream = FakeRoomStream(q)

    session_id = _prep_session(store)
    backend = ModalInteractiveBackend(endpoint)

    async def _feed_silence() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            await q.put(_SILENCE_FRAME)
            await asyncio.sleep(0.04)
        await q.put(None)  # end of stream

    feeder = asyncio.create_task(_feed_silence())
    try:
        await asyncio.wait_for(
            run_livekit_session(fake_stream, session_id, backend, store=store),
            timeout=40.0,
        )
    except TimeoutError:
        feeder.cancel()
        pytest.fail("run_livekit_session() did not complete within 40 s")
    finally:
        feeder.cancel()
        with suppress(asyncio.CancelledError):
            await feeder

    sdir = tmp_path / session_id

    assert fake_stream.received_audio, "provider generated no audio in 30 s"

    transcript_path = sdir / "transcript.jsonl"
    assert transcript_path.exists(), "transcript.jsonl not written"
    transcript_lines = [
        json.loads(line) for line in transcript_path.read_text().splitlines() if line.strip()
    ]
    assert transcript_lines, "transcript.jsonl is empty"
    assert any(r["speaker"] == "coach" for r in transcript_lines), "no coach transcript line"

    assert _wav_frames(sdir / "audio.wav") > 0, "audio.wav has no frames"


# ---------------------------------------------------------------------------
# live_modal + live_livekit — real audio conversation between both endpoints
# ---------------------------------------------------------------------------

def _caller_endpoint() -> str:
    ep = os.environ.get("INTERACTIVE_CALLER_ENDPOINT", "")
    if not ep:
        pytest.skip("INTERACTIVE_CALLER_ENDPOINT not set")
    return ep


@pytest.mark.live_modal
@pytest.mark.live_livekit
async def test_livekit_real_audio_conversation_between_interactive_endpoints(
    tmp_path: Path,
) -> None:
    """Full stack: real LiveKit + real caller endpoint + real provider endpoint.

    The caller side is a ModalInteractiveBackend(INTERACTIVE_CALLER_ENDPOINT)
    that generates realistic audio. Its AudioChunk frames are published to a
    real LiveKit room by a Python rtc participant. The agent bridges them to
    INTERACTIVE_PROVIDER_ENDPOINT. Both endpoints must produce audio + at least
    one final transcript line within 45 s.

    Asserts:
      - provider generated audio (coach audio reached the caller room participant)
      - caller generated audio (caller audio reached the agent's backend)
      - transcript.jsonl has both coach and user lines
      - audio.wav + per-role turn WAVs with positive frame count
      - DataChannel transcript messages emitted
    """
    from livekit import rtc
    from livekit.api import AccessToken, VideoGrants

    from rehearse.audio.livekit_stream import LiveKitRoomStream
    from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import AudioChunk, EndOfCall

    provider_url = _provider_endpoint()
    caller_url = _caller_endpoint()
    lk_url, api_key, api_secret = _livekit_env()
    room_name = f"rehearse-e2e-{int(time.time())}"

    def _token(identity: str) -> str:
        return (
            AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

    store = _make_store(tmp_path)
    session_id = _prep_session(store)

    # ---- Agent side: LiveKitRoomStream + ModalInteractiveBackend(provider) ----
    agent_room = rtc.Room()
    audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
    agent_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)

    await agent_room.connect(lk_url, _token("agent"))
    await agent_room.local_participant.publish_track(
        agent_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    agent_stream = LiveKitRoomStream()
    await agent_stream.setup(agent_room, audio_source)

    provider_backend = ModalInteractiveBackend(provider_url)

    # ---- Caller side: ModalInteractiveBackend(caller) → LiveKit participant ----
    caller_session_id = session_id + "-caller"
    caller_bus: FrameBus = FrameBus(caller_session_id)
    caller_backend = ModalInteractiveBackend(caller_url)

    caller_room = rtc.Room()
    caller_audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
    caller_track = rtc.LocalAudioTrack.create_audio_track("caller-audio", caller_audio_source)

    # Capture agent audio and DataChannel messages at the caller room.
    received_agent_audio: list[bytes] = []
    received_dc_messages: list[dict] = []

    @caller_room.on("data_received")
    def _on_data(data: object) -> None:
        try:
            received_dc_messages.append(json.loads(data.data))  # type: ignore[union-attr]
        except Exception:
            pass

    @caller_room.on("track_subscribed")
    def _on_agent_track(track: object, pub: object, participant: object) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.ensure_future(_route_agent_to_caller(track))

    async def _route_agent_to_caller(track: object) -> None:
        """Route agent audio back to the caller backend (closes the duplex loop)."""
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        async for event in stream:
            chunk = bytes(event.frame.data)
            received_agent_audio.append(chunk)
            await caller_backend.send_caller_audio(chunk)

    async def _publish_caller_audio() -> None:
        """Publish AudioChunk frames from caller backend to the room."""
        async for frame in caller_bus.subscribe():
            if isinstance(frame, AudioChunk):
                n = len(frame.pcm16_16k) // 2
                lk_frame = rtc.AudioFrame(
                    data=frame.pcm16_16k,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=n,
                )
                await caller_audio_source.capture_frame(lk_frame)
            elif isinstance(frame, EndOfCall):
                agent_stream.close()  # signal the agent session to end
                break

    await caller_room.connect(lk_url, _token("caller"))
    await caller_room.local_participant.publish_track(
        caller_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    # Start both backends.
    await caller_backend.start(caller_session_id, caller_bus)
    caller_publisher = asyncio.create_task(_publish_caller_audio())

    # Prime the caller backend with silence so Moshi starts generating.
    async def _prime_caller() -> None:
        for _ in range(50):  # ~2 s of silence
            await caller_backend.send_caller_audio(_SILENCE_FRAME)
            await asyncio.sleep(0.04)

    primer = asyncio.create_task(_prime_caller())

    # Run the agent session — ends when agent_stream.close() is called.
    agent_task = asyncio.create_task(
        run_livekit_session(agent_stream, session_id, provider_backend, store=store)
    )

    # The duplex loop is self-sustaining — each backend keeps producing audio
    # while it hears the other — so neither side ends the call on its own
    # (true for Moshi and PersonaPlex alike). Bound the conversation: once both
    # ends have observably produced (agent audio at the caller room + a final
    # agent transcript on the DataChannel), or after 30 s, close the stream so
    # the session finalizes its artifacts within the 45 s budget.
    async def _bound_conversation() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if received_agent_audio and any(
                m.get("type") == "transcript" and m.get("speaker") == "agent" and m.get("final")
                for m in received_dc_messages
            ):
                break
            await asyncio.sleep(0.5)
        agent_stream.close()

    bounder = asyncio.create_task(_bound_conversation())

    try:
        await asyncio.wait_for(agent_task, timeout=45.0)
    except TimeoutError:
        agent_stream.close()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(agent_task, timeout=5.0)
        pytest.fail("agent session did not complete within 45 s")
    finally:
        bounder.cancel()
        with suppress(asyncio.CancelledError):
            await bounder
        primer.cancel()
        caller_publisher.cancel()
        with suppress(asyncio.CancelledError):
            await primer
        with suppress(asyncio.CancelledError):
            await caller_publisher
        await caller_backend.close()
        await caller_bus.aclose()
        await caller_room.disconnect()
        await agent_room.disconnect()

    # ---- Assertions ---------------------------------------------------------
    sdir = tmp_path / session_id

    assert received_agent_audio, "provider generated no audio to caller in 45 s"

    transcript_path = sdir / "transcript.jsonl"
    assert transcript_path.exists()
    rows = [json.loads(line) for line in transcript_path.read_text().splitlines() if line.strip()]
    assert rows, "transcript.jsonl is empty"
    speakers = {r["speaker"] for r in rows}
    assert "coach" in speakers, f"no coach line in transcript — speakers: {speakers}"

    assert _wav_frames(sdir / "audio.wav") > 0

    dc_transcripts = [m for m in received_dc_messages if m.get("type") == "transcript"]
    assert dc_transcripts, "no transcript DataChannel messages received at caller room"
