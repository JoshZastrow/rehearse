"""LiveKit voice agent for Rehearse — bridges a WebRTC room to the Moshi backend.

Connects directly to a LiveKit room via rtc.Room (no livekit-agents framework),
then delegates to run_livekit_session() which handles all audio routing and
artifact writing.

Structure:
  run_agent()     — resident loop: run_call() for every caller, respawning after
                    finished or crashed calls until the process is killed.
  run_call()      — one call: resolves env, builds the real room/track/stream/
                    backend (needs livekit-rtc), waits for a caller, then calls
                    serve_session().
  serve_session() — transport-agnostic over `room`: waits for a participant, runs
                    the session, disconnects. No livekit import → hermetically
                    testable with a FakeRoom + FakeRoomStream (see test_livekit_e2e).

Environment variables:
  LIVEKIT_URL                    ws://localhost:7880 (or wss:// for cloud)
  LIVEKIT_API_KEY                devkey
  LIVEKIT_API_SECRET             secret
  LIVEKIT_ROOM_NAME              rehearse-room
  INTERACTIVE_PROVIDER_ENDPOINT  wss://...modal.run/ws
  SESSION_ROOT                   sessions (optional; default: ./sessions)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Make the repo root importable when run from web/livekit/agent/.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# livekit-free imports (safe to load without livekit-rtc installed).
from rehearse.session.livekit_session import (  # noqa: E402
    run_livekit_session,
    write_session_manifest,
)

log = logging.getLogger(__name__)

# room.disconnect() can wedge on a half-dead connection (observed 2026-06-10:
# a completed session's disconnect never returned, parking the agent in the
# room for hours). Teardown is timeboxed so the process always exits; the
# server prunes the participant once the socket closes.
_DISCONNECT_TIMEOUT_S = 10.0


async def _disconnect(room: object) -> None:
    """room.disconnect() bounded by _DISCONNECT_TIMEOUT_S — never hangs."""
    try:
        async with asyncio.timeout(_DISCONNECT_TIMEOUT_S):
            await room.disconnect()  # type: ignore[union-attr]
    except TimeoutError:
        log.warning(
            "room.disconnect() timed out after %ss — exiting anyway", _DISCONNECT_TIMEOUT_S
        )


# Resident-loop pacing: poll cadence while waiting for a caller, and the pause
# between calls (also the retry delay when livekit-server isn't up yet).
_PARTICIPANT_POLL_S = 0.5
_RESPAWN_DELAY_S = 2.0


async def _wait_for_participant(room: object) -> None:
    """Block until someone is in the room — never start a session into an empty room."""
    while not room.remote_participants:  # type: ignore[union-attr]
        await asyncio.sleep(_PARTICIPANT_POLL_S)


async def serve_session(
    room: object,
    stream: object,
    backend: object | None,
    store: object,
    session_id: str,
    *,
    participant_wait_s: float = 10.0,
) -> None:
    """Wait for a participant, run the session, then disconnect.

    Transport-agnostic over `room` (a real rtc.Room or a fake exposing
    ``remote_participants`` + ``disconnect()``). This is the hermetic seam: it
    contains the agent's orchestration logic (participant wait, no-backend guard,
    session run, clean disconnect) with no livekit-rtc dependency.
    """
    deadline_iters = max(1, int(participant_wait_s / 0.1))
    for _ in range(deadline_iters):
        if room.remote_participants:  # type: ignore[union-attr]
            break
        await asyncio.sleep(0.1)
    else:
        log.warning("no participant joined after %ss — proceeding anyway", participant_wait_s)

    if backend is None:
        log.error("no backend — agent will not process audio")
        await _disconnect(room)
        return

    log.info("starting session %s", session_id)
    try:
        await run_livekit_session(stream, session_id, backend, store=store, skip_consent=True)
    finally:
        await _disconnect(room)
        log.info("session %s complete", session_id)


async def run_call() -> None:
    """One full call: connect, wait for a caller, run the session, disconnect."""
    from livekit import rtc  # noqa: PLC0415
    from livekit.api import AccessToken, VideoGrants  # noqa: PLC0415

    from rehearse.audio.livekit_stream import LiveKitRoomStream  # noqa: PLC0415
    from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend  # noqa: PLC0415
    from rehearse.storage import LocalFilesystemStore  # noqa: PLC0415

    lk_url = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
    api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    room_name = os.environ.get("LIVEKIT_ROOM_NAME", "rehearse-room")
    endpoint = os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", "")

    if not endpoint:
        log.warning("INTERACTIVE_PROVIDER_ENDPOINT not set — audio bridge will fail")

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(f"agent-{uuid.uuid4().hex[:8]}")
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()

    @room.on("participant_connected")
    def _on_participant(participant: object) -> None:
        log.info("participant connected: %s", getattr(participant, "identity", "?"))

    @room.on("disconnected")
    def _on_disconnected(reason: object = None) -> None:
        log.info("room disconnected: %s", reason)

    log.info("connecting to LiveKit room '%s' at %s", room_name, lk_url)
    await room.connect(lk_url, token)
    log.info("connected — waiting for a caller")
    await _wait_for_participant(room)

    session_id = str(uuid.uuid4())
    session_root = Path(os.environ.get("SESSION_ROOT", "sessions"))
    store = LocalFilesystemStore(session_root, "http://localhost:8000")
    write_session_manifest(store, session_id)

    audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    stream = LiveKitRoomStream()
    await stream.setup(room, audio_source)

    # End the inbound stream promptly when the caller leaves, so the session
    # completes even if the audio-track end event is delayed. Without this the
    # session lingers, swallows the next caller's audio, and the resident loop
    # never gets to serve call #2 (observed: send_after_close_dropped on the
    # prior session id, then a dtls timeout for the new participant).
    @room.on("participant_disconnected")
    def _on_caller_left(participant: object) -> None:  # noqa: ANN401
        log.info("caller left — closing stream")
        stream.close()

    backend = ModalInteractiveBackend(endpoint) if endpoint else None
    await serve_session(room, stream, backend, store, session_id)


async def run_agent() -> None:
    """Resident loop — serve calls back-to-back until the process is killed."""
    while True:
        try:
            await run_call()
            log.info("call finished — respawning for the next caller")
        except Exception:
            log.exception("call failed — respawning")
        await asyncio.sleep(_RESPAWN_DELAY_S)


def main() -> None:
    load_dotenv(_REPO_ROOT / ".env", override=False)
    load_dotenv(Path(__file__).parent / ".env", override=False)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
