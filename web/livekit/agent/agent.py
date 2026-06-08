"""LiveKit voice agent for Rehearse — bridges a WebRTC room to the Moshi backend.

Connects directly to a LiveKit room via rtc.Room (no livekit-agents framework),
then delegates to run_livekit_session() which handles all audio routing and
artifact writing.

Structure:
  run_agent()     — resolves env, builds the real room/track/stream/backend (needs
                    livekit-rtc), then calls serve_session().
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
        await room.disconnect()  # type: ignore[union-attr]
        return

    log.info("starting session %s", session_id)
    try:
        await run_livekit_session(stream, session_id, backend, store=store, skip_consent=True)
    finally:
        await room.disconnect()  # type: ignore[union-attr]
        log.info("session %s complete", session_id)


async def run_agent() -> None:
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
    log.info("connected — waiting for participant")

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

    backend = ModalInteractiveBackend(endpoint) if endpoint else None
    await serve_session(room, stream, backend, store, session_id)


def main() -> None:
    load_dotenv(_REPO_ROOT / ".env", override=False)
    load_dotenv(Path(__file__).parent / ".env", override=False)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
