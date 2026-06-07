"""LiveKit voice agent for Rehearse — bridges a WebRTC room to the Moshi backend.

Thin entrypoint: builds a LiveKitRoomStream from the real rtc.Room, mints a
session, then delegates to run_livekit_session() (importable in tests).

Environment variables:
  LIVEKIT_URL                    ws://localhost:7880 (or wss:// for cloud)
  LIVEKIT_API_KEY                devkey
  LIVEKIT_API_SECRET             secret
  INTERACTIVE_PROVIDER_ENDPOINT  wss://...modal.run/ws
  SESSION_ROOT                   sessions (optional; default: ./sessions)
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions

# Make the repo root importable when run from web/livekit/agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

load_dotenv(Path(__file__).parent / ".env", override=False)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    await ctx.wait_for_participant()

    endpoint = (
        os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT")
        or os.environ.get("INTERACTIVE_MODAL_ENDPOINT", "")
    )
    if not endpoint:
        log.error("INTERACTIVE_PROVIDER_ENDPOINT not set — agent cannot start")
        return

    from rehearse.audio.livekit_stream import LiveKitRoomStream
    from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
    from rehearse.session.livekit_session import run_livekit_session, write_session_manifest
    from rehearse.storage import LocalFilesystemStore

    session_id = str(uuid.uuid4())
    session_root = Path(os.environ.get("SESSION_ROOT", "sessions"))
    store = LocalFilesystemStore(session_root, "http://localhost:8000")
    write_session_manifest(store, session_id)

    # Build and publish the agent's outbound audio track.
    audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
    await ctx.room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    stream = LiveKitRoomStream()
    await stream.setup(ctx.room, audio_source)

    backend = ModalInteractiveBackend(endpoint)
    await run_livekit_session(stream, session_id, backend, store=store, skip_consent=True)


if __name__ == "__main__":
    agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
