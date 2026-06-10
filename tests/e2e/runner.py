"""Test-only LiveKit agent runner for the browser-driven hermetic e2e.

Joins the same LiveKit room the browser joins, but instead of the real Modal
provider it uses ``LocalTtsProviderBackend`` (scripted turns + Pocket-TTS
provider audio). Writes session artifacts to ``SESSION_ROOT`` and prints machine-readable
markers so the Playwright test can locate the session folder and sequence the
call:

  SESSION_ID=<uuid>
  SESSION_DIR=<absolute path>
  AGENT_READY            (printed after room.connect — safe to start the call)

The process exits when the browser participant leaves: the agent's audio track
ends, ``LiveKitRoomStream.inbound()`` returns, ``run_livekit_session()``
completes, and ``serve_session()`` disconnects the room. That exit is the test's
"artifacts flushed" signal.

Run via:  uv run python tests/e2e/runner.py
Env:      LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_ROOM_NAME,
          SESSION_ROOT
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from rehearse.session.livekit_session import write_session_manifest  # noqa: E402
from rehearse.storage import LocalFilesystemStore  # noqa: E402
from tests._fakes import LocalTtsProviderBackend  # noqa: E402

log = logging.getLogger("e2e.runner")


def _load_serve_session():
    """Import serve_session() from web/livekit/agent/agent.py by path."""
    agent_path = _REPO_ROOT / "web" / "livekit" / "agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("_rehearse_livekit_agent", agent_path)
    assert spec and spec.loader, f"cannot load {agent_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.serve_session


def _emit(line: str) -> None:
    """Print a marker line and flush so the parent process sees it immediately."""
    print(line, flush=True)


async def run() -> None:
    from livekit import rtc  # noqa: PLC0415
    from livekit.api import AccessToken, VideoGrants  # noqa: PLC0415

    from rehearse.audio.livekit_stream import LiveKitRoomStream  # noqa: PLC0415

    lk_url = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
    api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    room_name = os.environ.get("LIVEKIT_ROOM_NAME", "rehearse-room")
    session_root = Path(os.environ.get("SESSION_ROOT", "tests/e2e/sessions")).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())
    store = LocalFilesystemStore(session_root, "http://localhost:8000")
    write_session_manifest(store, session_id)

    _emit(f"SESSION_ID={session_id}")
    _emit(f"SESSION_DIR={session_root / session_id}")

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(f"agent-{uuid.uuid4().hex[:8]}")
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()
    audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)

    await room.connect(lk_url, token)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    stream = LiveKitRoomStream()
    await stream.setup(room, audio_source)

    # Safety: end the inbound stream promptly when the browser leaves, so the
    # session completes even if the audio-track end event is delayed.
    @room.on("participant_disconnected")
    def _on_left(participant: object) -> None:  # noqa: ANN401
        log.info("participant left — closing stream")
        stream.close()

    _emit("AGENT_READY")

    serve_session = _load_serve_session()
    backend = LocalTtsProviderBackend()
    await serve_session(room, stream, backend, store, session_id, participant_wait_s=20.0)

    _emit("AGENT_DONE")


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(run())


if __name__ == "__main__":
    main()
