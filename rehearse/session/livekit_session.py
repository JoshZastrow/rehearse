"""Transport-agnostic wiring for a LiveKit-backed Rehearse session.

run_livekit_session() is the LiveKit analogue of telephony.py:media_stream():
  - accepts a stream (LiveKitRoomStream in production, FakeRoomStream in tests)
  - accepts a backend (ModalInteractiveBackend in production, fake in tests)
  - creates a FrameBus, starts the DataChannel bridge, calls run_session()

Importable without livekit-rtc because the stream is pre-built by the caller.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path

import structlog

from rehearse.audio.livekit_stream import LiveKitCallerParticipant
from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall, TranscriptDelta
from rehearse.memory.memory import NullCallerMemory
from rehearse.phases.phases import PhaseBudgets
from rehearse.session.conversation import run_session
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Session, Speaker

log = structlog.get_logger(__name__)

# DataChannel JSON schema version — shared with useVoiceSession.ts
_DC_SCHEMA_VERSION = 1


def _new_session(session_id: str | None = None) -> Session:
    """Mint a new Session, optionally overriding the auto-generated id."""
    from datetime import UTC, datetime

    session = Session(
        created_at=datetime.now(UTC),
        phone_number_hash="webrtc",
        consent=ConsentState.PENDING,
    )
    if session_id is not None:
        session.id = session_id
    return session


async def run_livekit_session(
    stream: object,
    session_id: str,
    backend: object,
    *,
    store: LocalFilesystemStore,
    skip_consent: bool = True,
) -> None:
    """Run a full Rehearse session via a LiveKit room stream.

    Args:
        stream: Duck-typed room stream (LiveKitRoomStream or FakeRoomStream).
                Must implement inbound(), send(), publish_data().
        session_id: Unique session identifier (written to the artifact store).
        backend: ConversationBackend (ModalInteractiveBackend or test fake).
        store: Session artifact store (must have session.json already written).
        skip_consent: Pre-grant consent (True for web prototype & tests).
    """
    caller = LiveKitCallerParticipant(stream, session_id)

    bus = FrameBus(session_id)
    dc_task = asyncio.create_task(
        _datachannel_bridge(stream, bus),
        name=f"dc-bridge-{session_id[:8]}",
    )

    log.info("livekit_session.start", session_id=session_id)
    try:
        await run_session(
            session_id,
            caller,
            backend,  # type: ignore[arg-type]
            store=store,
            memory=NullCallerMemory(),
            budgets=PhaseBudgets(),
            skip_consent=skip_consent,
            enable_consent=False,
            bus=bus,
        )
    finally:
        dc_task.cancel()
        with suppress(asyncio.CancelledError):
            await dc_task
        log.info("livekit_session.end", session_id=session_id)


async def _datachannel_bridge(stream: object, bus: FrameBus) -> None:
    """Forward TranscriptDelta / EndOfCall frames to the DataChannel."""
    async for frame in bus.subscribe():
        if isinstance(frame, TranscriptDelta):
            speaker = "agent" if frame.speaker == Speaker.COACH else "user"
            msg = {
                "v": _DC_SCHEMA_VERSION,
                "type": "transcript",
                "id": frame.utterance_id,
                "speaker": speaker,
                "text": frame.text,
                "final": frame.is_final,
            }
            await stream.publish_data(json.dumps(msg).encode())  # type: ignore[union-attr]
        elif isinstance(frame, EndOfCall):
            msg = {"v": _DC_SCHEMA_VERSION, "type": "end_of_call", "reason": frame.reason}
            await stream.publish_data(json.dumps(msg).encode())  # type: ignore[union-attr]
            break


def write_session_manifest(store: LocalFilesystemStore, session_id: str) -> None:
    """Write a session.json synchronously (for use in test setup)."""
    session = _new_session(session_id)
    session_dir = store.session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(session.model_dump_json(indent=2))
