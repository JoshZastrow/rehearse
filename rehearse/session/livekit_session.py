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

import structlog

from rehearse.audio.livekit_stream import LiveKitCallerParticipant
from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall, ProviderReady, TranscriptDelta
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
    """Forward ProviderReady / TranscriptDelta / EndOfCall frames to the DataChannel."""
    async for frame in bus.subscribe():
        if isinstance(frame, ProviderReady):
            # The model is connected (cold start over) — let the UI reveal the call.
            msg = {"v": _DC_SCHEMA_VERSION, "type": "provider_ready"}
            await stream.publish_data(json.dumps(msg).encode())  # type: ignore[union-attr]
        elif isinstance(frame, TranscriptDelta):
            speaker = "agent" if frame.speaker == Speaker.GUIDE else "user"
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


async def finalize_and_bill(
    store: LocalFilesystemStore,
    session_id: str,
    *,
    billing_store: object | None = None,
    clerk_user_id: str | None = None,
    stripe_customer_id: str | None = None,
    meter_reporter=None,
    now=None,
):
    """Stamp `finalized_at`, compute credits, and record the usage/meter event.

    Called from the agent's `serve_session` finally block after a session runs.
    Two responsibilities:

      1. Session lifecycle — always writes `finalized_at` + marks the manifest
         `completion_status="complete"`. This is what makes the wall-clock
         billable window exist; without it `session_credits()` can't run.
      2. Billing (only when `billing_store` + `clerk_user_id` are supplied) —
         inserts an idempotent `usage_events` row and reports a Stripe meter
         event for the credits, but only on a *newly* inserted row so a retried
         finalize can't double-report.

    Fully guarded: never raises. A billing failure must not crash the disconnect
    path, so everything is wrapped and logged at warning. Returns the computed
    `SessionCost`, or None if finalize failed.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from rehearse.billing.cost import session_credits  # noqa: PLC0415
    from rehearse.billing.stripe_meter import report_meter_event  # noqa: PLC0415

    finalized_at = now or datetime.now(UTC)

    def _mark_final(session):
        session.finalized_at = finalized_at
        if session.completion_status == "in_progress":
            session.completion_status = "complete"
        return session

    try:
        updated = await store.update_session(session_id, _mark_final)
    except Exception as exc:  # noqa: BLE001 — finalize must not crash disconnect
        log.warning("livekit_session.finalize_failed", session_id=session_id, error=str(exc))
        return None

    try:
        cost = session_credits(updated)
    except Exception as exc:  # noqa: BLE001
        log.warning("livekit_session.cost_failed", session_id=session_id, error=str(exc))
        return None

    log.info(
        "livekit_session.billed",
        session_id=session_id,
        gpu_seconds=round(cost.gpu_seconds, 2),
        credits=round(cost.credits, 4),
        billed_usd=round(cost.billed_usd, 4),
    )

    if billing_store is None or clerk_user_id is None:
        return cost  # local/hermetic path — no ledger, no Stripe.

    try:
        import asyncio  # noqa: PLC0415

        inserted = await asyncio.to_thread(
            billing_store.record_usage,
            session_id,
            clerk_user_id,
            cost.gpu_seconds,
            cost.credits,
        )
        if inserted:
            reporter = meter_reporter or report_meter_event
            await asyncio.to_thread(
                reporter, stripe_customer_id, cost.credits, session_id
            )
        else:
            log.info("livekit_session.usage_already_recorded", session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("livekit_session.meter_failed", session_id=session_id, error=str(exc))

    return cost
