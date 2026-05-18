from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rehearse.pipeline.outbox import EventOutbox
from rehearse.storage import LocalFilesystemStore
from rehearse.session import SessionOrchestrator, TriggerEvent
from rehearse.types import (
    ConsentState,
    OutboxEvent,
    SessionFinalizedEvent,
    VadCompleteEvent,
)


@pytest.fixture
def outbox(tmp_path: Path) -> EventOutbox:
    (tmp_path / ".outbox").mkdir()
    return EventOutbox(tmp_path / ".outbox")


def _make_session_finalized(session_id: str = "s1") -> SessionFinalizedEvent:
    return SessionFinalizedEvent(
        session_id=session_id,
        published_at=datetime.now(UTC),
        completion_status="complete",
        consent=ConsentState.GRANTED,
    )


def _make_vad_complete(
    session_id: str = "s1",
    *,
    trace_id: str | None = None,
    parent_event_id: str | None = None,
) -> VadCompleteEvent:
    kwargs: dict = dict(
        session_id=session_id,
        published_at=datetime.now(UTC),
        clip_count=10,
        accepted_count=8,
        rejected_too_short=2,
    )
    if trace_id is not None:
        kwargs["trace_id"] = trace_id
    if parent_event_id is not None:
        kwargs["parent_event_id"] = parent_event_id
    return VadCompleteEvent(**kwargs)


# ── publish ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_writes_line_to_file(outbox: EventOutbox, tmp_path: Path) -> None:
    event = _make_session_finalized()
    await outbox.publish(event)

    lines = (tmp_path / ".outbox" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["id"] == event.id
    assert data["type"] == "session_finalized"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_publish_multiple_events_appends(outbox: EventOutbox, tmp_path: Path) -> None:
    await outbox.publish(_make_session_finalized("s1"))
    await outbox.publish(_make_session_finalized("s2"))

    lines = (tmp_path / ".outbox" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2


# ── subscribe ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_receives_matching_type(outbox: EventOutbox) -> None:
    received: list[OutboxEvent] = []

    async def consumer() -> None:
        async for event in outbox.subscribe("session_finalized"):
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await outbox.publish(_make_session_finalized())
    await outbox.publish(_make_vad_complete())  # different type — should not appear
    await asyncio.sleep(0)

    await outbox.close()
    await task

    assert len(received) == 1
    assert received[0].type == "session_finalized"


@pytest.mark.asyncio
async def test_subscribe_does_not_receive_wrong_type(outbox: EventOutbox) -> None:
    received: list[OutboxEvent] = []

    async def consumer() -> None:
        async for event in outbox.subscribe("vad_complete"):
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await outbox.publish(_make_session_finalized())
    await asyncio.sleep(0)

    await outbox.close()
    await task

    assert received == []


# ── claim ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claim_sets_processing_status(outbox: EventOutbox, tmp_path: Path) -> None:
    event = _make_session_finalized()
    await outbox.publish(event)
    await outbox.claim(event.id)

    lines = (tmp_path / ".outbox" / "events.jsonl").read_text().splitlines()
    data = json.loads(lines[0])
    assert data["status"] == "processing"
    assert data["claimed_at"] is not None


# ── ack ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ack_sets_done_status(outbox: EventOutbox, tmp_path: Path) -> None:
    event = _make_session_finalized()
    await outbox.publish(event)
    await outbox.claim(event.id)
    await outbox.ack(event.id)

    lines = (tmp_path / ".outbox" / "events.jsonl").read_text().splitlines()
    data = json.loads(lines[0])
    assert data["status"] == "done"
    assert data["completed_at"] is not None


# ── fail ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_sets_failed_with_error(outbox: EventOutbox, tmp_path: Path) -> None:
    event = _make_session_finalized()
    await outbox.publish(event)
    await outbox.claim(event.id)
    await outbox.fail(event.id, "something went wrong")

    lines = (tmp_path / ".outbox" / "events.jsonl").read_text().splitlines()
    data = json.loads(lines[0])
    assert data["status"] == "failed"
    assert data["error"] == "something went wrong"
    assert data["completed_at"] is not None


# ── recover_pending ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recover_pending_reenqueues_pending_events(
    outbox: EventOutbox, tmp_path: Path
) -> None:
    e1 = _make_session_finalized("s1")
    e2 = _make_session_finalized("s2")
    outbox_file = tmp_path / ".outbox" / "events.jsonl"
    outbox_file.write_text(
        e1.model_dump_json() + "\n" + e2.model_dump_json() + "\n"
    )

    received: list[OutboxEvent] = []

    async def consumer() -> None:
        async for event in outbox.subscribe("session_finalized"):
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    count = await outbox.recover_pending()
    await asyncio.sleep(0)

    await outbox.close()
    await task

    assert count == 2
    assert len(received) == 2


@pytest.mark.asyncio
async def test_recover_skips_done_events(outbox: EventOutbox, tmp_path: Path) -> None:
    event = _make_session_finalized()
    data = json.loads(event.model_dump_json())
    data["status"] = "done"
    data["completed_at"] = datetime.now(UTC).isoformat()
    (tmp_path / ".outbox" / "events.jsonl").write_text(json.dumps(data) + "\n")

    count = await outbox.recover_pending()
    assert count == 0


@pytest.mark.asyncio
async def test_recover_reenqueues_stale_processing_events(
    outbox: EventOutbox, tmp_path: Path
) -> None:
    event = _make_session_finalized()
    data = json.loads(event.model_dump_json())
    data["status"] = "processing"
    # claimed 60 minutes ago — past the 30-minute default timeout
    data["claimed_at"] = (datetime.now(UTC) - timedelta(minutes=60)).isoformat()
    (tmp_path / ".outbox" / "events.jsonl").write_text(json.dumps(data) + "\n")

    received: list[OutboxEvent] = []

    async def consumer() -> None:
        async for event in outbox.subscribe("session_finalized"):
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    count = await outbox.recover_pending()
    await asyncio.sleep(0)

    await outbox.close()
    await task

    assert count == 1
    assert len(received) == 1


@pytest.mark.asyncio
async def test_recover_skips_recent_processing_events(
    outbox: EventOutbox, tmp_path: Path
) -> None:
    event = _make_session_finalized()
    data = json.loads(event.model_dump_json())
    data["status"] = "processing"
    # claimed only 5 minutes ago — within timeout
    data["claimed_at"] = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    (tmp_path / ".outbox" / "events.jsonl").write_text(json.dumps(data) + "\n")

    count = await outbox.recover_pending()
    assert count == 0


# ── trace propagation ─────────────────────────────────────────────────────────


def test_trace_propagation_contract() -> None:
    root = _make_session_finalized()
    child = _make_vad_complete(trace_id=root.trace_id, parent_event_id=root.id)

    assert child.trace_id == root.trace_id
    assert child.parent_event_id == root.id
    assert root.parent_event_id is None


# ── close ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_stops_subscribers(outbox: EventOutbox) -> None:
    terminated = asyncio.Event()

    async def consumer() -> None:
        async for _ in outbox.subscribe("session_finalized"):
            pass
        terminated.set()

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await outbox.close()
    await asyncio.wait_for(task, timeout=1.0)

    assert terminated.is_set()


# ── integration: finalize publishes event ─────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_publishes_session_finalized_event(tmp_path: Path) -> None:
    store = LocalFilesystemStore(root=tmp_path / "sessions", public_base_url="https://t.test")
    outbox = EventOutbox(tmp_path / ".outbox")
    orchestrator = SessionOrchestrator(store=store, outbox=outbox)

    trigger = TriggerEvent(
        from_number="+15550001234",
        body="test",
        received_at=datetime.now(UTC),
    )
    handle = await orchestrator.start(trigger)
    session_id = handle.session_id

    # Grant consent so finalize publishes the event
    await store.update_session(
        session_id,
        lambda s: _grant_consent_and_mark_in_progress(s),
    )

    await orchestrator.finalize(session_id, "complete")

    outbox_file = tmp_path / ".outbox" / "events.jsonl"
    assert outbox_file.exists(), "outbox file should be created on publish"
    lines = outbox_file.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["type"] == "session_finalized"
    assert data["session_id"] == session_id
    assert data["consent"] == "granted"
    assert data["status"] == "pending"


def _grant_consent_and_mark_in_progress(session):
    from rehearse.types import ConsentState
    session.consent = ConsentState.GRANTED
    return session
