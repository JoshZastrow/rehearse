from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rehearse.finalize_sweeper import FinalizeSweeper
from rehearse.session import SessionOrchestrator, TriggerEvent
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, Session, Speaker, TranscriptFrame


@pytest.fixture
def store(tmp_path: Path) -> LocalFilesystemStore:
    return LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")


async def _seed_session(
    store: LocalFilesystemStore,
    *,
    created_at: datetime,
    completion_status: str = "in_progress",
) -> str:
    """Write a minimal session manifest + transcript to make synthesis work."""
    session = Session(created_at=created_at)
    session_id = session.id
    payload = json.loads(session.model_dump_json())
    payload["completion_status"] = completion_status
    payload["artifact_paths"] = {"transcript": "transcript.jsonl"}
    await store.write(session_id, "session.json", json.dumps(payload, indent=2))
    frame = TranscriptFrame(
        session_id=session_id,
        utterance_id="u1",
        ts_start=0.0,
        ts_end=0.5,
        speaker=Speaker.USER,
        phase=Phase.INTAKE,
        text="hi",
    )
    await store.write(session_id, "transcript.jsonl", frame.model_dump_json() + "\n")
    return session_id


@pytest.mark.asyncio
async def test_sweep_finalizes_stale_in_progress_session(store: LocalFilesystemStore) -> None:
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    stale_id = await _seed_session(store, created_at=now - timedelta(seconds=600))

    sweeper = FinalizeSweeper(
        orchestrator, store, max_call_seconds=300, grace_seconds=60
    )
    finalized = await sweeper.sweep_once(now=now)

    assert finalized == [stale_id]
    manifest = json.loads((store.session_dir(stale_id) / "session.json").read_text())
    assert manifest["completion_status"] == "partial"
    assert manifest["artifact_paths"].get("story") == "story.md"
    assert manifest["artifact_paths"].get("feedback") == "feedback.md"


@pytest.mark.asyncio
async def test_sweep_skips_fresh_in_progress_session(store: LocalFilesystemStore) -> None:
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    fresh_id = await _seed_session(store, created_at=now - timedelta(seconds=30))

    sweeper = FinalizeSweeper(
        orchestrator, store, max_call_seconds=300, grace_seconds=60
    )
    finalized = await sweeper.sweep_once(now=now)

    assert finalized == []
    manifest = json.loads((store.session_dir(fresh_id) / "session.json").read_text())
    assert manifest["completion_status"] == "in_progress"


@pytest.mark.asyncio
async def test_sweep_skips_already_terminal_session(store: LocalFilesystemStore) -> None:
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    done_id = await _seed_session(
        store,
        created_at=now - timedelta(seconds=600),
        completion_status="complete",
    )

    sweeper = FinalizeSweeper(
        orchestrator, store, max_call_seconds=300, grace_seconds=60
    )
    finalized = await sweeper.sweep_once(now=now)

    assert finalized == []
    manifest = json.loads((store.session_dir(done_id) / "session.json").read_text())
    assert manifest["completion_status"] == "complete"


@pytest.mark.asyncio
async def test_sweep_skips_session_with_active_handle(store: LocalFilesystemStore) -> None:
    """An in-process call still owning the handle should finalize itself, not via sweep."""
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    handle = await orchestrator.start(
        TriggerEvent(
            from_number="+15551234567",
            body="hi",
            received_at=now - timedelta(seconds=600),
        )
    )

    sweeper = FinalizeSweeper(
        orchestrator, store, max_call_seconds=300, grace_seconds=60
    )
    finalized = await sweeper.sweep_once(now=now)

    assert finalized == []
    assert orchestrator.get(handle.session_id) is not None


@pytest.mark.asyncio
async def test_recover_orphans_finalizes_in_progress_without_handles(
    store: LocalFilesystemStore,
) -> None:
    """After a process restart no handles exist; every in-progress session is orphaned."""
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    fresh_id = await _seed_session(store, created_at=now - timedelta(seconds=10))
    older_id = await _seed_session(store, created_at=now - timedelta(seconds=120))
    done_id = await _seed_session(
        store,
        created_at=now - timedelta(seconds=600),
        completion_status="complete",
    )

    sweeper = FinalizeSweeper(
        orchestrator, store, max_call_seconds=300, grace_seconds=60
    )
    recovered = await sweeper.recover_orphans()

    assert sorted(recovered) == sorted([fresh_id, older_id])
    fresh_manifest = json.loads((store.session_dir(fresh_id) / "session.json").read_text())
    older_manifest = json.loads((store.session_dir(older_id) / "session.json").read_text())
    done_manifest = json.loads((store.session_dir(done_id) / "session.json").read_text())
    assert fresh_manifest["completion_status"] == "failed"
    assert older_manifest["completion_status"] == "failed"
    assert done_manifest["completion_status"] == "complete"


@pytest.mark.asyncio
async def test_recover_orphans_skips_session_with_active_handle(
    store: LocalFilesystemStore,
) -> None:
    """If an in-process call still owns a handle, recovery must not finalize it."""
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    handle = await orchestrator.start(
        TriggerEvent(
            from_number="+15551234567",
            body="hi",
            received_at=now,
        )
    )

    sweeper = FinalizeSweeper(
        orchestrator, store, max_call_seconds=300, grace_seconds=60
    )
    recovered = await sweeper.recover_orphans()

    assert recovered == []
    assert orchestrator.get(handle.session_id) is not None


@pytest.mark.asyncio
async def test_finalize_is_idempotent(store: LocalFilesystemStore) -> None:
    """A late Twilio status callback after sweeper already finalized must no-op."""
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    stale_id = await _seed_session(store, created_at=now - timedelta(seconds=600))

    await orchestrator.finalize(stale_id, "partial")
    manifest_after_first = json.loads(
        (store.session_dir(stale_id) / "session.json").read_text()
    )

    await orchestrator.finalize(stale_id, "complete")
    manifest_after_second = json.loads(
        (store.session_dir(stale_id) / "session.json").read_text()
    )

    assert manifest_after_first["completion_status"] == "partial"
    assert manifest_after_second["completion_status"] == "partial"
