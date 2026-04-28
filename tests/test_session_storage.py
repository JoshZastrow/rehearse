from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.session import SessionOrchestrator, TriggerEvent
from rehearse.storage import LocalFilesystemStore


@pytest.fixture
def store(tmp_path: Path) -> LocalFilesystemStore:
    return LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")


@pytest.mark.asyncio
async def test_store_write_append_read_list_and_public_url(store: LocalFilesystemStore) -> None:
    await store.write("s1", "note.txt", "hello")
    await store.write("s1", "blob.bin", b"\x00\x01")
    await store.append("s1", "log.jsonl", '{"a":1}')

    assert await store.read("s1", "note.txt") == b"hello"
    assert await store.read("s1", "blob.bin") == b"\x00\x01"
    assert await store.read("s1", "log.jsonl") == b'{"a":1}\n'
    assert [session async for session in store.list_sessions()] == ["s1"]
    assert store.public_url("s1", "note.txt") == "https://example.test/sessions/s1/note.txt"


@pytest.mark.asyncio
async def test_session_orchestrator_unknown_and_missing_manifest_paths(
    store: LocalFilesystemStore,
) -> None:
    orchestrator = SessionOrchestrator(store=store)
    now = datetime.now(UTC)
    handle = await orchestrator.start(
        TriggerEvent(from_number="+15551234567", body="hi", received_at=now)
    )

    assert orchestrator.get(handle.session_id) is not None
    assert orchestrator.find_by_call_sid("missing") is None

    await orchestrator.attach_call("unknown-session", "CA_missing")
    await orchestrator.finalize("unknown-session", "failed")

    await orchestrator.attach_call(handle.session_id, "CA123")
    assert orchestrator.find_by_call_sid("CA123") == handle.session_id

    session_file = store.session_dir(handle.session_id) / "session.json"
    session_file.unlink()
    await orchestrator.finalize(handle.session_id, "complete")

    assert orchestrator.get(handle.session_id) is None
