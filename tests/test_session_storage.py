from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.session import SessionOrchestrator, TriggerEvent
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, Speaker, TranscriptFrame


class FakeNotifier:
    """Record viewer-link SMS notifications sent during finalize."""

    def __init__(self) -> None:
        """Start with an empty list of sent messages."""
        self.messages: list[tuple[str, str]] = []

    async def send_sms(self, to: str, body: str) -> str:
        """Store the outbound SMS and return a fake message id."""
        self.messages.append((to, body))
        return "SM_test"


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


@pytest.mark.asyncio
async def test_session_orchestrator_finalize_writes_story_and_feedback(
    store: LocalFilesystemStore,
) -> None:
    notifier = FakeNotifier()
    orchestrator = SessionOrchestrator(store=store, notifier=notifier)
    now = datetime.now(UTC)
    handle = await orchestrator.start(
        TriggerEvent(from_number="+15551234567", body="help me negotiate", received_at=now)
    )
    frame = TranscriptFrame(
        session_id=handle.session_id,
        utterance_id="u1",
        ts_start=0.0,
        ts_end=0.5,
        speaker=Speaker.USER,
        phase=Phase.INTAKE,
        text="I want to ask for more compensation.",
    )
    await store.write(handle.session_id, "transcript.jsonl", frame.model_dump_json() + "\n")
    manifest = await store.read(handle.session_id, "session.json")
    session = json.loads(manifest)
    session["artifact_paths"]["transcript"] = "transcript.jsonl"
    await store.write(handle.session_id, "session.json", json.dumps(session, indent=2))

    await orchestrator.finalize(handle.session_id, "complete")

    story = (await store.read(handle.session_id, "story.md")).decode("utf-8")
    feedback = (await store.read(handle.session_id, "feedback.md")).decode("utf-8")
    manifest_after = json.loads(await store.read(handle.session_id, "session.json"))
    assert "# Story" in story
    assert "# Feedback" in feedback
    assert manifest_after["artifact_paths"]["story"] == "story.md"
    assert manifest_after["artifact_paths"]["feedback"] == "feedback.md"
    assert manifest_after["completion_status"] == "complete"
    assert notifier.messages == [
        (
            "+15551234567",
            f"Your Rehearse session is ready. View your artifacts here: "
            f"https://example.test/viewer?session_id={handle.session_id}",
        )
    ]
