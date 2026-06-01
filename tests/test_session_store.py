"""Tests for session audio+transcript store components."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "infra"))

from rehearse.frames import SessionStoredEvent  # noqa: E402


@pytest.mark.asyncio
async def test_session_store_writer_appends_volume_path(tmp_path):
    from rehearse.writers.artifacts import SessionStoreWriter
    from rehearse.storage import LocalFilesystemStore
    from rehearse.types import Session, ConsentState

    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    session_id = session.id
    store.session_dir(session_id)
    (tmp_path / session_id / "session.json").write_text(session.model_dump_json(indent=2))

    writer = SessionStoreWriter(session_id=session_id, store=store)

    async def _frames():
        yield SessionStoredEvent(
            session_id=session_id,
            volume_path="/mnt/sessions/sess-test",
            artifacts=["caller_stream.pcm", "provider_stream.pcm"],
            ts=1.0,
        )

    await writer.run(_frames())

    updated = json.loads((tmp_path / session_id / "session.json").read_text())
    assert updated["artifact_paths"].get("modal_volume") == "/mnt/sessions/sess-test"


@pytest.mark.asyncio
async def test_session_store_writer_ignores_other_frames(tmp_path):
    from rehearse.writers.artifacts import SessionStoreWriter
    from rehearse.frames import AudioChunk
    from rehearse.types import Speaker, Session, ConsentState
    from rehearse.storage import LocalFilesystemStore

    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    session_id = session.id
    store.session_dir(session_id)
    (tmp_path / session_id / "session.json").write_text(session.model_dump_json(indent=2))

    writer = SessionStoreWriter(session_id=session_id, store=store)

    async def _frames():
        yield AudioChunk(
            session_id=session_id,
            speaker=Speaker.USER,
            pcm16_16k=b"\x00" * 640,
            ts=0.0,
        )

    await writer.run(_frames())

    data = json.loads((tmp_path / session_id / "session.json").read_text())
    assert "modal_volume" not in data.get("artifact_paths", {})


def test_session_stored_event_fields():
    ev = SessionStoredEvent(
        session_id="sess-1",
        volume_path="/sessions/sess-1",
        artifacts=["caller_stream.pcm", "provider_stream.pcm", "tokens.jsonl", "mask.jsonl"],
        ts=1.0,
    )
    assert ev.session_id == "sess-1"
    assert ev.volume_path == "/sessions/sess-1"
    assert "mask.jsonl" in ev.artifacts


def test_write_mask_labels_padding_as_caller(tmp_path):
    from interactive import _write_mask

    token_rows = [
        '{"frame_idx": 0, "t_ms": 0.0, "text_token_id": 3, "text_piece": "", "is_padding": true}',
        '{"frame_idx": 1, "t_ms": 80.0, "text_token_id": 42, "text_piece": "▁hello", "is_padding": false}',
        '{"frame_idx": 2, "t_ms": 160.0, "text_token_id": 3, "text_piece": "", "is_padding": true}',
    ]
    path = tmp_path / "mask.jsonl"
    _write_mask(path, token_rows)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["speaker"] == "caller"    # padding → caller frame
    assert rows[1]["speaker"] == "provider"  # text token → provider frame
    assert rows[2]["speaker"] == "caller"    # padding → caller frame


def test_write_mask_preserves_frame_idx_and_t_ms(tmp_path):
    from interactive import _write_mask

    token_rows = [
        '{"frame_idx": 5, "t_ms": 400.0, "text_token_id": 7, "text_piece": "▁hi", "is_padding": false}',
    ]
    path = tmp_path / "mask.jsonl"
    _write_mask(path, token_rows)

    row = json.loads(path.read_text())
    assert row["frame_idx"] == 5
    assert row["t_ms"] == 400.0
    assert row["speaker"] == "provider"


def test_token_row_required_fields():
    row = json.dumps({
        "frame_idx": 10,
        "t_ms": 800.0,
        "text_token_id": 3,
        "text_piece": "",
        "is_padding": True,
    })
    parsed = json.loads(row)
    for field in ("frame_idx", "t_ms", "text_token_id", "text_piece", "is_padding"):
        assert field in parsed
