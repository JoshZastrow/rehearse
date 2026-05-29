"""Integration test: annotation background task fires when a session finalizes.

Exercises the full SessionOrchestrator.finalize() → asyncio.create_task() →
annotate_session_async() path. Modal is never contacted — process_remote.spawn
is patched to return a canned result immediately.

Fixture audio is copied from a real session in sessions/ so the bytes path is
realistic, but the WAV content is irrelevant since Whisper is mocked.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rehearse.session.session import SessionOrchestrator
from rehearse.session.synthesis import SessionSynthesizer
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Session

# A real session that has audio.wav — used as a fixture asset.
_REAL_AUDIO = Path(__file__).parents[1] / "sessions" / "c4e6e8332a2e4708ae631c912977f9fd" / "audio.wav"

_FAKE_ALIGNMENTS = {
    "alignments": [
        ["Hello", [0.0, 0.4], "SPEAKER_MAIN"],
        ["world", [0.5, 0.9], "SPEAKER_MAIN"],
    ]
}


@pytest.fixture
def store(tmp_path: Path) -> LocalFilesystemStore:
    return LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")


class _NoopSynthesizer(SessionSynthesizer):
    """Synthesis stub — returns empty artifacts without calling Anthropic."""

    async def synthesize(self, store, session):
        return ("# Story\n\nstub", "# Feedback\n\nstub")


async def _drain_background_tasks() -> None:
    """Yield to the event loop until all background tasks finish."""
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.skipif(not _REAL_AUDIO.exists(), reason="real audio fixture not found")
async def test_annotation_task_fires_and_writes_audio_json(store: LocalFilesystemStore) -> None:
    session = Session(
        created_at=datetime(2026, 5, 28, tzinfo=UTC),
        completion_status="in_progress",
    )
    await store.write(session.id, "session.json", session.model_dump_json(indent=2))

    session_dir = store.session_dir(session.id)
    shutil.copy(_REAL_AUDIO, session_dir / "audio.wav")

    fake_call = MagicMock()
    fake_call.get.return_value = _FAKE_ALIGNMENTS

    orchestrator = SessionOrchestrator(store, synthesizer=_NoopSynthesizer())

    with patch("train.pipeline.annotate.process_remote") as mock_remote:
        mock_remote.spawn.return_value = fake_call
        await orchestrator.finalize(session.id, "complete")
        await _drain_background_tasks()

    audio_json = session_dir / "audio.json"
    assert audio_json.exists(), "audio.json was not written after finalize"
    written = json.loads(audio_json.read_text())
    assert written == _FAKE_ALIGNMENTS
    mock_remote.spawn.assert_called_once()
    # first positional arg is the audio bytes sent to Modal
    assert mock_remote.spawn.call_args.args[0] == _REAL_AUDIO.read_bytes()


@pytest.mark.asyncio
@pytest.mark.skipif(not _REAL_AUDIO.exists(), reason="real audio fixture not found")
async def test_annotation_skipped_when_audio_json_already_exists(
    store: LocalFilesystemStore,
) -> None:
    session = Session(
        created_at=datetime(2026, 5, 28, tzinfo=UTC),
        completion_status="in_progress",
    )
    await store.write(session.id, "session.json", session.model_dump_json(indent=2))

    session_dir = store.session_dir(session.id)
    shutil.copy(_REAL_AUDIO, session_dir / "audio.wav")
    existing = {"alignments": [["pre-existing", [0.0, 0.1], "SPEAKER_MAIN"]]}
    (session_dir / "audio.json").write_text(json.dumps(existing))

    orchestrator = SessionOrchestrator(store, synthesizer=_NoopSynthesizer())

    with patch("train.pipeline.annotate.process_remote") as mock_remote:
        await orchestrator.finalize(session.id, "complete")
        await _drain_background_tasks()
        mock_remote.spawn.assert_not_called()

    assert json.loads((session_dir / "audio.json").read_text()) == existing
