from __future__ import annotations

import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.frames import AudioChunk, ProsodyEvent, TranscriptDelta
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Phase, ProsodyScores, Session, Speaker
from rehearse.writers import AudioRecorder, ProsodyWriter, TelemetryLogger, TranscriptWriter


@pytest.fixture
def writer_store(tmp_path: Path) -> LocalFilesystemStore:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    (store.session_dir(session.id) / "session.json").write_text(session.model_dump_json(indent=2))
    store._test_session_id = session.id  # type: ignore[attr-defined]
    return store


async def _run_writer(writer, frames) -> None:
    await writer.run(_iter_frames(frames))


async def _iter_frames(frames):
    for frame in frames:
        yield frame


@pytest.mark.asyncio
async def test_transcript_writer_persists_transcript_and_manifest(
    writer_store: LocalFilesystemStore,
) -> None:
    session_id = writer_store._test_session_id  # type: ignore[attr-defined]
    writer = TranscriptWriter(session_id, writer_store, phase_getter=lambda: Phase.PRACTICE)
    frame = TranscriptDelta(
        session_id=session_id,
        utterance_id="u1",
        speaker=Speaker.USER,
        text="hello",
        is_final=True,
        ts_start=0.1,
        ts_end=0.3,
    )

    await writer.run(_iter_frames([frame]))

    transcript = (writer_store.session_dir(session_id) / "transcript.jsonl").read_text().strip()
    payload = json.loads(transcript)
    assert payload["text"] == "hello"
    assert payload["phase"] == "practice"
    manifest = json.loads((writer_store.session_dir(session_id) / "session.json").read_text())
    assert manifest["artifact_paths"]["transcript"] == "transcript.jsonl"


@pytest.mark.asyncio
async def test_prosody_writer_persists_prosody_and_manifest(
    writer_store: LocalFilesystemStore,
) -> None:
    session_id = writer_store._test_session_id  # type: ignore[attr-defined]
    writer = ProsodyWriter(session_id, writer_store)
    frame = ProsodyEvent(
        session_id=session_id,
        utterance_id="u1",
        speaker=Speaker.USER,
        scores=ProsodyScores(arousal=0.4, valence=0.2, emotions={"joy": 0.8}),
        ts_start=0.1,
        ts_end=0.3,
    )

    await writer.run(_iter_frames([frame]))

    prosody = (writer_store.session_dir(session_id) / "prosody.jsonl").read_text().strip()
    payload = json.loads(prosody)
    assert payload["scores"]["emotions"]["joy"] == 0.8
    manifest = json.loads((writer_store.session_dir(session_id) / "session.json").read_text())
    assert manifest["artifact_paths"]["prosody"] == "prosody.jsonl"


@pytest.mark.asyncio
async def test_audio_recorder_writes_wav_and_manifest(writer_store: LocalFilesystemStore) -> None:
    session_id = writer_store._test_session_id  # type: ignore[attr-defined]
    writer = AudioRecorder(session_id, writer_store)
    frames = [
        AudioChunk(
            session_id=session_id,
            speaker=Speaker.USER,
            pcm16_16k=b"\x00\x00\x01\x00",
            ts=0.0,
        ),
        AudioChunk(
            session_id=session_id,
            speaker=Speaker.COACH,
            pcm16_16k=b"\x02\x00\x03\x00",
            ts=0.1,
        ),
    ]

    await writer.run(_iter_frames(frames))

    wav_path = writer_store.session_dir(session_id) / "audio.wav"
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1
        assert wav_file.readframes(wav_file.getnframes()) == b"\x00\x00\x01\x00\x02\x00\x03\x00"
    manifest = json.loads((writer_store.session_dir(session_id) / "session.json").read_text())
    assert manifest["artifact_paths"]["audio"] == "audio.wav"


@pytest.mark.asyncio
async def test_telemetry_logger_writes_assistant_events_only(
    writer_store: LocalFilesystemStore,
) -> None:
    session_id = writer_store._test_session_id  # type: ignore[attr-defined]
    writer = TelemetryLogger(
        session_id,
        writer_store,
        model="cfg-test",
        phase_getter=lambda: Phase.FEEDBACK,
    )
    frames = [
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u1",
            speaker=Speaker.USER,
            text="hello",
            is_final=True,
            ts_start=0.1,
            ts_end=0.2,
        ),
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u2",
            speaker=Speaker.COACH,
            text="hi there",
            is_final=True,
            ts_start=0.3,
            ts_end=0.4,
        ),
    ]

    await writer.run(_iter_frames(frames))

    telemetry = (
        (writer_store.session_dir(session_id) / "telemetry.jsonl")
        .read_text()
        .strip()
        .splitlines()
    )
    assert len(telemetry) == 1
    payload = json.loads(telemetry[0])
    assert payload["provider"] == "hume"
    assert payload["model"] == "cfg-test"
    assert payload["phase"] == "feedback"
    manifest = json.loads((writer_store.session_dir(session_id) / "session.json").read_text())
    assert manifest["artifact_paths"]["telemetry"] == "telemetry.jsonl"
