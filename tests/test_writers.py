from __future__ import annotations

import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.frames import AudioChunk, ProsodyEvent, TranscriptDelta
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Phase, ProsodyScores, Session, Speaker
from rehearse.writers import (
    AudioRecorder,
    ProsodyWriter,
    TelemetryLogger,
    TimingWriter,
    TranscriptWriter,
)


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
async def test_audio_recorder_keeps_wav_playable_when_stream_aborts(
    writer_store: LocalFilesystemStore,
) -> None:
    session_id = writer_store._test_session_id  # type: ignore[attr-defined]
    writer = AudioRecorder(session_id, writer_store)

    async def _aborting_frames():
        yield AudioChunk(
            session_id=session_id,
            speaker=Speaker.USER,
            pcm16_16k=b"\x10\x00\x20\x00",
            ts=0.0,
        )
        yield AudioChunk(
            session_id=session_id,
            speaker=Speaker.COACH,
            pcm16_16k=b"\x30\x00\x40\x00",
            ts=0.05,
        )
        raise RuntimeError("simulated mid-call crash")

    with pytest.raises(RuntimeError, match="simulated mid-call crash"):
        await writer.run(_aborting_frames())

    wav_path = writer_store.session_dir(session_id) / "audio.wav"
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1
        assert wav_file.readframes(wav_file.getnframes()) == (
            b"\x10\x00\x20\x00\x30\x00\x40\x00"
        )


@pytest.mark.asyncio
async def test_audio_recorder_header_valid_after_each_chunk(
    writer_store: LocalFilesystemStore,
) -> None:
    session_id = writer_store._test_session_id  # type: ignore[attr-defined]
    wav_path = writer_store.session_dir(session_id) / "audio.wav"
    writer = AudioRecorder(session_id, writer_store)
    snapshots: list[bytes] = []

    async def _snapshotting_frames():
        yield AudioChunk(
            session_id=session_id,
            speaker=Speaker.USER,
            pcm16_16k=b"\xaa\x00\xbb\x00\xcc\x00\xdd\x00",
            ts=0.0,
        )
        snapshots.append(wav_path.read_bytes())
        yield AudioChunk(
            session_id=session_id,
            speaker=Speaker.COACH,
            pcm16_16k=b"\xee\x00\xff\x00",
            ts=0.05,
        )
        snapshots.append(wav_path.read_bytes())

    await writer.run(_snapshotting_frames())

    from io import BytesIO

    assert len(snapshots) == 2
    for snapshot in snapshots:
        with wave.open(BytesIO(snapshot), "rb") as wav_file:
            assert wav_file.getframerate() == 16_000
            assert wav_file.getnchannels() == 1
            wav_file.readframes(wav_file.getnframes())
    assert len(snapshots[0]) < len(snapshots[1])


@pytest.mark.asyncio
async def test_timing_writer_segments_turns_and_persists(
    writer_store: LocalFilesystemStore,
) -> None:
    session_id = writer_store._test_session_id  # type: ignore[attr-defined]

    # Drive the writer's internal clock from a controllable list of monotonic
    # readings — one tick per AudioChunk seen, in seconds.
    ticks = iter([0.0, 0.1, 0.2, 1.0, 1.5, 2.5])

    writer = TimingWriter(
        session_id,
        writer_store,
        silence_threshold_ms=600,
        clock=lambda: next(ticks),
    )

    # 100ms PCM16 chunk = 16000 samples/s * 0.1s * 2 bytes/sample = 3200 bytes.
    chunk_100ms = b"\x00\x00" * 1600

    frames = [
        # Coach turn 0: three contiguous chunks at t=0.0, 0.1, 0.2.
        AudioChunk(session_id=session_id, speaker=Speaker.COACH, pcm16_16k=chunk_100ms, ts=0.0),
        AudioChunk(session_id=session_id, speaker=Speaker.COACH, pcm16_16k=chunk_100ms, ts=0.0),
        AudioChunk(session_id=session_id, speaker=Speaker.COACH, pcm16_16k=chunk_100ms, ts=0.0),
        # User turn 0 at t=1.0 (>600ms after coach last_end at 300ms; new role anyway).
        AudioChunk(session_id=session_id, speaker=Speaker.USER, pcm16_16k=chunk_100ms, ts=0.0),
        # Coach turn 1 at t=1.5 (1200ms after coach turn-0 end of 300ms → new turn).
        AudioChunk(session_id=session_id, speaker=Speaker.COACH, pcm16_16k=chunk_100ms, ts=0.0),
        # User turn 1 at t=2.5 (1400ms after user turn-0 end of 1100ms → new turn).
        AudioChunk(session_id=session_id, speaker=Speaker.USER, pcm16_16k=chunk_100ms, ts=0.0),
    ]

    await writer.run(_iter_frames(frames))

    timing_path = writer_store.session_dir(session_id) / "timing.jsonl"
    rows = [json.loads(line) for line in timing_path.read_text().splitlines() if line]

    by_role: dict[str, list[dict]] = {"user": [], "coach": []}
    for row in rows:
        by_role[row["role"]].append(row)

    coach = by_role["coach"]
    assert [(r["turn_index"], r["event"]) for r in coach] == [
        (0, "audio_start"),
        (0, "audio_end"),
        (1, "audio_start"),
        (1, "audio_end"),
    ]
    # Coach turn 0: started at 0ms, last chunk ends at 200ms+100ms = 300ms.
    assert coach[0]["t_ms"] == 0
    assert coach[1]["t_ms"] == 300
    assert coach[1]["duration_ms"] == 300
    # Coach turn 1: started at 1500ms, ends at 1600ms.
    assert coach[2]["t_ms"] == 1500
    assert coach[3]["t_ms"] == 1600
    assert coach[3]["duration_ms"] == 100

    user = by_role["user"]
    assert [(r["turn_index"], r["event"]) for r in user] == [
        (0, "audio_start"),
        (0, "audio_end"),
        (1, "audio_start"),
        (1, "audio_end"),
    ]
    assert user[0]["t_ms"] == 1000
    assert user[1]["t_ms"] == 1100
    assert user[2]["t_ms"] == 2500
    assert user[3]["t_ms"] == 2600

    manifest = json.loads((writer_store.session_dir(session_id) / "session.json").read_text())
    assert manifest["artifact_paths"]["timing"] == "timing.jsonl"


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
