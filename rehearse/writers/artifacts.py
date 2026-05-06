"""Persist runtime frames into session artifact files.

This file turns live runtime frames into the on-disk files the rest of the
system expects, such as transcript logs, prosody logs, the session WAV, and
basic telemetry.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import BinaryIO

from rehearse.frames import AudioChunk, Frame, ProsodyEvent, TranscriptDelta
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import (
    InferenceLogEntry,
    ModelProvider,
    Phase,
    ProsodyFrame,
    ProsodySource,
    Session,
    TranscriptFrame,
)


class TranscriptWriter:
    """Write transcript frames from the runtime bus into `transcript.jsonl`."""

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        *,
        phase_getter: Callable[[], Phase] | None = None,
    ) -> None:
        """Store the session id and artifact store used for transcript writes."""
        self._session_id = session_id
        self._store = store
        self._phase_getter = phase_getter or (lambda: Phase.INTAKE)

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume frames and append transcript rows for transcript events."""
        await _register_artifact(self._store, self._session_id, "transcript", "transcript.jsonl")
        async for frame in frames:
            if not isinstance(frame, TranscriptDelta):
                continue
            record = TranscriptFrame(
                session_id=frame.session_id,
                utterance_id=frame.utterance_id,
                ts_start=frame.ts_start,
                ts_end=frame.ts_end or frame.ts_start,
                speaker=frame.speaker,
                phase=self._phase_getter(),
                text=frame.text,
                is_interim=not frame.is_final,
            )
            await self._store.append(
                self._session_id,
                "transcript.jsonl",
                record.model_dump_json(),
            )


class ProsodyWriter:
    """Write prosody frames from the runtime bus into `prosody.jsonl`."""

    def __init__(self, session_id: str, store: LocalFilesystemStore) -> None:
        """Store the session id and artifact store used for prosody writes."""
        self._session_id = session_id
        self._store = store

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume frames and append prosody rows for prosody events."""
        await _register_artifact(self._store, self._session_id, "prosody", "prosody.jsonl")
        async for frame in frames:
            if not isinstance(frame, ProsodyEvent):
                continue
            record = ProsodyFrame(
                session_id=frame.session_id,
                utterance_id=frame.utterance_id,
                ts_start=frame.ts_start,
                ts_end=frame.ts_end,
                speaker=frame.speaker,
                source=ProsodySource.HUME_LIVE,
                scores=frame.scores,
            )
            await self._store.append(
                self._session_id,
                "prosody.jsonl",
                record.model_dump_json(),
            )


class AudioRecorder:
    """Stream audio chunks to `audio.wav` as the call runs."""

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        """Store the session id and artifact store used for audio writes."""
        self._session_id = session_id
        self._store = store
        self._sample_rate = sample_rate

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Stream audio frames into a WAV on disk, closing it when the stream ends."""
        path = self._store.session_dir(self._session_id) / "audio.wav"
        writer = await asyncio.to_thread(
            StreamingWavWriter.open, path, self._sample_rate
        )
        await self._store.update_session(
            self._session_id,
            lambda session: _add_artifact_path(session, key="audio", file_name="audio.wav"),
        )
        try:
            async for frame in frames:
                if isinstance(frame, AudioChunk):
                    await asyncio.to_thread(writer.write, frame.pcm16_16k)
        finally:
            await asyncio.to_thread(writer.close)


class TelemetryLogger:
    """Write minimal runtime telemetry rows into `telemetry.jsonl`."""

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        model: str,
        *,
        phase_getter: Callable[[], Phase] | None = None,
    ) -> None:
        """Store the session id, artifact store, and model label for telemetry."""
        self._session_id = session_id
        self._store = store
        self._model = model
        self._phase_getter = phase_getter or (lambda: Phase.INTAKE)

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume frames and append coarse telemetry for assistant responses."""
        await _register_artifact(self._store, self._session_id, "telemetry", "telemetry.jsonl")
        async for frame in frames:
            if not isinstance(frame, TranscriptDelta) or frame.speaker.value == "user":
                continue
            record = InferenceLogEntry(
                session_id=self._session_id,
                ts=utcnow(),
                phase=self._phase_getter(),
                provider=ModelProvider.HUME,
                model=self._model,
                latency_ms=0,
                stop_reason="stream_event",
            )
            await self._store.append(
                self._session_id,
                "telemetry.jsonl",
                record.model_dump_json(),
            )


async def _register_artifact(
    store: LocalFilesystemStore,
    session_id: str,
    key: str,
    file_name: str,
) -> None:
    """Add one artifact file path to the session manifest if it is missing."""
    path = store.session_dir(session_id) / file_name
    if not path.exists():
        await store.write(session_id, file_name, "")
    await store.update_session(
        session_id,
        lambda session: _add_artifact_path(session, key=key, file_name=file_name),
    )


def _add_artifact_path(session: Session, *, key: str, file_name: str) -> Session:
    """Set one artifact path on the session manifest if it is missing."""
    if session.artifact_paths.get(key) != file_name:
        session.artifact_paths[key] = file_name
    return session


class StreamingWavWriter:
    """Append PCM16 mono audio to a WAV file, keeping the header valid as it grows.

    The WAV header is rewritten after every chunk write so a process killed
    mid-call leaves a playable (truncated) file on disk.
    """

    _CHANNELS = 1
    _SAMPWIDTH = 2

    def __init__(self, path: Path, sample_rate: int, file: BinaryIO) -> None:
        """Bind a sample rate and an already-open binary file handle."""
        self._path = path
        self._sample_rate = sample_rate
        self._file = file
        self._data_bytes = 0

    @classmethod
    def open(cls, path: Path, sample_rate: int) -> StreamingWavWriter:
        """Create the file with a zero-length WAV header and return the writer."""
        path.parent.mkdir(parents=True, exist_ok=True)
        file = open(path, "wb")
        file.write(_wav_header(0, sample_rate, cls._CHANNELS, cls._SAMPWIDTH))
        file.flush()
        return cls(path, sample_rate, file)

    def write(self, pcm16: bytes) -> None:
        """Append PCM16 bytes and rewrite the header sizes so the file stays valid."""
        if not pcm16 or self._file.closed:
            return
        self._file.write(pcm16)
        self._data_bytes += len(pcm16)
        self._file.seek(4)
        self._file.write((36 + self._data_bytes).to_bytes(4, "little"))
        self._file.seek(40)
        self._file.write(self._data_bytes.to_bytes(4, "little"))
        self._file.seek(0, 2)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle if it is still open."""
        if not self._file.closed:
            self._file.close()


def _wav_header(data_size: int, sample_rate: int, channels: int, sampwidth: int) -> bytes:
    """Return a 44-byte canonical PCM WAV header for the given data length."""
    byte_rate = sample_rate * channels * sampwidth
    block_align = channels * sampwidth
    return (
        b"RIFF"
        + (36 + data_size).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + (sampwidth * 8).to_bytes(2, "little")
        + b"data"
        + data_size.to_bytes(4, "little")
    )
