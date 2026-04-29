"""Persist runtime frames into session artifact files.

This file turns live runtime frames into the on-disk files the rest of the
system expects, such as transcript logs, prosody logs, the session WAV, and
basic telemetry.
"""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterator, Callable

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
    """Collect audio chunks and write the full call recording to `audio.wav`."""

    def __init__(self, session_id: str, store: LocalFilesystemStore) -> None:
        """Store the session id and artifact store used for audio writes."""
        self._session_id = session_id
        self._store = store

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume audio frames and write one final WAV when the stream ends."""
        await _register_artifact(self._store, self._session_id, "audio", "audio.wav")
        chunks: list[bytes] = []
        async for frame in frames:
            if isinstance(frame, AudioChunk):
                chunks.append(frame.pcm16_16k)
        wav_bytes = _pcm16_to_wav(b"".join(chunks), sample_rate=16_000)
        await self._store.write(self._session_id, "audio.wav", wav_bytes)


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


def _pcm16_to_wav(pcm16: bytes, *, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono audio bytes in a small WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)
    return buffer.getvalue()
