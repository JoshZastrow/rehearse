"""Session artifact I/O.

The asset layer. Reads and writes session directories as defined in SPEC §5.2:

    sessions/{session_id}/
      intake.json        IntakeRecord
      story.md           markdown
      transcript.jsonl   TranscriptFrame per line
      prosody.jsonl      ProsodyFrame per line
      audio.wav          PCM audio
      feedback.md        markdown
      session.json       Session (the manifest)
      telemetry.jsonl    InferenceLogEntry per line

Every writer is append-only where possible so a mid-call crash leaves a
replayable partial session.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    def session_dir(self, session_id: str) -> Path: ...
    async def write(self, session_id: str, name: str, data: bytes | str) -> None: ...
    async def append(self, session_id: str, name: str, line: str) -> None: ...
    async def read(self, session_id: str, name: str) -> bytes: ...
    def list_sessions(self) -> AsyncIterator[str]: ...
    def public_url(self, session_id: str, name: str) -> str: ...


class LocalFilesystemStore:
    def __init__(self, root: Path, public_base_url: str) -> None:
        self._root = root
        self._public_base_url = public_base_url.rstrip("/")
        self._root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def session_dir(self, session_id: str) -> Path:
        path = self._root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def write(self, session_id: str, name: str, data: bytes | str) -> None:
        path = self.session_dir(session_id) / name
        mode = "wb" if isinstance(data, bytes) else "w"
        async with self._lock(f"{session_id}/{name}"):
            await asyncio.to_thread(_write_file, path, data, mode)

    async def append(self, session_id: str, name: str, line: str) -> None:
        path = self.session_dir(session_id) / name
        if not line.endswith("\n"):
            line = line + "\n"
        async with self._lock(f"{session_id}/{name}"):
            await asyncio.to_thread(_append_file, path, line)

    async def read(self, session_id: str, name: str) -> bytes:
        path = self.session_dir(session_id) / name
        return await asyncio.to_thread(path.read_bytes)

    async def list_sessions(self) -> AsyncIterator[str]:
        for entry in sorted(self._root.iterdir()):
            if entry.is_dir():
                yield entry.name

    def public_url(self, session_id: str, name: str) -> str:
        return f"{self._public_base_url}/sessions/{session_id}/{name}"


def _write_file(path: Path, data: bytes | str, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        f.write(data)


def _append_file(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line)
