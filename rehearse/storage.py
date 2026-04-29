"""Read and write runtime session artifacts on disk.

This file owns the local filesystem store used by the live runtime. It creates
session directories, writes manifests and append-only logs, and builds public
artifact URLs served by the FastAPI app.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    """Storage methods the runtime expects from a session artifact store."""

    def session_dir(self, session_id: str) -> Path: ...
    async def write(self, session_id: str, name: str, data: bytes | str) -> None: ...
    async def append(self, session_id: str, name: str, line: str) -> None: ...
    async def read(self, session_id: str, name: str) -> bytes: ...
    def list_sessions(self) -> AsyncIterator[str]: ...
    def public_url(self, session_id: str, name: str) -> str: ...
    def viewer_url(self, session_id: str) -> str: ...


class LocalFilesystemStore:
    """Store session artifacts in `sessions/{session_id}/` on local disk."""

    def __init__(self, root: Path, public_base_url: str) -> None:
        self._root = root
        self._public_base_url = public_base_url.rstrip("/")
        self._root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def session_dir(self, session_id: str) -> Path:
        """Return the session directory path, creating it if needed."""
        path = self._root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _lock(self, key: str) -> asyncio.Lock:
        """Return a per-file async lock so concurrent writes stay ordered."""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def write(self, session_id: str, name: str, data: bytes | str) -> None:
        """Write one full artifact file and replace any previous contents."""
        path = self.session_dir(session_id) / name
        mode = "wb" if isinstance(data, bytes) else "w"
        async with self._lock(f"{session_id}/{name}"):
            await asyncio.to_thread(_write_file, path, data, mode)

    async def append(self, session_id: str, name: str, line: str) -> None:
        """Append one text line to an artifact log file."""
        path = self.session_dir(session_id) / name
        if not line.endswith("\n"):
            line = line + "\n"
        async with self._lock(f"{session_id}/{name}"):
            await asyncio.to_thread(_append_file, path, line)

    async def read(self, session_id: str, name: str) -> bytes:
        """Read one artifact file and return its raw bytes."""
        path = self.session_dir(session_id) / name
        return await asyncio.to_thread(path.read_bytes)

    async def list_sessions(self) -> AsyncIterator[str]:
        """Yield known session directory names in sorted order."""
        for entry in sorted(self._root.iterdir()):
            if entry.is_dir():
                yield entry.name

    def public_url(self, session_id: str, name: str) -> str:
        """Return the public URL that serves one artifact file."""
        return f"{self._public_base_url}/sessions/{session_id}/{name}"

    def viewer_url(self, session_id: str) -> str:
        """Return the public viewer URL for one session."""
        return f"{self._public_base_url}/viewer?session_id={session_id}"


def _write_file(path: Path, data: bytes | str, mode: str) -> None:
    """Write one file to disk using the provided open mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        f.write(data)


def _append_file(path: Path, line: str) -> None:
    """Append one line of text to a file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line)
