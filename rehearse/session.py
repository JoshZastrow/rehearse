"""Manage the lifecycle of one live runtime session.

This file creates session manifests, keeps lightweight in-memory handles for
active calls, and marks sessions complete, partial, or failed when the call
ends.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog

from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Session

log = structlog.get_logger(__name__)

CompletionStatus = Literal["complete", "partial", "failed", "in_progress"]


@dataclass
class SessionHandle:
    """Small in-memory record for one active call session."""

    session_id: str
    session_dir: Path
    started_at: datetime
    consent: ConsentState
    from_number_hash: str | None
    call_sid: str | None = None


@dataclass
class TriggerEvent:
    """Inbound trigger data used to start a new session."""

    from_number: str
    body: str
    received_at: datetime


class SessionOrchestrator:
    """Create, track, and finalize live runtime sessions."""

    def __init__(self, store: LocalFilesystemStore) -> None:
        self._store = store
        self._handles: dict[str, SessionHandle] = {}
        self._by_call_sid: dict[str, str] = {}

    async def start(self, trigger: TriggerEvent) -> SessionHandle:
        """Create a new session manifest and return its active handle."""
        session = Session(
            created_at=trigger.received_at,
            phone_number_hash=_hash_number(trigger.from_number),
            consent=ConsentState.PENDING,
        )
        session_dir = self._store.session_dir(session.id)
        await self._store.write(
            session.id,
            "session.json",
            session.model_dump_json(indent=2),
        )
        handle = SessionHandle(
            session_id=session.id,
            session_dir=session_dir,
            started_at=session.created_at,
            consent=session.consent,
            from_number_hash=session.phone_number_hash,
        )
        self._handles[session.id] = handle
        log.info("session.start", session_id=session.id)
        return handle

    def get(self, session_id: str) -> SessionHandle | None:
        """Return the active session handle for an id, if it exists."""
        return self._handles.get(session_id)

    @property
    def store(self) -> LocalFilesystemStore:
        """Return the artifact store used for session persistence."""
        return self._store

    async def attach_call(self, session_id: str, call_sid: str) -> None:
        """Attach a Twilio call SID to an existing active session."""
        handle = self._handles.get(session_id)
        if handle is None:
            log.warning("session.attach_call.unknown", session_id=session_id)
            return
        handle.call_sid = call_sid
        if call_sid:
            self._by_call_sid[call_sid] = session_id
        log.info("session.attach_call", session_id=session_id, call_sid=call_sid)

    def find_by_call_sid(self, call_sid: str) -> str | None:
        """Look up a session id from a Twilio call SID."""
        return self._by_call_sid.get(call_sid)

    async def finalize(self, session_id: str, status: CompletionStatus) -> None:
        """Mark a session finished and persist its final completion status."""
        handle = self._handles.pop(session_id, None)
        if handle and handle.call_sid:
            self._by_call_sid.pop(handle.call_sid, None)
        if handle is None:
            log.warning("session.finalize.unknown", session_id=session_id)
            return
        try:
            existing = await self._store.read(session_id, "session.json")
            session = Session.model_validate_json(existing)
        except FileNotFoundError:
            log.warning("session.finalize.missing_manifest", session_id=session_id)
            return
        session.completion_status = status
        await self._store.write(
            session_id,
            "session.json",
            session.model_dump_json(indent=2),
        )
        log.info("session.finalize", session_id=session_id, status=status)


def _hash_number(number: str) -> str:
    """Return a short stable hash of a phone number for storage."""
    return hashlib.sha256(number.encode("utf-8")).hexdigest()[:16]


def utcnow() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)
