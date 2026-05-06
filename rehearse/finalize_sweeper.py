"""Sweep stale `in_progress` sessions and finalize them.

If Twilio drops the `/twilio/status` callback, a session manifest can sit at
`completion_status = "in_progress"` forever — synthesis never runs and no SMS
goes out. This module periodically scans the session store for manifests
older than a max-call budget and finalizes them as `partial` so the
post-call pipeline still runs.

The sweeper also handles restart recovery: an in-progress session whose
in-memory handle was lost on process restart will be picked up here on the
next sweep tick.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog

from rehearse.session import SessionOrchestrator
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Session

log = structlog.get_logger(__name__)


class FinalizeSweeper:
    """Finalize sessions whose `in_progress` status has outlived the call budget."""

    def __init__(
        self,
        orchestrator: SessionOrchestrator,
        store: LocalFilesystemStore,
        *,
        max_call_seconds: int = 360,
        grace_seconds: int = 60,
        sweep_interval_seconds: int = 60,
    ) -> None:
        """Bind the orchestrator + store and the time budgets used per sweep."""
        self._orchestrator = orchestrator
        self._store = store
        self._max_call_seconds = max_call_seconds
        self._grace_seconds = grace_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._task: asyncio.Task[None] | None = None

    @property
    def stale_threshold(self) -> timedelta:
        """Return the age at which an in-progress session is considered stale."""
        return timedelta(seconds=self._max_call_seconds + self._grace_seconds)

    def start(self) -> None:
        """Spawn the background sweep loop on the running event loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="finalize-sweeper")

    async def stop(self) -> None:
        """Cancel the sweep loop and wait for it to exit."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    async def _run(self) -> None:
        """Sweep on a fixed interval until the task is cancelled."""
        while True:
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("finalize_sweeper.error", error=str(exc))
            try:
                await asyncio.sleep(self._sweep_interval_seconds)
            except asyncio.CancelledError:
                raise

    async def sweep_once(self, *, now: datetime | None = None) -> list[str]:
        """Finalize every stale in-progress session and return their ids."""
        now = now or datetime.now(UTC)
        cutoff = now - self.stale_threshold
        finalized: list[str] = []
        async for session_id in self._store.list_sessions():
            session = await self._read_session(session_id)
            if session is None:
                continue
            if session.completion_status != "in_progress":
                continue
            if session.created_at > cutoff:
                continue
            if self._orchestrator.get(session_id) is not None:
                # Live in-process call still active; let it finalize itself.
                continue
            log.warning(
                "finalize_sweeper.stale_session",
                session_id=session_id,
                created_at=session.created_at.isoformat(),
                age_seconds=(now - session.created_at).total_seconds(),
            )
            await self._orchestrator.finalize(session_id, "partial")
            finalized.append(session_id)
        return finalized

    async def _read_session(self, session_id: str) -> Session | None:
        """Load one session manifest from disk, returning None on read errors."""
        try:
            payload = await self._store.read(session_id, "session.json")
        except FileNotFoundError:
            return None
        try:
            return Session.model_validate_json(payload)
        except Exception as exc:
            log.warning(
                "finalize_sweeper.invalid_manifest",
                session_id=session_id,
                error=str(exc),
            )
            return None
