"""File-backed event outbox for the training data pipeline.

Producers append `OutboxEvent` records to a JSONL file for durability, then
deliver them to in-process subscribers via per-type asyncio queues. Workers
claim events before processing and ack or fail them after. On restart,
`recover_pending` re-enqueues any events that were not processed.

Choreography chain (no orchestrator):
    finalize() → SessionFinalizedEvent
               → [RubricScorerWorker] → rubric_scored
               → [CurationWorker]     → session_curated
               → ...
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from rehearse.types import (
    EnhancementCompleteEvent,
    OutboxEvent,
    SessionFinalizedEvent,
    VadCompleteEvent,
)

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

_SENTINEL = object()

_TYPE_MAP: dict[str, type[OutboxEvent]] = {
    "session_finalized": SessionFinalizedEvent,
    "vad_complete": VadCompleteEvent,
    "enhancement_complete": EnhancementCompleteEvent,
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _deserialize(line: str) -> OutboxEvent:
    data = json.loads(line)
    cls = _TYPE_MAP.get(data.get("type", ""), OutboxEvent)
    return cls.model_validate(data)


def _rewrite_event(lines: list[str], event_id: str, updates: dict[str, object]) -> list[str]:
    result = []
    for line in lines:
        stripped = line.rstrip("\n")
        if not stripped:
            result.append(line)
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            result.append(line)
            continue
        if data.get("id") == event_id:
            data.update(updates)
            result.append(json.dumps(data) + "\n")
        else:
            result.append(line if line.endswith("\n") else line + "\n")
    return result


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text().splitlines(keepends=True)
    except FileNotFoundError:
        return []


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines))


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line)


class EventOutbox:
    """File-backed pub/sub outbox for pipeline events.

    Two delivery layers:
    - Durable: every event is appended to `{outbox_dir}/events.jsonl` before
      in-process delivery. The file survives process restarts.
    - Ephemeral: events are put on per-type asyncio.Queues for immediate
      in-process delivery to subscribed workers.

    Workers call `claim` before processing, then `ack` or `fail` after.
    Call `recover_pending` once on startup to re-enqueue any events that
    were pending or stuck in processing when the process last exited.
    """

    def __init__(self, outbox_dir: Path, *, claim_timeout_minutes: int = 30) -> None:
        """Store the outbox path; the directory is created on first publish."""
        self._path = outbox_dir / "events.jsonl"
        self._claim_timeout = timedelta(minutes=claim_timeout_minutes)
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, list[asyncio.Queue[OutboxEvent | object]]] = {}

    async def publish(self, event: OutboxEvent) -> None:
        """Append event to the outbox file and deliver to in-process subscribers."""
        line = event.model_dump_json() + "\n"
        await asyncio.to_thread(_append_line, self._path, line)

        async with self._lock:
            queues = list(self._subscribers.get(event.type, []))
        for queue in queues:
            await queue.put(event)

        log.info(
            "pipeline.event.published",
            event_id=event.id,
            type=event.type,
            session_id=event.session_id,
            trace_id=event.trace_id,
            parent_event_id=event.parent_event_id,
        )

    async def subscribe(self, event_type: str) -> AsyncIterator[OutboxEvent]:
        """Yield events of the given type until `close()` is called."""
        queue: asyncio.Queue[OutboxEvent | object] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(event_type, []).append(queue)
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    return
                yield item  # type: ignore[misc]
        finally:
            async with self._lock:
                bucket = self._subscribers.get(event_type, [])
                if queue in bucket:
                    bucket.remove(queue)

    async def claim(self, event_id: str) -> None:
        """Mark event as processing. Call before starting work."""
        claimed_at = _utcnow()
        async with self._lock:
            lines = await asyncio.to_thread(_read_lines, self._path)
            event_data = _find_event(lines, event_id)
            updated = _rewrite_event(
                lines,
                event_id,
                {"status": "processing", "claimed_at": claimed_at.isoformat()},
            )
            await asyncio.to_thread(_write_lines, self._path, updated)

        published_at = _parse_dt(event_data.get("published_at")) if event_data else None
        queue_wait_ms = (
            int((claimed_at - published_at).total_seconds() * 1000)
            if published_at
            else None
        )
        log.info(
            "pipeline.event.claimed",
            event_id=event_id,
            type=event_data.get("type") if event_data else None,
            session_id=event_data.get("session_id") if event_data else None,
            trace_id=event_data.get("trace_id") if event_data else None,
            queue_wait_ms=queue_wait_ms,
        )

    async def ack(self, event_id: str) -> None:
        """Mark event as done. Call after successful processing."""
        completed_at = _utcnow()
        async with self._lock:
            lines = await asyncio.to_thread(_read_lines, self._path)
            event_data = _find_event(lines, event_id)
            updated = _rewrite_event(
                lines,
                event_id,
                {"status": "done", "completed_at": completed_at.isoformat()},
            )
            await asyncio.to_thread(_write_lines, self._path, updated)

        claimed_at = _parse_dt(event_data.get("claimed_at")) if event_data else None
        processing_ms = (
            int((completed_at - claimed_at).total_seconds() * 1000)
            if claimed_at
            else None
        )
        log.info(
            "pipeline.event.acked",
            event_id=event_id,
            type=event_data.get("type") if event_data else None,
            session_id=event_data.get("session_id") if event_data else None,
            trace_id=event_data.get("trace_id") if event_data else None,
            processing_ms=processing_ms,
        )

    async def fail(self, event_id: str, error: str) -> None:
        """Mark event as failed with an error message."""
        completed_at = _utcnow()
        async with self._lock:
            lines = await asyncio.to_thread(_read_lines, self._path)
            event_data = _find_event(lines, event_id)
            updated = _rewrite_event(
                lines,
                event_id,
                {
                    "status": "failed",
                    "completed_at": completed_at.isoformat(),
                    "error": error,
                },
            )
            await asyncio.to_thread(_write_lines, self._path, updated)

        log.warning(
            "pipeline.event.failed",
            event_id=event_id,
            type=event_data.get("type") if event_data else None,
            session_id=event_data.get("session_id") if event_data else None,
            trace_id=event_data.get("trace_id") if event_data else None,
            error=error,
        )

    async def recover_pending(self) -> int:
        """Re-enqueue pending or stale-processing events from the outbox file.

        Called once at app startup. Returns the number of events re-enqueued.
        """
        lines = await asyncio.to_thread(_read_lines, self._path)
        if not lines:
            return 0

        now = _utcnow()
        to_recover: list[OutboxEvent] = []

        for line in lines:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            status = data.get("status")
            if status == "pending":
                to_recover.append(_deserialize(stripped))
            elif status == "processing":
                claimed_at = _parse_dt(data.get("claimed_at"))
                if claimed_at and (now - claimed_at) > self._claim_timeout:
                    to_recover.append(_deserialize(stripped))

        for event in to_recover:
            async with self._lock:
                queues = list(self._subscribers.get(event.type, []))
            for queue in queues:
                await queue.put(event)

        if to_recover:
            log.info(
                "pipeline.event.recovered",
                count=len(to_recover),
                event_types=list({e.type for e in to_recover}),
            )

        return len(to_recover)

    async def close(self) -> None:
        """Signal all subscribers to stop iterating."""
        async with self._lock:
            all_queues = [q for bucket in self._subscribers.values() for q in bucket]
            self._subscribers.clear()
        for queue in all_queues:
            await queue.put(_SENTINEL)


def _find_event(lines: list[str], event_id: str) -> dict | None:
    for line in lines:
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if data.get("id") == event_id:
            return data
    return None


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None
