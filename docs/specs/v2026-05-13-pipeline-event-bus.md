# rehearse — Pipeline Event Bus (Outbox Pattern)

**Status**: draft  
**Owner**: jz  
**Depends on**: `rehearse/session.py` (SessionOrchestrator.finalize),
  `rehearse/storage.py`, `rehearse/types.py` (OutboxEvent, pipeline event types)  
**Relates to**: `v2026-05-13-vad-turn-segmentation.md`,
  `v2026-05-13-audio-source-separation.md`

---

## 0. One-line summary

A file-backed outbox bus that decouples session finalization from the audio
processing pipeline: producers append durable events; workers subscribe and
process them in order; no orchestrator, no scheduler, no external queue.

---

## 1. Design principles

**Choreography, not orchestration.** No component knows the full pipeline
sequence. Each worker reacts to one event type and emits the next.
`SessionOrchestrator.finalize()` emits `session_finalized`. The VAD worker
consumes it and emits `vad_complete`. The enhancement worker consumes that
and emits `enhancement_complete`. No central process directs this chain.

**Outbox for durability.** Every event is written to disk before it is
enqueued in memory. On process restart, unprocessed events are recovered
from the outbox file and re-enqueued. No event is silently lost because the
process died between finalization and processing.

**One bus, two delivery layers.**
- **Durable layer**: `{sessions_root}/.outbox/events.jsonl` — append-only,
  one JSON line per event. The ground truth. Workers that run out-of-process
  (e.g., enhancement on a separate CPU-bound process) read from this file.
- **Ephemeral layer**: `asyncio.Queue` per subscriber — in-process delivery
  for workers that share the event loop (e.g., the VAD segmenter task).

**Workers claim events atomically.** Before processing, a worker rewrites
its event's `status` from `pending` to `processing` with a `claimed_at`
timestamp. This prevents duplicate processing if multiple worker instances
run concurrently.

**Trace propagation.** Every event carries two tracing fields:
- `trace_id` — generated once by the chain root (`SessionFinalizedEvent`)
  and copied verbatim onto every downstream event. Uniquely identifies one
  pipeline execution, even if the session is reprocessed later.
- `parent_event_id` — the `id` of the specific `OutboxEvent` that triggered
  this one (`None` on the root). Together with `trace_id`, these let you
  reconstruct the full event DAG from the outbox file alone and correlate
  every structlog line across the entire pipeline run.

---

## 2. Event types

All events share a base schema (`OutboxEvent` in `rehearse/types.py`):

```json
{
  "id": "<uuid4.hex>",
  "type": "session_finalized",
  "session_id": "...",
  "published_at": "2026-05-13T12:00:00Z",
  "status": "pending",
  "claimed_at": null,
  "completed_at": null,
  "error": null,
  "payload": {},
  "trace_id": "<uuid4.hex>",
  "parent_event_id": null
}
```

`status` transitions: `pending → processing → done | failed`.

### Tracing rules

1. `SessionFinalizedEvent` generates a fresh `trace_id = _new_id()` and sets
   `parent_event_id = None`. It is the trace root.
2. Every worker that publishes a downstream event MUST set:
   - `trace_id = triggering_event.trace_id`
   - `parent_event_id = triggering_event.id`
3. Every structlog call in the pipeline MUST include `trace_id` in the bound
   context so all log lines for one pipeline run are co-queryable:
   ```python
   log = structlog.get_logger(__name__).bind(
       trace_id=event.trace_id,
       session_id=event.session_id,
   )
   ```
4. The outbox file is the trace store. Given a `trace_id`, filter the file
   to reconstruct the full run:
   ```bash
   grep '"trace_id": "abc123"' .outbox/events.jsonl | jq .
   ```
   Each event's `claimed_at - published_at` is queue wait time;
   `completed_at - claimed_at` is processing time.

### 2.1 `session_finalized`

Published by `SessionOrchestrator.finalize()` for every consented, complete
or partial session. Triggers the VAD segmentation worker.

```json
"payload": {
  "completion_status": "complete",
  "consent": "granted"
}
```

### 2.2 `vad_complete`

Published by the VAD segmentation worker after `pipeline/clips/clips.jsonl`
is written. Triggers the audio enhancement worker.

```json
"payload": {
  "clip_count": 18,
  "accepted_count": 14,
  "rejected_too_short": 4
}
```

### 2.3 `enhancement_complete`

Published by the audio enhancement worker after
`pipeline/enhanced/voice_training.jsonl` is written. Terminal event;
no downstream worker currently registered.

```json
"payload": {
  "accepted_count": 11,
  "rejected_quality": 3,
  "total_duration_s": 94.3
}
```

---

## 3. Outbox file

**Location**: `{sessions_root}/.outbox/events.jsonl`

Append-only. Each line is one JSON-serialized `OutboxEvent`. Workers
update event status by rewriting individual lines (read full file, update
matching id, rewrite — protected by a file lock).

The outbox file is NOT a session artifact. It lives at the storage root,
not inside a session directory, and is not registered in `artifact_paths`.
It is managed solely by `EventOutbox`.

```
sessions/
├── .outbox/
│   └── events.jsonl      ← one JSON line per event
├── {session_id}/
│   └── ...
```

---

## 4. Components

### 4.1 `EventOutbox` (`rehearse/pipeline/outbox.py`)

Owns the file and the in-process queue.

```python
class EventOutbox:
    async def publish(self, event: OutboxEvent) -> None:
        """Write event to file, then enqueue for in-process subscribers."""

    async def subscribe(
        self, event_type: str
    ) -> AsyncIterator[OutboxEvent]:
        """Yield events of the given type from the in-process queue."""

    async def ack(self, event_id: str) -> None:
        """Mark event done. Rewrites the matching line in the outbox file."""

    async def fail(self, event_id: str, error: str) -> None:
        """Mark event failed with error message."""

    async def recover_pending(self) -> int:
        """Re-enqueue all pending/processing events on startup. Returns count."""
```

All file operations run in `asyncio.to_thread()`. The file is protected by
an `asyncio.Lock` for rewrite operations; appends use `open(mode='a')` which
is atomic for single-line writes on POSIX.

### 4.2 `VadSegmentWorker` (`rehearse/pipeline/vad_segment.py`)

In-process asyncio task. Consumes `session_finalized` events.

```python
async def run(outbox: EventOutbox) -> None:
    async for event in outbox.subscribe("session_finalized"):
        await outbox.claim(event.id)
        try:
            result = await asyncio.to_thread(segment_session, event.session_id)
            await outbox.publish(VadCompleteEvent.from_result(event, result))
            await outbox.ack(event.id)
        except Exception as exc:
            await outbox.fail(event.id, str(exc))
```

### 4.3 `AudioEnhanceWorker` (`rehearse/pipeline/audio_enhance.py`)

Can run in two modes:

**In-process** (default, uses `ProcessPoolExecutor` for CPU isolation):
```python
async def run(outbox: EventOutbox, executor: ProcessPoolExecutor) -> None:
    async for event in outbox.subscribe("vad_complete"):
        await outbox.claim(event.id)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor, enhance_session, event.session_id
            )
            await outbox.publish(EnhancementCompleteEvent.from_result(event, result))
            await outbox.ack(event.id)
        except Exception as exc:
            await outbox.fail(event.id, str(exc))
```

**Out-of-process** (when run as a separate process on the same machine):
```bash
python -m rehearse.pipeline.audio_enhance --worker
```
This mode reads `events.jsonl` directly, claims `vad_complete` events, and
processes them. It does not use the asyncio queue — it polls the file.
This mode is intended for when enhancement runs on a separate Fly.io machine
that mounts the same persistent volume as the web process.

---

## 5. Wiring into the FastAPI app

`create_app()` in `app.py` creates the outbox and starts worker tasks in the
lifespan:

```python
outbox = EventOutbox(sessions_root / ".outbox")
executor = ProcessPoolExecutor(max_workers=1)

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await outbox.recover_pending()       # re-enqueue anything unprocessed
    sweeper.start()
    vad_task = asyncio.create_task(
        VadSegmentWorker(outbox, store).run(),
        name="vad-segment-worker",
    )
    enhance_task = asyncio.create_task(
        AudioEnhanceWorker(outbox, store, executor).run(),
        name="audio-enhance-worker",
    )
    try:
        yield
    finally:
        vad_task.cancel()
        enhance_task.cancel()
        executor.shutdown(wait=False)
        await sweeper.stop()
```

`SessionOrchestrator` receives the outbox at construction time and calls
`outbox.publish(SessionFinalizedEvent(...))` at the end of `finalize()`.

---

## 6. Observability — structured logging

All log events use `structlog` and follow the `component.action` naming
pattern already used in `session.py` and `finalize_sweeper.py`.

All log lines in the pipeline bind `trace_id` and `session_id` to the
structlog context at the start of event handling. Every field in the tables
below is present in addition to those two.

### 6.1 Outbox events

| Log key | When | Additional fields |
|---|---|---|
| `pipeline.event.published` | `EventOutbox.publish()` | `event_id`, `type`, `parent_event_id` |
| `pipeline.event.claimed` | Worker before processing | `event_id`, `type`, `worker`, `queue_wait_ms` |
| `pipeline.event.acked` | Worker after success | `event_id`, `type`, `processing_ms` |
| `pipeline.event.failed` | Worker on exception | `event_id`, `type`, `error` |
| `pipeline.event.recovered` | Startup re-enqueue | `count`, `event_types` |

`queue_wait_ms = claimed_at - published_at`. Surfaces how long events waited
before a worker picked them up — the primary signal for worker backpressure.

### 6.2 VAD segmentation

| Log key | When | Additional fields |
|---|---|---|
| `pipeline.vad.started` | Begin processing session | `event_id` |
| `pipeline.vad.clip_written` | Each accepted clip | `clip_index`, `duration_ms`, `has_transcript` |
| `pipeline.vad.clip_rejected` | Rejected clip | `clip_index`, `reason` (`too_short`) |
| `pipeline.vad.completed` | All clips written | `accepted`, `rejected`, `processing_ms` |
| `pipeline.vad.failed` | Unhandled exception | `error` |

### 6.3 Audio enhancement

| Log key | When | Additional fields |
|---|---|---|
| `pipeline.enhance.started` | Begin processing session | `event_id`, `clip_count` |
| `pipeline.enhance.clip_done` | Each clip processed | `clip_index`, `dnsmos_ovrl`, `status`, `duration_s` |
| `pipeline.enhance.completed` | All clips processed | `accepted`, `rejected_quality`, `total_duration_s`, `processing_ms` |
| `pipeline.enhance.failed` | Unhandled exception | `error` |

---

## 7. Failure handling

**Failed events** (`status = "failed"`) are not retried automatically.
They remain in the outbox file indefinitely and can be manually inspected
or re-queued via a CLI:

```bash
python -m rehearse.pipeline.outbox retry-failed --type vad_complete
```

**Claim timeout**: If a worker crashes mid-processing, the event stays
`status = "processing"` indefinitely. `recover_pending()` at startup
re-claims events that have been `processing` for more than
`CLAIM_TIMEOUT_MINUTES` (default: 30). This prevents silent stalls after
a process restart.

**Idempotency**: Both `vad_segment` and `audio_enhance` are already
idempotent by spec — re-running overwrites outputs. A double-claimed event
produces the same result.

---

## 8. Out-of-scope

- Distributed queue (Redis, SQS, RabbitMQ) — not needed until sessions_root
  moves to shared object storage.
- Dead-letter queue — failed events stay in the outbox file; manual retry
  via CLI is sufficient at current scale.
- Fan-out to multiple workers per event type — one worker per event type
  is sufficient; add a subscriber registry when needed.
- Backpressure / rate limiting — `asyncio.Queue(maxsize=N)` on the outbox
  handles this if enhancement becomes a bottleneck.

---

## 9. Sequence diagram

```
finalize()
  │
  ├─ persist_synthesis()
  │
  └─ outbox.publish(SessionFinalizedEvent)
       │
       ├─ write to .outbox/events.jsonl   [durable]
       └─ put on asyncio.Queue            [ephemeral]

VadSegmentWorker (asyncio task)
  │
  ├─ receives SessionFinalizedEvent from queue
  ├─ log: pipeline.event.claimed
  ├─ asyncio.to_thread(segment_session)
  │     ├─ log: pipeline.vad.started
  │     ├─ log: pipeline.vad.clip_written × N
  │     └─ log: pipeline.vad.completed
  ├─ outbox.publish(VadCompleteEvent)
  └─ outbox.ack(event.id)
       └─ log: pipeline.event.acked

AudioEnhanceWorker (asyncio task → ProcessPoolExecutor)
  │
  ├─ receives VadCompleteEvent from queue
  ├─ log: pipeline.event.claimed
  ├─ loop.run_in_executor(enhance_session)
  │     ├─ log: pipeline.enhance.started
  │     ├─ log: pipeline.enhance.clip_done × N
  │     └─ log: pipeline.enhance.completed
  ├─ outbox.publish(EnhancementCompleteEvent)
  └─ outbox.ack(event.id)
       └─ log: pipeline.event.acked
```
