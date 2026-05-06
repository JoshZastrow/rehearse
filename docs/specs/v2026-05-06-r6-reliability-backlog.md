# R6 reliability backlog

**Status:** todo
**Date:** 2026-05-06
**Owner:** Josh Zastrow
**Type:** punch list, not a design spec

A scoped backlog of the four R6 gaps left open in the runtime workstream. Each
item is independently shippable. None is currently scheduled — this doc exists
so the work isn't lost between the persona-routing push (R6 successor) and the
consent + outcome implementation (R8).

Promote any item to its own design spec (`docs/specs/v<date>-<topic>.md`)
when it gets picked up.

## Severity guide

- **P1** — call data loss or stuck sessions in the wild today.
- **P2** — recovery friction; degrades reliability under load or restart.
- **P3** — operational paper cut; safe to defer.

---

## 1. Stream WAV to disk during the call

**Severity:** P1 — long calls and crashes lose audio.

Audio is buffered in RAM by the audio writer and only flushed at end-of-call.
A 5-minute call at 16 kHz PCM16 is ~9 MB per speaker, so memory pressure isn't
the issue — durability is. If the process crashes or OOMs, the call's audio is
gone.

**What to do:** open one WAV file per speaker per call at session start, write
each `AudioChunk` frame as it arrives, and close the file in the finalize
path. Keep the existing `audio.wav` artifact path in `Session.artifact_paths`.

**Files likely touched:** `rehearse/writers/`, `rehearse/audio/`,
`rehearse/session.py` (finalize hook).

**Acceptance:** kill `-9` mid-call leaves a playable (truncated) WAV on disk.
A unit test that crashes the writer mid-stream verifies the file is still
valid PCM16.

---

## 2. Hume reconnect with backoff

**Severity:** P2 — one-shot reconnect already exists; sustained disruption
ends the call.

`HumeEVIClient.run_event_loop` retries once on socket failure
(`reconnect_backoff_s=0.1`) and then publishes `EndOfCall(reason="error")`.
That covers a single transient blip, not a 5-second WiFi hiccup.

**What to do:** retry up to N attempts with exponential backoff (e.g.
`0.1, 0.5, 2.0, 5.0`), and only emit `EndOfCall(reason="error")` after the
budget is exhausted. Bound the total reconnect window so we don't hold a dead
call open forever (e.g. 15 seconds — under the Hume 5-min cap with margin).

**Files likely touched:** `rehearse/services/hume_evi.py:86-108`.

**Acceptance:** unit test with a `connect_fn` that fails twice then succeeds
should produce one continuous event loop without an `EndOfCall(error)` frame.

---

## 3. `/twilio/status` finalize fallback

**Severity:** P1 — sessions stuck in `in_progress` if Twilio drops the
webhook.

Today's finalize path runs from the `/twilio/status` callback. If Twilio
fails to deliver it (network, credential rotation, queue lag), the session
manifest stays at `completion_status="in_progress"` indefinitely. The viewer
shows a half-rendered call.

**What to do:** add a watchdog. Two options worth weighing:

- **Per-call timer:** when the call starts, schedule a finalize at
  `start + max_duration + grace`. If the status callback already fired, the
  timer no-ops.
- **Periodic sweep:** a background task scans `sessions/` for manifests in
  `in_progress` whose start time is older than `max_duration + grace` and
  finalizes them. Survives restarts.

The sweep is more robust (handles restart + missed callback together).

**Files likely touched:** `rehearse/app.py`, `rehearse/session.py`,
new `rehearse/finalize_sweeper.py` if we go that route.

**Acceptance:** drop the `/twilio/status` callback (e.g. block the route in a
test) and confirm the session still finalizes within `max_duration + grace`.

---

## 4. Persist `SessionHandle` across restarts

**Severity:** P2 — restart loses in-flight calls. Acceptable in dev, not in
prod.

Active call state lives in process memory: the `SessionHandle` (orchestrator
+ bus + writer tasks) is held in a dict keyed by session id. A restart drops
all handles, leaving Twilio with a live media stream pointed at nothing.

**What to do:** on startup, scan `sessions/` for manifests in
`in_progress`. For each, decide: resume (reattach to the live stream — likely
not feasible, Twilio's stream is gone) or finalize-as-failed (mark
`completion_status="failed"`, run synthesis on whatever audio + transcript
landed before the crash). The second option is the realistic answer. Pairs
naturally with item 3.

**Files likely touched:** `rehearse/app.py` startup, `rehearse/session.py`.

**Acceptance:** kill the server mid-call, restart, verify the session is
marked failed and any partial artifacts are still readable in the viewer.

---

## Sequencing notes

- Items 3 and 4 share the "find stale in_progress sessions" primitive. Doing
  3 first and reusing the sweeper for 4 is cheaper than the reverse.
- Item 1 is independent; it's a small, surgical writer change.
- Item 2 is independent and the smallest of the four.
- All four are below the persona-routing push (items 1–3 in the runtime
  next-up list) and below R8 (consent + outcome) in priority. Consent is the
  legal gate to inviting non-founder users; the reliability items reduce the
  blast radius once those users start calling.
