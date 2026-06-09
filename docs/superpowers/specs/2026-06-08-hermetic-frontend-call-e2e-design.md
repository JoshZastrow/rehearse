# Hermetic Full-Stack Frontend Call E2E — Design

**Date:** 2026-06-08
**Status:** Approved (brainstorming) — ready for implementation plan

## Goal (verifiable)

A single command (`npm run test:e2e:full` in `web/livekit/app`) runs one Playwright test that
passes when, and only when, all of the following hold in one run against a local self-contained
stack (no cloud, no Modal/Hume, no API keys):

1. The browser clicks **Start call** and the UI reaches **Session Active** (token fetched, room
   connected) within the test timeout.
2. At least one coach transcript entry appears in the UI (the Pocket-TTS coach turn over the
   DataChannel) — proving a real response, not just a connection.
3. The browser clicks **End call** and the UI returns to idle.
4. After the agent runner exits, `SESSION_ROOT/{session_id}/` contains: `session.json`,
   `transcript.jsonl` with both `coach` and `user` speakers, `prosody.jsonl`, `audio.wav` with
   > 0 frames, and per-role turn WAVs under `audio/coach/` and `audio/user/`.
5. The session folder is deleted before the test ends (post-delete assertion: it no longer exists).

Verification: the test is red before the implementation lands and green after, and re-running it
leaves no residual session folder under `SESSION_ROOT`.

## Problem

The frontend voice path (`useVoiceSession()` + the React UI) has no automated coverage
that exercises a real session through the browser. The existing `e2e/call.spec.ts` requires
a hand-started LiveKit server + token server and only checks UI state transitions; it never
verifies that a call produces artifacts. The Python `live_livekit` tier
(`tests/test_livekit_e2e.py::test_livekit_agent_session_via_real_room`) runs the agent
against a real `livekit-server --dev`, but its "caller" is a Python `rtc` participant — the
browser UI is never in the loop.

This is the one remaining "full product" gap: nothing proves that **clicking Start call in the
Web UI initiates a session and that artifacts land in the session folder after the call ends.**

## Goal

A hermetic, opt-in Playwright test that drives the real browser UI through a full call against
a local self-contained stack (no cloud, no Modal/Hume, no API keys) and asserts that real
artifacts are written to the session folder, then deletes the session.

Success criteria:
- Browser clicks **Start call** → UI reaches **Session Active** (token fetched, room connected).
- The scripted coach responds; the response surfaces in the UI (transcript over DataChannel).
- Browser clicks **End call** → UI returns to idle.
- `SESSION_ROOT/{session_id}/` contains `session.json`, `transcript.jsonl` (coach + user
  speakers), `prosody.jsonl`, `audio.wav` (>0 frames), and per-role turn WAVs.
- The session folder is deleted at the end.

## Non-Goals

- Not part of the fast default `test:e2e` run (heavy: needs the `livekit-server` binary and
  downloads the Pocket TTS model on first run).
- No real provider audio (no Modal/Hume). Coach audio is synthesized locally by Pocket TTS.
- No changes to production `agent.py` or `useVoiceSession()` behavior beyond what's needed to
  make the stack wireable (see Risks: session-end signal).

## Architecture

Four local processes, orchestrated by the Playwright test:

```
Playwright (chromium, fake mic)
   │  click "Start call" → fetch token → room.connect()
   ▼
[livekit-server --dev]  ws://localhost:7880        ← long-lived (Playwright webServer)
   ▲                              ▲
   │ browser joins room           │ agent joins same room
[token_server.py] :8765 ─────────┘                 ← long-lived (Playwright webServer)
   (mints JWT for fixed LIVEKIT_ROOM_NAME)
[vite dev] :3000  (serves the React app)           ← long-lived (Playwright webServer)

[scripted agent runner]                            ← one-shot, spawned by a test fixture
   reuses agent.serve_session() + a real rtc.Room, swaps the backend to
   LocalTtsCoachBackend. Writes artifacts to SESSION_ROOT = a per-test temp dir.
```

The hermetic seam already exists: `serve_session()` in `web/livekit/agent/agent.py` is
transport-agnostic, and the `live_livekit` tier proves it runs against `livekit-server --dev`.
This test adds the missing half — driving the session from the **browser UI** instead of a
Python `rtc` participant — and asserts artifacts.

## Components

### 1. `LocalTtsCoachBackend` — `tests/_fakes.py`

A `ConversationBackend` sibling of `_ScriptedCoachBackend`. Same scripted, deterministic turns
(coach `TranscriptDelta` + `ProsodyEvent`, then a user `TranscriptDelta`), but instead of canned
PCM it synthesizes the coach line with Pocket TTS:

- Load once: `TTSModel.load_model()` + `get_state_for_audio_prompt("alba")` (cached on the
  instance; load is the slow part).
- `generate_audio(voice_state, "Hello, let's begin your rehearsal.")` → 1-D PCM torch tensor at
  `tts_model.sample_rate`.
- Resample to 16 kHz, convert to PCM16 bytes, slice into 20 ms (640-byte) `AudioChunk` frames,
  publish to the bus (mirrors `_ScriptedCoachBackend`'s frame cadence so `AudioRecorder` produces
  a non-trivial `audio.wav` + per-role turn WAVs).
- All Pocket TTS calls (`load_model`, `generate_audio`) run via `asyncio.to_thread` so the event
  loop is never blocked.
- Keeps `_ScriptedCoachBackend`'s 0.2 s pre-publish sleep so the artifact writers subscribe first.

Naming: `LocalTtsCoachBackend` describes the function (local synthesis), keeping the Pocket TTS
library an implementation detail — consistent with the no-vendor-names-in-files convention.

### 2. `tests/e2e_livekit/runner.py` — test-only scripted agent runner

A trimmed `run_agent()`: builds the real `rtc.Room` + `LiveKitRoomStream` + published agent
audio track exactly like production `run_agent()`, mints an agent JWT, mints a session id, writes
the manifest to `SESSION_ROOT`, then calls
`serve_session(room, stream, LocalTtsCoachBackend(), store, session_id)`.

- Reads `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM_NAME`,
  `SESSION_ROOT` from env (same names production uses).
- **Prints the session id to stdout** (one known line, e.g. `SESSION_ID=<uuid>`) so the test can
  locate the folder without guessing.
- Exits when the browser participant disconnects (stream ends → `run_livekit_session` returns →
  `serve_session` disconnects the room → process exits). Its exit is the test's "artifacts
  flushed" signal.

Imports `livekit.rtc` lazily (only when run), matching `agent.py`'s pattern, and adds the repo
root to `sys.path` so `rehearse.*` and `tests._fakes` import.

### 3. `web/livekit/app/e2e/full-call.spec.ts` — the test

Flow:
1. A fixture spawns `runner.py` (env: shared room name + dev keys, `SESSION_ROOT` = a temp dir
   the test owns). Capture stdout; parse `SESSION_ID=`. Runner connects and waits for a
   participant.
2. `page.goto('/')`, click **Start call**, assert UI reaches **Session Active**.
3. Wait for a transcript entry (the Pocket-TTS coach turn arriving over the DataChannel) to
   confirm a real response — "asserts a response".
4. Click **End call**, assert UI returns to idle.
5. Await runner process exit, timeout-bounded.
6. Assert artifacts in `SESSION_ROOT/{SESSION_ID}/`: `session.json`, `transcript.jsonl`
   (coach + user speakers), `prosody.jsonl`, `audio.wav` (>0 frames via a WAV header check),
   per-role turn WAVs under `audio/coach/` and `audio/user/`.
7. Delete the session folder.

## Orchestration & wiring fixes

- Add a `webServer` array to `playwright.config.ts` for the three long-lived processes
  (livekit-server, token-server, vite), each with a health-check URL and `reuseExistingServer`.
- Fix the existing `baseURL`/port mismatch: Vite serves on `:3000` but `baseURL` is `:5173`.
  Standardize on `:3000`.
- The one-shot runner is **not** a webServer — it's spawned per-test by a fixture so the test
  owns its lifecycle and `SESSION_ROOT`.
- Gating: a separate `test:e2e:full` npm script and a Playwright tag/grep so this heavy test is
  opt-in. Default `test:e2e` stays fast.
- Add `pocket-tts` as a dependency (`uv add pocket-tts`) in the Python project.

## Risks / open verifications

1. **Session-end signal.** Must confirm `LiveKitRoomStream` ends (so `run_livekit_session`
   returns) when the browser participant disconnects. If it does not, the runner needs a
   `room.on("participant_disconnected")` → `stream.close()` hook. Verify first during
   implementation; this is the highest-risk unknown.
2. **Pocket TTS cost in CI.** First-run model download (~100M params, PyTorch CPU). Mitigated by
   opt-in gating; can later pre-export a voice safetensors to speed load.
3. **Sample-rate handling.** Pocket TTS sample rate ≠ 16 kHz; resampling must be correct or the
   WAVs/durations look wrong. Keep the utterance short.
4. **Single fixed room.** Workers must be 1 for this test (one room, one one-shot agent). Set in
   config for the tagged project.

## Testing the test

- Manual first pass: run the `test:e2e:full` script locally with `livekit-server` on PATH; confirm
  green and that the temp `SESSION_ROOT` is empty after the run (folder deleted).
- The existing hermetic Python tiers (`test_livekit_e2e.py`) remain the fast safety net and are
  untouched.
