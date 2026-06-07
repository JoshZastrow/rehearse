# Spec: End-to-End Test for the LiveKit / WebRTC Voice Runtime

**Status:** proposed
**Audience:** the engineer building the LiveKit WebRTC runtime (`web/livekit/`) and its tests
**Builds on:** `tests/test_call_server_e2e.py` (the Twilio call-server e2e test)

---

## 0. TL;DR

The existing e2e test proves a full call lifecycle against the **Twilio** transport
without real Twilio/GPU/Modal. Its design works because the conversation core —
`run_session()` + the `VoiceParticipant` interface + the `FrameBus` + the artifact
writers — is **transport-agnostic**. Only two files are Twilio-specific.

To test the LiveKit runtime you do **not** rewrite the core. You:

1. Build a `LiveKitCallerParticipant(VoiceParticipant)` (the WebRTC analogue of
   `TwilioCallerParticipant`) and a thin agent entrypoint that wires it into
   `run_session()` — mirroring `telephony.py:media_stream`.
2. Structure the agent so its conversation-running body is callable **in-process
   with a fake room** (the way `run_session` is callable with a fake caller today).
3. Write two test tiers: **Tier A** (hermetic, agent-level, no browser/server —
   the direct analogue of today's test) and **Tier B** (browser + `useVoiceSession()`
   via Playwright).

The hard part is the audio/event **adapter**, not the test harness.

---

## 1. How the current Twilio e2e test works (run it first)

### Run it

```bash
uv run pytest tests/test_call_server_e2e.py -v
```

It is in the default suite (`asyncio_mode = "auto"`), needs **no** credentials,
GPU, Twilio, or Modal, and finishes in ~1s.

### What it does — `test_full_call_lifecycle_on_real_server`

It boots a **real `uvicorn.Server` in-process** (not `TestClient`) and drives a
complete call, asserting all seven goal conditions:

| # | Verifies | How |
|---|----------|-----|
| 1 | Server starts | `uvicorn.Server.serve()` as a task; waits for `server.started`; hits `/healthz` |
| 2 | Call starts | `POST /twilio/voice/inbound` mints a session; parses `session_id` from the returned TwiML |
| 3 | Full-duplex audio | Real `websockets` client connects to `/media/{session_id}`, speaks the Twilio Media-Stream protocol (`connected`/`start`/`media`), drains coach audio pushed back |
| 4 | Ends with no server traceback | Sends `stop`; an `_ErrorLogCollector` on root + `uvicorn.error` asserts **zero** `ERROR`/`exc_info` records |
| 5 | Transcript written | `transcript.jsonl` non-empty, contains both `coach` and `user` rows |
| 6 | Audio saved with length | Opens `audio.wav`, `audio/coach/turn_0.wav`, `audio/user/turn_0.wav` with `wave`; asserts `getnframes() > 0` |
| 7 | Shuts down cleanly | `server.should_exit = True`; awaits the serve task; re-asserts no errors |

### The four seams that make it hermetic (you will re-use all four)

1. **Transport-agnostic core.** `run_session(session_id, caller, backend, *, store, …)`
   (`rehearse/session/conversation.py`) owns phases, consent, writers, and the bus.
   It knows nothing about Twilio. The Twilio WS handler
   (`telephony.py:media_stream`) is a ~30-line adapter that builds a
   `TwilioCallerParticipant` and calls `run_session`.
2. **Injectable backend.** The coach backend comes from `create_backend(config)`.
   The test sets `backend_type="fake"` and monkeypatches
   `telephony.create_backend` → `_ScriptedCoachBackend` (a `ConversationBackend`
   that publishes a scripted coach turn). No GPU/Modal.
3. **Injectable telephony client.** `app.TwilioRestClient` is monkeypatched to a
   fake so finalize/SMS never hit the network (`disable_sms=True` too).
4. **In-process log capture** for the no-traceback assertion (works because the
   server runs in the same process).

### One gotcha to carry forward (it will bite you again)

In `run_session`, the artifact writers run several `await` store-registration
calls **before** they subscribe to the `FrameBus`. A scripted backend that emits
its whole turn in one burst inside `start()` will publish *before* the writers
attach, and the frames are silently dropped (empty `transcript.jsonl`). The fix
in `_ScriptedCoachBackend` is a small `await asyncio.sleep(0.2)` at the top of
`start()`. Any new fake backend must do the same.

---

## 2. What is transport-agnostic vs. transport-specific

**Reusable as-is (do not touch):**
- `rehearse/session/conversation.py` — `run_session()`
- `rehearse/audio/participants.py` — `VoiceParticipant` ABC (the seam)
- `rehearse/bus.py` — `FrameBus`
- `rehearse/frames.py` — `AudioChunk`, `TranscriptDelta`, `ProsodyEvent`, `EndOfCall`, …
- `rehearse/writers/artifacts.py` — `TranscriptWriter`, `ProsodyWriter`, `AudioRecorder`, `TimingWriter`, `TelemetryLogger` (produce `transcript.jsonl`, `prosody.jsonl`, `audio.wav`, per-role turn WAVs, `timing.jsonl`, `telemetry.jsonl`)
- `rehearse/storage.py` — `LocalFilesystemStore` (same session-artifact schema)
- `create_backend()` + `ConversationBackend` — the coach backend

**Twilio-specific (the parts you replace):**
- `rehearse/audio/twilio_stream.py` — `TwilioStream` (wire codec/handshake) +
  `TwilioCallerParticipant` (the `VoiceParticipant` impl). **This file is your
  template.**
- `rehearse/api/telephony.py:media_stream` — the WS handler that wires
  `TwilioStream` → `run_session`. **The agent entrypoint is the analogue.**

### `VoiceParticipant` — the exact contract you must implement

```python
class VoiceParticipant(ABC):
    @property
    def config(self) -> ParticipantConfig: ...          # stable identity (role="caller")
    async def receive_audio(self, pcm16_16k: bytes) -> None: ...   # coach audio → out to the human
    async def say(self, request: SpeakRequest) -> None: ...        # deterministic line (usually no-op)
    def audio_stream(self, bus: FrameBus) -> AsyncIterator[bytes]: ...  # human audio in; publish AudioChunk(USER)
    async def run(self, bus: FrameBus) -> None: ...     # default: drain audio_stream
```

`TwilioCallerParticipant` shows the pattern: `audio_stream()` decodes inbound
audio, publishes `AudioChunk(speaker=Speaker.USER, pcm16_16k=…)` to the bus, and
yields the chunk; `receive_audio()` encodes coach PCM and sends it back out.

---

## 3. The LiveKit target (from the architecture diagram)

```
Browser ──┐
  React + Vite app (web/livekit/app/)   routes: / , /design/{gemini,minimal,waveform}
  useVoiceSession() hook
    ├─ livekit-client  ←── WebRTC (DTLS/SRTP/ICE)   (audio tracks)
    └─ transcript state ←── DataChannel (JSON events)
          │  WebRTC / WebSocket
   LiveKit Server (cloud or self-hosted)
          │  Worker SDK
   Python Agent (web/livekit/agent/)   entrypoint(ctx: JobContext)
```

### Concept mapping (LiveKit ⇄ existing runtime)

| LiveKit concept | Existing analogue | Notes |
|---|---|---|
| `JobContext` / room session | `telephony.media_stream` request context | Where the agent joins and runs the call |
| Subscribed **caller** audio track | `TwilioStream.inbound()` | Decode → resample to **PCM16 mono 16 kHz** → `audio_stream()` |
| Published **coach** audio track | `TwilioStream.send()` | `receive_audio()` resamples 16 kHz PCM → LiveKit frame |
| **DataChannel** JSON events | `TranscriptDelta`/`ProsodyEvent` frames | Agent publishes transcript/prosody to the browser here |
| Agent process (Worker SDK) | the uvicorn "server" in the test | The thing whose logs must show **no traceback** |
| Room token / dispatch | `POST /twilio/voice/inbound` | How a session starts |

**Audio-format warning:** WebRTC delivers Opus; the LiveKit Python `rtc` SDK
gives you PCM frames at its own sample rate (commonly 48 kHz). The runtime bus is
**PCM16 mono 16 kHz**. `LiveKitCallerParticipant` must resample both directions —
this is the LiveKit equivalent of `TwilioStream`'s μ-law + 8↔16 kHz resampling.
Get this right or `AudioRecorder` writes garbage-length WAVs.

---

## 4. Production code to build (prerequisite for the test)

> The test is only as good as the seams. Build these so the agent body is
> callable in-process with a fake room — exactly as `run_session` is callable
> with a fake caller today.

1. **`rehearse/audio/livekit_stream.py`** (mirror `twilio_stream.py`)
   - `LiveKitRoomStream` — wraps the room/track I/O: yields inbound caller PCM
     (resampled to 16 kHz), sends coach PCM as a published track, and emits
     DataChannel JSON for transcript/prosody.
   - `LiveKitCallerParticipant(VoiceParticipant)` — `config` (role `"caller"`,
     backend `"livekit"`), `audio_stream()` publishes `AudioChunk(USER)` + yields,
     `receive_audio()` → publish coach track, `say()` no-op.

2. **`web/livekit/agent/` — the agent entrypoint.** Keep `entrypoint(ctx)` thin:
   resolve the `session_id`, build the memory/LLM/backend exactly like
   `media_stream` does, then call a new **`run_livekit_session(ctx_or_room, …)`**
   that constructs `LiveKitCallerParticipant` and calls `run_session(...)`.
   **Critical:** the conversation-running body must take an abstracted room handle
   (a `Protocol`), not a concrete `JobContext`, so a fake room can be injected in
   Tier A. (This is the same refactor that extracted `run_session` out of
   `media_stream`.)

3. **DataChannel transcript bridge.** A bus subscriber (sibling of
   `TranscriptWriter`) that serializes `TranscriptDelta`/`ProsodyEvent`/`EndOfCall`
   to DataChannel JSON for the browser. Define the JSON schema once and share it
   between the agent and the `useVoiceSession()` hook.

4. **Token endpoint** (small FastAPI route or script) so the browser/test can
   join a room with a signed token.

---

## 5. The new tests

### Tier A — Agent-level e2e (hermetic; the primary test, analogue of today's)

**Goal:** prove the full lifecycle against the LiveKit runtime **without** a real
LiveKit server, browser, GPU, or Modal — just like `test_call_server_e2e.py`.

**Approach:** inject a **fake room** into `run_livekit_session`. The fake room:
- exposes a "caller" track the test feeds PCM into (silence + a few frames),
- captures the published "coach" track (assert audio came back),
- captures DataChannel JSON (assert transcript events).

Re-use `_ScriptedCoachBackend` (lift it into a shared `tests/_fakes.py`) as the
coach `ConversationBackend`. Drive a short conversation, then assert **parity**
with the Twilio test:

| Twilio test assertion | LiveKit Tier-A equivalent |
|---|---|
| server starts / `/healthz` | `run_livekit_session` starts; fake room "connected" |
| call starts (TwiML) | session minted; agent joins the (fake) room |
| full-duplex audio | fake room received coach track frames; backend received caller PCM |
| no server traceback | `_ErrorLogCollector` on the agent loggers → zero ERROR/exc_info |
| transcript written | `transcript.jsonl` has `coach` + `user` rows |
| audio saved with length | `audio.wav` + per-role turn WAVs, `getnframes() > 0` |
| clean shutdown | `run_livekit_session` returns; room closed; no errors |
| **(new)** DataChannel events | captured DataChannel JSON contains the transcript deltas |

Keep it in the default suite. Reuse the in-process log-capture and the
`sleep(0.2)` writer-subscribe fix.

### Tier A′ — Live LiveKit integration (opt-in, `live_livekit` marker)

Same flow but against a **real local `livekit-server --dev`** + the agent worker
+ a programmatic participant joined via the LiveKit Python `rtc` SDK publishing
audio. Mark `@pytest.mark.live_livekit` and add `and not live_livekit` to the
`addopts` deselection (mirror the `live_api`/`live_modal`/`pipeline` convention in
`pyproject.toml`). This catches real codec/resample/ICE issues Tier A can't.

### Tier B — Frontend e2e (`useVoiceSession()` + the web app, Playwright)

The repo already has the Playwright plugin. Launch the Vite dev server and:
- assert the three design routes render (`/design/gemini|minimal|waveform`) and
  the orb/waveform mounts;
- with a **stub agent** that emits deterministic DataChannel JSON, assert
  `useVoiceSession()` transcript state updates from those events (this is the
  hook's core contract and can be tested without real audio);
- with `live_livekit`, a full path: hook connects via `livekit-client`, publishes
  a mic track (fake/loopback media), receives the coach track, and the transcript
  populates end-to-end.

Recommended split: a **hermetic** hook test (stubbed DataChannel) in the default
frontend suite, and a **`live_livekit`** full-stack browser test gated like Tier A′.

---

## 6. Infra, fakes, and conventions

- **Local server:** `livekit-server --dev` (Docker or binary) exposes a known
  API key/secret for token signing; only needed for Tier A′ / live Tier B.
- **Markers:** add `live_livekit` to `pyproject.toml` `markers` and to `addopts`
  (`-m "not live_api and not live_modal and not pipeline and not live_livekit"`).
- **Shared fakes:** move `_ScriptedCoachBackend` to `tests/_fakes.py`; add a
  `FakeRoom`/`FakeJobContext` there.
- **No-traceback capture:** if the agent runs **in-process** (Tier A), reuse the
  `_ErrorLogCollector` on the agent's loggers. If a tier runs the agent
  **out-of-process** (real worker), capture its **stderr** and assert no
  `Traceback`.
- **Artifact parity:** the agent must write the **same session-store schema**
  (`sessions/{id}/transcript.jsonl`, `audio.wav`, per-role turn WAVs, …) via the
  existing writers, so prod and eval artifacts stay identical (this is a hard
  project rule — do not diverge the schema).

---

## 7. Task breakdown

- [ ] **T1** Extract `run_livekit_session(room: RoomProtocol, …)` from a thin
      `entrypoint(ctx)`; define the `RoomProtocol` seam.
- [ ] **T2** Implement `rehearse/audio/livekit_stream.py`
      (`LiveKitRoomStream` + `LiveKitCallerParticipant`) with bidirectional
      16 kHz resampling. Unit-test the resampler against known PCM.
- [ ] **T3** DataChannel transcript bridge + shared JSON schema.
- [ ] **T4** Token endpoint/script.
- [ ] **T5** `tests/_fakes.py` (`FakeRoom`, lifted `_ScriptedCoachBackend`).
- [ ] **T6** **Tier A** hermetic agent e2e (parity matrix in §5).
- [ ] **T7** `live_livekit` marker + `addopts`; **Tier A′** against
      `livekit-server --dev`.
- [ ] **T8** Web app `web/livekit/app/` + `useVoiceSession()`; **Tier B**
      hermetic hook test (stubbed DataChannel).
- [ ] **T9** `live_livekit` full-stack Playwright test.
- [ ] **T10** Update `pyproject.toml` markers/`addopts`; confirm default suite
      stays green and the new live tiers are deselected by default.

---

## 8. Open questions for the engineer

1. **Agent in-process vs. out-of-process for Tier A.** In-process gives you the
   log-capture no-traceback check for free; the real worker is out-of-process.
   Recommendation: make the conversation body in-process-callable (T1) and run
   Tier A in-process; reserve out-of-process for Tier A′.
2. **Who mints the session?** Twilio uses `POST /twilio/voice/inbound`. LiveKit
   could mint on agent dispatch (room name = session id?) or via the token
   endpoint. Pick one and keep `session_id` the single key across artifacts +
   DataChannel + the hook.
3. **DataChannel JSON schema** — own it explicitly; it's the contract between
   agent and `useVoiceSession()`. Version it.
4. **Resampler** — reuse `rehearse/audio/resample.py` if its rates fit, else add
   48↔16 kHz. This is the highest-risk correctness item.
5. **Consent/phases in a web context** — the runtime has a consent gate
   (`enable_consent`); decide whether the web flow runs it or pre-grants
   (`skip_consent=True`, as eval does).

---

## Appendix — key files to read

| Purpose | File |
|---|---|
| The test to mirror | `tests/test_call_server_e2e.py` |
| Transport-agnostic core | `rehearse/session/conversation.py` (`run_session`) |
| The seam to implement | `rehearse/audio/participants.py` (`VoiceParticipant`) |
| Your template adapter | `rehearse/audio/twilio_stream.py` |
| Wiring template | `rehearse/api/telephony.py` (`media_stream`) |
| Artifact writers | `rehearse/writers/artifacts.py` |
| Bus + frames | `rehearse/bus.py`, `rehearse/frames.py` |
| Marker/deselect convention | `pyproject.toml` (`live_api`/`live_modal`/`pipeline`) |
| Speaker labels | `rehearse/types.py` (`Speaker`: provider/caller alias coach/user) |
```
