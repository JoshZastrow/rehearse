# rehearse — Spec: Drop Pipecat

**Status**: draft (decision + build plan)
**Owner**: jz
**Supersedes (in part)**: `docs/specs/v2026-04-27-runtime.md` §3 C3, C5, C7; `docs/specs/v2026-04-28-hume-evi-bridge.md` (subsumes the bridge decision)
**Affects**: runtime build phases R2–R7, eval harness `SimulatedTransport`

---

## 0. One-line summary

Replace Pipecat with a small, owned audio runtime: a Twilio Media Streams handler, a Hume EVI client, an asyncio frame bus, and a writer pool. The runtime is rehearse-shaped end-to-end; no framework idioms to wrap our custom code in.

## 1. Why

Discovered during R2 spike: `pipecat-ai[hume]` (1.1.0) ships only Hume TTS, not the EVI speech-to-speech service the runtime spec assumed (`HumeEVIService`). With that gap, Pipecat's value reduces to:

- the Twilio Media Streams serializer (~100 LOC of plumbing we'd otherwise write),
- a frame-typing system,
- a parallel-fanout primitive,
- optionality on services we don't currently use (VAD, alt STT, alt TTS).

The cost is a moderate framework dependency, idioms to learn, and code shaped to fit Pipecat rather than rehearse. For a pipeline that is `Twilio ↔ EVI ↔ writers`, the framework wraps custom code we'd have written anyway. We accept losing the optionality. If we later need VAD or a second voice provider, that's a focused PR — not a reason to carry a framework today.

This spec is the build plan for the replacement.

## 2. Audience

The engineer building R2. Assumes: async Python, FastAPI WebSocket, basic audio (sample rate, μ-law, PCM16), Hume EVI's WebSocket protocol (we link the docs in §6). No Pipecat knowledge required.

## 3. What Pipecat was doing for us — itemized replacement

| Pipecat capability | Used by rehearse? | Replacement |
|---|---|---|
| `TwilioFrameSerializer` (μ-law 8k decode/encode, Twilio WS protocol) | yes | `rehearse/audio/twilio_stream.py` (§5.1) |
| `Frame` types + `FrameDirection` | yes | Plain rehearse-shaped dataclasses (§5.2) |
| `FrameProcessor` lifecycle (start/end/process_frame) | yes | Plain async classes; one method per frame type |
| `Pipeline` runtime (frame routing, backpressure) | yes | `FrameBus` — one `asyncio.Queue` per subscriber (§5.3) |
| `ParallelPipeline` / `ProducerProcessor` (fanout to writers) | yes | `FrameBus.subscribe()` returns an independent queue |
| `FastAPIWebsocketTransport` | yes | FastAPI `@app.websocket(...)` directly (already in our stack) |
| Audio resampling (8k ↔ 16k) | yes | `rehearse/audio/resample.py` — linear interp via NumPy (~30 LOC) |
| Built-in services (Deepgram STT, OpenAI/Anthropic LLM, Cartesia TTS, Hume TTS, Silero VAD) | **no** (EVI integrates STT+LLM+TTS) | not needed |
| `SimulatedTransport` for eval | yes | `rehearse/audio/simulated.py` — feeds prerecorded frames into a `FrameBus` |
| Interruption / barge-in handling | partial (EVI emits the events, plumbing belongs to us) | Handled in EVI client + Twilio handler (§5.4) |

The custom code we have to write — the Hume EVI WebSocket protocol — is the *same code* either way. Pipecat doesn't reduce that line count; it shapes the wrapper.

## 4. Architecture

```
                    ┌─────────────────────────────────┐
                    │      rehearse FastAPI app       │
                    │                                 │
   Twilio  ◀─WS──▶  │  /media/{session_id}            │
                    │     │                           │
                    │     ▼                           │
                    │  TwilioStream                   │
                    │   - parses Twilio WS events     │
                    │   - μ-law 8k ↔ PCM16 16k        │
                    │     │     ▲                     │
                    │     ▼     │                     │
                    │  HumeEVIClient                  │
                    │   - hume.empathic_voice WS      │
                    │   - emits events to FrameBus    │  ─WS─▶ Hume Cloud
                    │     │                           │
                    │     ▼                           │
                    │  FrameBus  ──▶ TranscriptWriter │
                    │            ──▶ ProsodyWriter    │
                    │            ──▶ AudioRecorder    │
                    │            ──▶ TelemetryLogger  │
                    │            ──▶ PhaseProcessor   │
                    │                                 │
                    └─────────────────────────────────┘
```

**One `asyncio.TaskGroup` per call**, scoped to the `/media/{id}` WebSocket. When the WS closes (Twilio hangup), the TaskGroup cancels every subscriber. Each writer is responsible for flushing on cancel.

## 5. Components to build

### 5.1 `rehearse/audio/twilio_stream.py`

**Purpose**: read/write the Twilio Media Streams WebSocket protocol; translate audio to/from PCM16 16kHz.

```python
class TwilioStream:
    def __init__(self, ws: WebSocket): ...
    async def __aenter__(self) -> TwilioStream: ...    # waits for "connected" + "start"
    async def __aexit__(self, ...): ...

    @property
    def session_id(self) -> str: ...                    # from Twilio "start" event
    @property
    def stream_sid(self) -> str: ...

    async def inbound(self) -> AsyncIterator[bytes]:    # PCM16 16k chunks from user
    async def send(self, pcm16_16k: bytes) -> None:     # encode + frame to Twilio
    async def send_mark(self, name: str) -> None:       # for barge-in tracking
```

**Twilio Media Streams WS protocol** (the bits we care about):
- Inbound JSON events: `{"event": "connected"}`, `{"event": "start", "start": {"streamSid": "...", ...}}`, `{"event": "media", "media": {"payload": "<base64 ulaw 8k>"}}`, `{"event": "stop"}`.
- Outbound JSON: `{"event": "media", "streamSid": "...", "media": {"payload": "<base64 ulaw 8k>"}}`. Optional `mark` events to track when our audio finished playing.

**Codecs**:
- μ-law decode/encode: `audioop` (deprecated in 3.13 but still present; we pin a fallback). `audioop` is the obvious dep, with no real alternative beyond hand-rolling the lookup table (~30 LOC, well-known). Decision: ship a tiny pure-Python μ-law module, no reliance on stdlib `audioop`.
- Resampling 8k ↔ 16k: `rehearse/audio/resample.py` — NumPy linear interpolation. ~30 LOC.

**Failure modes**:
- WS receives garbage → log + drop frame, continue.
- Twilio sends `stop` → `inbound()` exits cleanly; outer TaskGroup tears down.
- WS error → bubble up; orchestrator marks session `partial`.

### 5.2 Frame types — `rehearse/frames.py`

Plain `Strict` (pydantic) classes. No subclassing of any framework's frame.

```python
class AudioChunk(Strict):
    session_id: str
    speaker: Speaker            # USER or COACH/CHARACTER
    pcm16_16k: bytes            # raw PCM16 mono 16kHz
    ts: float                   # seconds since session start

class TranscriptDelta(Strict):
    session_id: str
    utterance_id: str
    speaker: Speaker
    text: str
    is_final: bool
    ts_start: float
    ts_end: float | None = None

class ProsodyEvent(Strict):
    session_id: str
    utterance_id: str
    speaker: Speaker
    scores: ProsodyScores       # reuses types.ProsodyScores
    ts_start: float
    ts_end: float

class PhaseSignal(Strict):
    session_id: str
    from_phase: Phase | None
    to_phase: Phase
    reason: Literal["budget", "cue", "consent_decline"]
    ts: float

class EndOfCall(Strict):
    session_id: str
    reason: Literal["hangup", "error", "budget_exceeded"]
    ts: float
```

These are the *only* frame types in the runtime. R3 writers and R4 the phase processor consume them; R2 only emits `AudioChunk`, `TranscriptDelta`, `ProsodyEvent`, `EndOfCall`.

### 5.3 `rehearse/bus.py` — FrameBus

**Purpose**: one-to-many fanout. Publishers `publish(frame)`; subscribers `subscribe()` → independent `asyncio.Queue`. Closing the bus drains all queues with sentinels.

```python
class FrameBus:
    def __init__(self, session_id: str, maxsize: int = 256): ...
    def subscribe(self) -> AsyncIterator[Frame]: ...      # async-iterates a private queue
    async def publish(self, frame: Frame) -> None: ...
    async def aclose(self) -> None: ...                   # signals all subscribers to stop
```

~50 LOC. Backpressure: per-subscriber `asyncio.Queue` with `maxsize=256` (`~10s` of audio at 25 chunks/s); if a writer is slow, `publish()` blocks for that subscriber only — others continue. Slow-writer policy: log + drop oldest if full (audio is fault-tolerant, transcripts are not — we make this configurable per subscriber).

### 5.4 `rehearse/services/hume_evi.py` — HumeEVIClient

**Purpose**: open Hume EVI WebSocket, stream audio in/out, parse events, publish to `FrameBus`.

```python
class HumeEVIClient:
    def __init__(self, *, api_key: str, config_id: str, bus: FrameBus, session_id: str): ...
    async def __aenter__(self) -> HumeEVIClient: ...    # connects to EVI
    async def __aexit__(self, ...): ...

    async def send_audio(self, pcm16_16k: bytes) -> None:  # forwards to EVI
    async def run_event_loop(self) -> None:                # consume EVI events → bus

    # later (R6):
    async def swap_config(self, config_id: str, system_prompt: str | None) -> None:
```

Built on `hume.empathic_voice.AsyncHumeClient`. Hume EVI's documented event types we map to our frames:

| Hume event | Our frame |
|---|---|
| `audio_output` (base64 PCM) | `AudioChunk(speaker=COACH/CHARACTER, …)` — also forwarded to Twilio |
| `user_message` (text + prosody scores on the message) | `TranscriptDelta(speaker=USER, is_final=True)` + `ProsodyEvent` |
| `assistant_message` | `TranscriptDelta(speaker=COACH/CHARACTER, is_final=True)` |
| `user_interruption` | bus signal: cancel any in-flight outbound audio task in `TwilioStream` |
| `error` | log + close; orchestrator marks `partial` |

**Failure modes**:
- WS drop → reconnect once with ≤2s backoff (per runtime spec §3 C5). Second failure → publish `EndOfCall(reason="error")`, exit.
- EVI sends a frame format we don't recognize → log warning, continue.

**Open question (carried from runtime spec Q1)**: hot-swap of `config_id` mid-session vs. WS teardown+reopen. Spike during R2; if teardown is required, we accept the latency at the explicit phase boundary (R6).

### 5.5 `rehearse/audio/resample.py`

```python
def upsample_8k_to_16k(pcm16_8k: bytes) -> bytes: ...
def downsample_16k_to_8k(pcm16_16k: bytes) -> bytes: ...
```

Linear interpolation via NumPy. ~30 LOC. Test against a reference fixture (sine wave round-trip) to keep the math honest.

### 5.6 `rehearse/audio/mulaw.py`

Pure-Python μ-law codec — encode/decode tables. ~50 LOC. Avoids the deprecated `audioop` and any C-extension dep. Test against a reference table.

### 5.7 `rehearse/audio/simulated.py` — for eval

**Purpose**: drive the runtime in tests + eval without Twilio or Hume. Reads a script of frames and `publish()`es them on a `FrameBus` at simulated wall-clock pace.

```python
class SimulatedTransport:
    def __init__(self, bus: FrameBus, frames: Iterable[Frame], speed: float = 1.0): ...
    async def run(self) -> None: ...
```

The eval harness uses this; R2 unit tests use this; the live runtime uses `TwilioStream` instead. Same `FrameBus` interface.

### 5.8 Wiring — replace the no-op `/media/{session_id}`

```python
@app.websocket("/media/{session_id}")
async def media(ws: WebSocket, session_id: str):
    await ws.accept()
    bus = FrameBus(session_id)
    async with asyncio.TaskGroup() as tg, \
               TwilioStream(ws) as twilio, \
               HumeEVIClient(api_key=..., config_id=..., bus=bus, session_id=session_id) as hume:

        async def pump_user_audio() -> None:
            async for chunk in twilio.inbound():
                pcm16 = upsample_8k_to_16k(chunk)
                await hume.send_audio(pcm16)
                await bus.publish(AudioChunk(session_id, USER, pcm16, ts=...))

        async def pump_assistant_audio() -> None:
            async for frame in bus.subscribe():
                if isinstance(frame, AudioChunk) and frame.speaker != USER:
                    await twilio.send(downsample_16k_to_8k(frame.pcm16_16k))

        tg.create_task(pump_user_audio())
        tg.create_task(pump_assistant_audio())
        tg.create_task(hume.run_event_loop())
        # R3: tg.create_task(transcript_writer(bus.subscribe(), store))
        # R3: tg.create_task(prosody_writer(bus.subscribe(), store))
        # R3: tg.create_task(audio_recorder(bus.subscribe(), store))
        # R4: tg.create_task(phase_processor(bus, ...))

    await orchestrator.finalize(session_id, "complete")
```

**TwiML change in `/twilio/voice/inbound`**: replace the `<Say>` block with `<Connect><Stream url="wss://{base}/media/{session_id}"/></Connect>`. The session is minted before TwiML is returned (already happens in R1 inbound handler) and the `session_id` is encoded in the URL path.

## 6. Files added / removed

**Added**:
```
rehearse/
├── frames.py
├── bus.py
├── audio/
│   ├── __init__.py
│   ├── twilio_stream.py
│   ├── mulaw.py
│   ├── resample.py
│   └── simulated.py
└── services/
    ├── __init__.py
    └── hume_evi.py
```

**Removed**:
- `rehearse/pipeline.py` (Pipecat-shaped) → still exists as a name but reshaped to `build_runtime(...)` returning the wired `FrameBus` + tasks; tiny.

**`pyproject.toml`**: drop `pipecat-ai[anthropic,hume,twilio,silero]>=0.0.50`. Add `numpy>=1.26`. `hume>=0.13` already pulled transitively; declare it explicitly.

## 7. Sizing

| Area | LOC (est.) | Test LOC |
|---|---|---|
| `frames.py` | 60 | 30 |
| `bus.py` | 50 | 80 |
| `audio/mulaw.py` | 50 | 50 |
| `audio/resample.py` | 30 | 40 |
| `audio/twilio_stream.py` | 150 | 120 |
| `audio/simulated.py` | 50 | 30 |
| `services/hume_evi.py` | 200 | 150 |
| `/media/{id}` wiring + TwiML change | 50 | 50 (integration) |
| **R2 total** | **~640** | **~550** |

R3 (writers, phase processor) layers on top with no churn to the bridge — that's a separate phase.

## 8. What we lose, explicit

1. **Free Silero VAD.** If user-speaking-detection becomes a need, ~50 LOC via the `silero-vad` package. Acceptable.
2. **Free Deepgram/AssemblyAI fallback.** If EVI's STT becomes unreliable and we want a fallback transcription pass, ~100 LOC integration. Acceptable.
3. **Free alt TTS providers.** If we ever want a non-Hume voice, ~100 LOC. Acceptable.
4. **Pipecat's barge-in plumbing.** EVI emits user-interruption events; we handle the audio cancellation in `TwilioStream` ourselves (~30 LOC). Acceptable.
5. **Pipecat ecosystem reference examples.** Replaced by smaller, self-contained code. Net positive for onboarding.

If multiple of these become needs in the same quarter, the calculus flips and we reconsider Pipecat. Realistic horizon: not before live deployment to non-founder users.

## 9. Migration plan (R2 PR shape)

1. Land the audio primitives (`mulaw.py`, `resample.py`) with reference-fixture tests. No app changes yet.
2. Land `frames.py` + `bus.py` with unit tests. No app changes.
3. Land `services/hume_evi.py` with a fake Hume WS server in tests. No app changes.
4. Land `audio/twilio_stream.py` with a recorded Twilio frame fixture in tests. No app changes.
5. Land `audio/simulated.py`.
6. Wire `/media/{session_id}` to use them; change `/twilio/voice/inbound` TwiML to `<Stream>`. Manual demo at this point: dial → talk to Hume's default LLM → hear back. R2 done.
7. Remove `pipecat-ai` from `pyproject.toml`. Run `uv lock` + `uv sync`. Tests must still pass.

Each of 1–6 is a small standalone PR. Step 7 is the cleanup commit.

## 10. Affected specs to update

After this lands:
- `docs/specs/v2026-04-27-runtime.md` §3 C3, C5, C7 — replace Pipecat references with `FrameBus` + components from §5 above.
- `docs/specs/v2026-04-27-eval-harness.md` — `SimulatedTransport` references reshape to use `FrameBus`. (Not yet read by me; the engineer applies the same renaming.)
- `docs/specs/v2026-04-28-hume-evi-bridge.md` — superseded by this spec; mark as resolved with pointer here.

## 11. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-28 | Drop Pipecat for R2; build owned bridge per this spec | Pipecat's value reduced to Twilio serializer + frame primitives once `HumeEVIService` was confirmed missing; rest of framework wraps custom code. ~640 LOC fully owned vs. ~500 LOC inside a framework — sizing comparable, design freedom higher. Optionality on alt providers explicitly deferred. |
