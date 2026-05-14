# Conversation Backend Abstraction

**Status:** draft  
**Date:** 2026-05-14  
**Owner:** Josh Zastrow  

---

## 1. Outcomes

| # | Outcome | Verifiable by |
|---|---|---|
| O1 | A Twilio call routes to either backend with no change to the calling client | Same `/twilio/voice` and `/twilio/media` webhooks handle both |
| O2 | All session artifacts (transcript, audio, prosody, timing) are produced identically regardless of backend | Artifact schema diff = empty across backend A/B run |
| O3 | Training data collected from a pipeline backend call has the same shape as training data from a managed backend call | Schema validation on `TranscriptFrame`, `ProsodyFrame`, `AudioChunk` |
| O4 | Swapping backends requires only an environment variable change, no code deploy | `BACKEND_TYPE=managed|pipeline` toggles at process start |
| O5 | A pipeline backend with open-source models (STT + LLM + TTS) completes a full call and produces a scoreable session | Integration test passes with no managed API keys |
| O6 | The existing CLM webhook (`POST /chat/completions`) serves both backends without modification | Request/response shape unchanged across backend types |

---

## 2. Problem

The current system is built around one managed speech API that owns the conversation loop: it handles voice activity detection, turn detection, ASR, TTS, and prosody classification. It calls our server for LLM inference via a webhook. All application logic — routing, memory, persona selection, phase management, training data collection — is downstream of that API's event stream.

This creates two problems:

**Model lock-in.** We cannot prototype open-source speech models (self-hosted STT, TTS, S2S) without rewriting the call path. The audio pipeline, event protocol, and conversation loop are entangled with one provider's SDK.

**Training data dependency.** Our prosody labels, transcripts, and session artifacts are produced by inference running inside the managed API's black box. We have no path to collect equivalent data from our own inference.

The goal is a `ConversationBackend` abstraction thin enough to not add friction to the common path, but clear enough that a second backend can be built and tested against the same client.

---

## 3. What Is Not Changing

- The Twilio WebSocket bridge (`telephony.py`) — same entry point for all backends
- The CLM webhook (`POST /chat/completions`) — both backend types call this endpoint
- The `FrameBus` and frame types (`AudioChunk`, `TranscriptDelta`, `ProsodyEvent`, `EndOfCall`)
- All application logic downstream of the bus: `IntakeProcessor`, `PhaseProcessor`, `AgentRouter`, `PersonaSelectionRecorder`, `HonchoCallerMemory`, eval harness
- Session artifact writing and training data export

---

## 4. Architecture

### 4.1 The `ConversationBackend` Protocol

```python
# rehearse/backends/base.py

class PersonaSpec(TypedDict):
    """Provider-agnostic description of a conversation persona."""
    name: str                            # character name ("Alex")
    gender: Literal["male", "female"]
    system_prompt: str                   # full character instructions
    voice_ref: str | None                # backend-specific voice identifier


class ConversationBackend(Protocol):
    """Owns or delegates the audio conversation loop for one live call.

    Receives raw caller audio from the telephony layer.
    Emits typed frames to a FrameBus.
    The rest of the runtime is backend-agnostic.
    """

    async def start(self, session_id: str, bus: FrameBus) -> None:
        """Connect to backing services and begin processing."""
        ...

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        """Push one chunk of caller audio (PCM16, 16kHz, mono)."""
        ...

    async def inject_speech(self, text: str) -> None:
        """Speak a deterministic line bypassing the LLM.
        Used for consent prompts, phase bridge utterances, etc.

        Fire-and-forget: returns immediately after handing text to the
        backend's TTS queue. The caller may barge in and interrupt mid-line;
        backends must handle this gracefully (barge-in support is required).
        Callers of inject_speech() must not assume the line has been heard
        before proceeding."""
        ...

    async def swap_persona(self, persona: PersonaSpec) -> None:
        """Change the voice and character prompt at a phase transition.
        Called at intake→practice boundary by PersonaSwapCoordinator."""
        ...

    async def close(self) -> None:
        """Tear down connections and release resources."""
        ...
```

The backend does not return frames directly. It publishes to the `FrameBus` passed in `start()`. Subscribers (transcript writer, intake processor, phase processor, audio recorder) are bus-side and unchanged.

### 4.2 Frame Contract

Every backend **must** publish these frame types on the bus. This is the behavioral contract, not just the structural one:

| Frame | Trigger | Required fields |
|---|---|---|
| `TranscriptDelta(speaker=USER, is_final=False)` | User speech in progress | `text`, `utterance_id`, `ts_start` |
| `TranscriptDelta(speaker=USER, is_final=True)` | User turn complete | `text`, `utterance_id`, `ts_start`, `ts_end` |
| `TranscriptDelta(speaker=COACH, is_final=True)` | Assistant utterance complete | `text`, `utterance_id`, `ts_start`, `ts_end` |
| `AudioChunk(speaker=USER)` | Caller audio | `pcm16_16k` |
| `AudioChunk(speaker=COACH)` | Assistant audio | `pcm16_16k` |
| `ProsodyEvent(speaker=USER)` | Per user utterance | `scores` (may be zeroed — see §4.5) |
| `EndOfCall` | Call terminates | `reason` |

Backends **must not** publish frames outside this set unless they are forwarding existing frame types already handled by bus subscribers.

### 4.3 Backend Types

**`ManagedBackend`** — the external service owns the loop. The backend relays audio over a WebSocket, the external service runs VAD, turn detection, ASR, prosody, and TTS. When a user turn completes, the external service calls our CLM webhook. We respond with text; the service synthesises audio and sends it back. The current Hume integration is the first implementation.

**`PipelineBackend`** — we own the loop. The backend runs a pipeline of composable services: VAD, turn detection, STT, and one of:
- **Modular speech**: separate STT + LLM (via CLM webhook) + TTS services
- **End-to-end speech**: a single speech-to-speech model that handles all three

In the modular case the CLM webhook is still called, identically to `ManagedBackend`. In the end-to-end case the model produces both transcript tokens and audio tokens internally; the CLM webhook is bypassed.

### 4.4 Shared CLM Webhook

`POST /chat/completions` is not backend-specific. It is OpenAI SSE-compatible and called by both:

- `ManagedBackend`: the external managed service calls it after detecting a user turn
- `PipelineBackend` (modular): the pipeline calls it after STT completes and turn is confirmed

Neither backend changes the endpoint. The endpoint routes through `IntakeAwareRouter`, calls the current character agent, streams back text. All persona routing and memory logic is in the webhook handler, which is backend-agnostic.

The CLM webhook is **not called** in the end-to-end speech configuration (Moshi, Gemini Multimodal, etc.) because the model produces text internally.

### 4.5 Prosody

Every backend must emit `ProsodyEvent` with populated scores per final user utterance. Prosody cues are injected into the CLM message history by `AnthropicTransport` — the top-3 emotion scores are appended as text to each user message before the LLM call. This is how the character agent currently knows the caller's emotional state and it must be consistent across backends so the LLM receives equivalent context.

The managed backend gets scores from the external service's built-in emotion model. Pipeline backends run a local prosody classifier on each final user audio segment after turn detection confirms the turn is complete.

**Required `ProsodyService` interface for `PipelineBackend`:**

```python
class ProsodyService(Protocol):
    """Classify emotion scores from a completed user audio segment."""

    async def score(self, pcm16_16k: bytes) -> ProsodyScores:
        """Return emotion scores for the audio segment.
        Called once per final user utterance, after SmartTurn confirms end."""
        ...
```

Pluggable implementations:

| Implementation | Model | Latency | Notes |
|---|---|---|---|
| `Wav2VecProsodyService` | wav2vec2-large-superb-er | ~150ms GPU | 4-class (happy/sad/angry/neutral) |
| `SpeechBrainProsodyService` | speechbrain/emotion-recognition-wav2vec2 | ~200ms CPU | 4-class |
| `ManagedProsodyService` | External API (standalone call) | ~300ms network | Same scores as managed backend |
| `NullProsodyService` | None | 0ms | Emits zeroed scores; for local dev only |

For the first `PipelineBackend` implementation, use `Wav2VecProsodyService` on GPU or `SpeechBrainProsodyService` on CPU. `NullProsodyService` is available for local development where GPU is absent.

The `ProsodyFrame.source` field encodes the classifier used: add `ProsodySource.LOCAL_CLASSIFIER` alongside the existing `ProsodySource.MANAGED`. Session metadata records which service was active (see §9).

**Pipeline placement**: the `ProsodyService` runs in parallel with the CLM webhook call, not sequentially. After SmartTurn fires:

```
[final audio segment] ──┬── ProsodyService.score() ──► ProsodyEvent on bus
                        └── CLMWebhookService ──────► LLM call (with prosody cues from this event)
```

The prosody scores must be available before the CLM call completes (not before it starts), since `AnthropicTransport` appends them to the message when streaming begins. A short await (up to 250ms) before the first LLM token is acceptable given typical prosody classifier latency.

**End-to-end speech models**: audio is available at each turn boundary. Run `ProsodyService.score()` on the user's audio segment and emit `ProsodyEvent` before the model response. The scores are not injected into the model's context (S2S models do not have a CLM webhook call), but they are written to session artifacts for training data.

---

## 5. `ManagedBackend` Implementation

Thin wrapper over `HumeEVIClient`. No new logic — extracts existing code from `telephony.py` into the protocol.

```
rehearse/backends/managed.py
  ManagedBackend(api_key, config_id, session_id)
    start()         → open HumeEVIClient, subscribe to its events, publish frames to bus
    send_caller_audio() → HumeEVIClient.send_audio()
    inject_speech()     → HumeEVIClient.say()
    swap_persona()      → HumeEVIClient.send_session_settings(voice_id, system_prompt)
    close()         → close HumeEVIClient
```

The config concept (`config_id`) stays inside `ManagedBackend`. It is not on the `ConversationBackend` protocol. `PersonaSpec.voice_ref` maps to `config_id` inside this implementation only.

---

## 6. `PipelineBackend` Implementation

Runs a service pipeline that owns the conversation loop. Intended first implementation uses the Pipecat framework from `lib/pipecat`, but the backend protocol does not require Pipecat — any implementation that drives the same pipeline contract and emits the required frames satisfies it.

### 6.1 Modular speech configuration

```
SileroVAD  →  SmartTurnAnalyzer  →  STTService  →  ┬─ ProsodyService ─► ProsodyEvent (bus)
                                                    │
                                                    └─ CLMWebhookService  →  TTSService
                                                              │
                                                      POST /chat/completions
                                                      (prosody scores appended to message)
```

`CLMWebhookService` is a thin Pipecat processor that:
- Receives a `TranscriptionFrame` from STT
- Awaits `ProsodyEvent` from the parallel `ProsodyService` (up to 250ms)
- POSTs to `http://localhost:{port}/chat/completions` with OpenAI message format, including prosody cues in the user message the same way `AnthropicTransport` does today
- Streams the SSE response as `LLMTextFrame` downstream to TTS
- Publishes `TranscriptDelta(speaker=COACH)` to the Rehearse bus

`swap_persona()` pushes `LLMUpdateSettingsFrame` (changes system prompt injected into the next CLM call) and `TTSUpdateSettingsFrame` (changes voice_ref on the TTS service).

Pluggable services (injectable via config):

| Stage | Default (open-source) | Alternatives |
|---|---|---|
| VAD | Silero | WebRTC VAD |
| Turn detection | SmartTurn v3 (`lib/smart-turn`) | Energy-based threshold |
| STT | Whisper (local) | Deepgram, AssemblyAI |
| Prosody | `Wav2VecProsodyService` (GPU) / `SpeechBrainProsodyService` (CPU) | `NullProsodyService` for dev |
| TTS | Kokoro / Piper | Fish, Cartesia, ElevenLabs |

### 6.2 End-to-end speech configuration

```
SileroVAD  →  SmartTurnAnalyzer  →  SpeechToSpeechService
```

`SpeechToSpeechService` is a Pipecat S2S service (same interface as `OpenAIRealtimeService`, `GeminiMultimodalService`). Receives audio, emits audio + transcript tokens. No CLM webhook call.

`swap_persona()` for end-to-end models: injects a persona conditioning token or changes which model checkpoint is used. Implementation is model-specific. For Moshi (two checkpoints: male/female), persona swap at the checkpoint level requires a reconnect — acceptable latency at phase transition since a bridge utterance already covers the gap.

### 6.3 Frame translation

The Pipecat pipeline emits Pipecat frame types. A `BusPublisher` processor at the end of the pipeline translates:

| Pipecat frame | Rehearse frame |
|---|---|
| `TranscriptionFrame(text, user_id)` | `TranscriptDelta(speaker=USER, is_final=True, text)` |
| `InterimTranscriptionFrame(text)` | `TranscriptDelta(speaker=USER, is_final=False, text)` |
| `LLMFullResponseFrame(text)` | `TranscriptDelta(speaker=COACH, is_final=True, text)` |
| `AudioRawFrame(audio, speaker=bot)` | `AudioChunk(speaker=COACH, pcm16_16k)` |
| `AudioRawFrame(audio, speaker=user)` | `AudioChunk(speaker=USER, pcm16_16k)` |
| `ProsodyService.score()` result | `ProsodyEvent(speaker=USER, source=LOCAL_CLASSIFIER, scores)` |
| `EndFrame` / `ErrorFrame` | `EndOfCall(reason)` |

---

## 7. Factory

```python
# rehearse/backends/factory.py

def create_backend(config: RuntimeConfig) -> ConversationBackend:
    match config.backend_type:
        case "managed":
            from rehearse.backends.managed import ManagedBackend
            return ManagedBackend(
                api_key=config.managed_api_key,
                config_id=config.managed_config_id,
            )
        case "pipeline":
            from rehearse.backends.pipeline import PipelineBackend
            return PipelineBackend(
                speech_config=config.pipeline_speech_config,
                clm_url=f"http://localhost:{config.port}/chat/completions",
            )
        case _:
            raise ValueError(f"Unknown backend_type: {config.backend_type!r}")
```

`RuntimeConfig` additions:

```
BACKEND_TYPE=managed|pipeline          # required
PIPELINE_SPEECH_MODE=modular|e2e       # required when BACKEND_TYPE=pipeline
PIPELINE_STT_MODEL=whisper             # when modular
PIPELINE_TTS_MODEL=kokoro              # when modular
PIPELINE_E2E_MODEL=moshi               # when e2e
PIPELINE_E2E_CHECKPOINT=...            # path or HF repo
MANAGED_API_KEY=...                    # when managed (currently HUME_API_KEY)
MANAGED_CONFIG_ID=...                  # when managed (currently HUME_CONFIG_ID)
```

The existing `HUME_API_KEY` and `HUME_CONFIG_ID` env vars are aliases for `MANAGED_API_KEY` and `MANAGED_CONFIG_ID` during migration. Both are read; `MANAGED_*` takes precedence.

---

## 8. Telephony Changes

`telephony.py` currently instantiates `HumeEVIParticipant` directly. Replace with backend factory:

```python
# Before (lines ~230-234)
async with HumeEVIParticipant(
    api_key=config.hume_api_key,
    config_id=config.hume_config_id,
    session_id=session_id,
) as coach:
    ...

# After
backend = create_backend(config)
async with contextlib.AsyncExitStack() as stack:
    await stack.enter_async_context(backend)
    await backend.start(session_id, bus)
    ...
    # caller audio pump unchanged: push chunks to backend.send_caller_audio()
    # inject_speech() replaces direct coach.say() calls
    # swap_persona() replaces PersonaSwapCoordinator's direct Hume calls
```

`PersonaSwapCoordinator` currently calls `hume_client.send_session_settings()` directly. It gains a reference to `ConversationBackend` and calls `backend.swap_persona(persona_spec)` instead.

---

## 9. Training Data Collection

Training data collection is already downstream of the bus — `TranscriptWriter`, `AudioRecorder`, `ProsodyWriter` subscribe to frames and write artifacts. If all backends emit the required frame types (§4.2), training data collection requires no changes.

Session metadata should record which backend and which speech services produced the data. Add to `Session`:

```python
backend_type: Literal["managed", "pipeline"] | None = None
speech_services: dict[str, str] | None = None
# e.g. {"stt": "whisper-large-v3", "tts": "kokoro-v1", "turn": "smart-turn-v3"}
# or   {"managed": "hume-evi-v2", "config_id": "abc123"}
prosody_source: Literal["managed", "local_classifier", "none"] = "none"
```

This allows downstream eval and training pipelines to filter by data provenance without changing how data is collected.

---

## 10. Interface Interchangeability Test

### Goal

A single integration test verifies that both backends produce equivalent session outputs when given the same synthetic call. This is the falsifiable claim: "backends are interchangeable."

### Test structure

```python
# tests/integration/test_backend_interchangeability.py

@pytest.mark.parametrize("backend_type", ["managed", "pipeline"])
@pytest.mark.integration
async def test_backend_produces_complete_session(
    backend_type: str,
    synthetic_caller: SyntheticCaller,
    tmp_path: Path,
):
    """Each backend, given the same synthetic caller script, produces
    a session with the required artifact set and frame sequence."""

    config = _build_config(backend_type, session_root=tmp_path)
    backend = create_backend(config)
    bus = FrameBus(session_id="test")
    collector = FrameCollector(bus)  # records all frames in order

    await backend.start("test", bus)

    # Synthetic caller drives audio into the backend
    for chunk in synthetic_caller.audio_chunks():
        await backend.send_caller_audio(chunk)
        await asyncio.sleep(0)  # yield

    await backend.close()

    # 1. Frame sequence contains required types in order
    frame_types = [type(f).__name__ for f in collector.frames]
    assert "TranscriptDelta" in frame_types
    assert "AudioChunk" in frame_types
    assert "EndOfCall" in frame_types

    # 2. At least one final user transcript was emitted
    user_finals = [
        f for f in collector.frames
        if isinstance(f, TranscriptDelta)
        and f.speaker == Speaker.USER
        and f.is_final
    ]
    assert len(user_finals) >= 1

    # 3. At least one coach audio chunk was emitted
    coach_audio = [
        f for f in collector.frames
        if isinstance(f, AudioChunk) and f.speaker == Speaker.COACH
    ]
    assert len(coach_audio) >= 1

    # 4. Session artifacts exist on disk
    session_dir = tmp_path / "test"
    assert (session_dir / "transcript.jsonl").exists()
    assert (session_dir / "audio.wav").exists()

    # 5. ProsodyEvent was emitted (may have zeroed scores — that is allowed)
    assert any(isinstance(f, ProsodyEvent) for f in collector.frames)
```

### Managed backend test setup

The managed backend test requires a live API key. Mark `@pytest.mark.live_api`. Use the smallest possible synthetic call (consent + 2 user turns + hang up). Cap at 30 seconds.

```python
@pytest.fixture
def managed_backend_config(monkeypatch) -> RuntimeConfig:
    api_key = os.environ.get("MANAGED_API_KEY", "")
    if not api_key:
        pytest.skip("MANAGED_API_KEY not set")
    return RuntimeConfig(
        backend_type="managed",
        managed_api_key=api_key,
        managed_config_id=os.environ["MANAGED_CONFIG_ID"],
    )
```

### Pipeline backend test setup

The pipeline backend test requires no external API keys. Uses local Whisper (tiny), Kokoro TTS, SmartTurn from `lib/smart-turn`. Pre-built synthetic audio fixtures replace microphone input.

```python
@pytest.fixture
def pipeline_backend_config(tmp_path) -> RuntimeConfig:
    return RuntimeConfig(
        backend_type="pipeline",
        pipeline_speech_mode="modular",
        pipeline_stt_model="whisper-tiny",    # fast, no API key
        pipeline_tts_model="kokoro",           # open-source, local
        clm_url="http://localhost:0",          # test server below
    )

@pytest.fixture
def local_clm_server():
    """Minimal FastAPI server that returns a scripted SSE response
    for the CLM webhook. No LLM API key required."""
    ...
```

### Schema equivalence test

A separate test compares the artifact schema between two completed sessions (one per backend) and asserts the schema is identical even if values differ:

```python
@pytest.mark.integration
def test_session_artifact_schema_equivalent(managed_session_dir, pipeline_session_dir):
    managed_transcript = _load_jsonl(managed_session_dir / "transcript.jsonl")
    pipeline_transcript = _load_jsonl(pipeline_session_dir / "transcript.jsonl")

    managed_keys = set(managed_transcript[0].keys())
    pipeline_keys = set(pipeline_transcript[0].keys())
    assert managed_keys == pipeline_keys, (
        f"Schema mismatch. Managed-only: {managed_keys - pipeline_keys}. "
        f"Pipeline-only: {pipeline_keys - managed_keys}."
    )
```

---

## 11. File Inventory

| File | Change |
|---|---|
| `rehearse/backends/__init__.py` | New package |
| `rehearse/backends/base.py` | `ConversationBackend` protocol, `PersonaSpec` |
| `rehearse/backends/managed.py` | `ManagedBackend` — wraps existing Hume client |
| `rehearse/backends/pipeline.py` | `PipelineBackend` — Pipecat pipeline, modular + e2e |
| `rehearse/backends/factory.py` | `create_backend(config)` factory |
| `rehearse/backends/bus_publisher.py` | Pipecat→Rehearse frame translation processor |
| `rehearse/backends/prosody.py` | `ProsodyService` protocol + `Wav2VecProsodyService`, `SpeechBrainProsodyService`, `NullProsodyService` |
| `rehearse/config.py` | Add `backend_type`, `pipeline_*`, `managed_*` fields |
| `rehearse/telephony.py` | Replace `HumeEVIParticipant` with `create_backend()` |
| `rehearse/agents/persona_swap.py` | Call `backend.swap_persona()` instead of Hume client directly |
| `rehearse/types.py` | Add `backend_type`, `speech_services`, `prosody_source` to `Session` |
| `tests/integration/test_backend_interchangeability.py` | New — parametrized backend equivalence test |
| `tests/integration/conftest.py` | Fixtures: `local_clm_server`, `synthetic_caller`, `FrameCollector` |

---

## 12. Out of Scope

| Item | Reason |
|---|---|
| Prosody classifier model selection and benchmarking | First implementation ships with Wav2Vec2; accuracy vs latency tradeoffs deferred |
| Simultaneous A/B routing (split traffic between backends) | Operational concern; not part of this abstraction |
| Session replay across backends | Requires byte-identical audio fixtures; separate eval concern |
| Persona checkpoint swapping for end-to-end models | Model-specific; each S2S backend implements swap_persona() independently |
| Production deployment of pipeline backend | Operational; this spec covers the software abstraction only |

---

## 13. Decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| Q1 | `inject_speech()` — blocking or fire-and-forget? | **Fire-and-forget.** | Caller may barge in with "yes" mid-consent notice and proceed. Blocking would delay the happy path and add latency the managed backend doesn't have. All backends must support barge-in on injected speech. |
| Q3 | Include prosody cues in CLM message history for pipeline backend? | **Yes, required.** | The LLM uses prosody cues to read caller emotional state. Omitting them from pipeline backend calls would produce a different character agent experience and a training data distribution shift. Pipeline backends run a local prosody classifier (`ProsodyService`) in parallel with the CLM call; see §4.5. |

## 14. Open Questions

| # | Question | Impact |
|---|---|---|
| Q2 | Does `swap_persona()` need to await completion before the phase transition bridge utterance fires, or can they overlap? | Affects whether the bridge utterance ("Okay, let's run it — I'll be Alex now") is spoken by the old or new voice. Currently the managed backend handles this via its own buffering. |
