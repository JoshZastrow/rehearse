# rehearse — Spec: Hume EVI ↔ Pipecat bridge (decision)

**Status**: draft (decision needed)
**Owner**: jz
**Depends on**: `docs/specs/v2026-04-27-runtime.md` (Phase R2)
**Blocks**: Phase R2 implementation

---

## 0. One-line summary

The runtime spec assumes `HumeEVIService` ships in `pipecat-ai[hume]`. It doesn't. We need to choose how to bridge Hume EVI (speech-to-speech + prosody) into the Pipecat pipeline before R2 can land.

## 1. The gap

Runtime spec §3 C5 states:

> **C5. HumeEVIService (Pipecat service, configured by us)**
> **Purpose**: speech-to-speech voice with prosody as first-class events. We do **not** write this — it ships in `pipecat-ai[hume]`.

Verified against the installed environment:

| Package | Version | What it ships |
|---|---|---|
| `pipecat-ai[hume]` | 1.1.0 | `pipecat.services.hume.tts` only — TTS service. **No EVI service**, no STT, no prosody event types. |
| `hume` (Hume's official SDK) | 0.13.11 | `hume.empathic_voice` — async WebSocket client for EVI. No Pipecat integration. |

There is no off-the-shelf code path from Hume EVI's WebSocket protocol to Pipecat's frame system. The runtime spec's "trivial pass-through" wiring assumed a class that doesn't exist.

## 2. Why this matters

Three things in the rehearse design lean on EVI specifically (not a generic STT→LLM→TTS chain):

1. **Live prosody as a first-class event stream.** EVI emits per-utterance prosody scores (~48-dimension emotion vector + arousal/valence) alongside transcription, in real time during the call. Generic STT services don't. The runtime persists these to `prosody.jsonl`; the eval harness scores prosody-citation accuracy; training corpora use prosody as a signal. Losing live prosody means a separate offline pass over recorded audio — a different system shape.
2. **Speech-to-speech latency.** EVI is a single integrated service (STT + LLM-or-CLM + TTS in one WS session). The 800ms p50 latency budget (runtime spec §6.2) is achievable because of that integration. Decomposing into STT + LLM + TTS adds round-trips.
3. **The CLM webhook (R4 dependency).** Hume EVI calls our `/hume/clm/{session_id}` webhook on every turn so Claude drives the dialogue. This is the spec's primary swap point for character/coach intelligence. It's an EVI-specific feature; STT/LLM/TTS pipelines have no equivalent.

A change here that drops EVI cascades into R3 (prosody writer has nothing to write), R4 (no CLM webhook target), and the eval harness (prosody scoring becomes batch instead of live).

## 3. Options

### Option 1 — Build a custom `HumeEVIService` FrameProcessor

Write the bridge ourselves. A Pipecat `FrameProcessor` that wraps `hume.empathic_voice.AsyncHumeClient`:

- **Inbound**: consumes `AudioRawFrame` from the Twilio transport, streams to Hume EVI's WebSocket as audio chunks.
- **Outbound**: receives EVI events (audio output, transcription, prosody) and emits as Pipecat frames — `AudioRawFrame`, `TranscriptionFrame`, and a new `ProsodyEventFrame` (defined by us, mapped to `types.ProsodyFrame` by `ProsodyWriter` in R3).
- **Lifecycle**: opens the WS on first audio frame, closes on `EndFrame`. Emits a control frame on `PersonaSwitchFrame` to swap config (R6 wiring).

| | |
|---|---|
| ✅ | Matches spec intent exactly. Live prosody events, integrated latency, EVI's CLM webhook path stays available for R4. |
| ✅ | Bounded scope — Hume EVI's WS protocol is documented, Pipecat's `FrameProcessor` API is small. Estimate ~300–500 lines + tests. |
| ✅ | Owned by us, evolved with our needs. The mapping from EVI events to `types.ProsodyFrame` is rehearse-specific anyway. |
| ❌ | Real engineering work, not configuration. Adds a session of build time before R2's manual demo lands. |
| ❌ | Maintenance burden: Hume API changes are ours to track. Pipecat frame-type changes are ours to track. |
| ❌ | Open question Q1 from runtime spec (config hot-swap latency) becomes our problem to spike rather than Pipecat's to document. |
| ❌ | Tests harder — EVI's WS protocol needs to be faked or recorded for unit tests. |

### Option 2 — Pin Pipecat to a version that shipped `HumeEVIService`

Investigate whether `HumeEVIService` ever existed in any Pipecat release (or under a different name like `HumeVoiceService`, `EVIService`).

| | |
|---|---|
| ✅ | If it exists somewhere, free integration — same as the original spec assumption. |
| ❌ | Unverified. Initial check (Pipecat 1.1.0, Hume submodule has only `tts.py`) suggests it's never been there. Worth maybe an hour of investigation, not more. |
| ❌ | Pinning Pipecat backward locks the rest of the codebase to old framework APIs. Pipecat moves fast; rest of the project probably wants current. |
| ❌ | Even if found, it might be deprecated/abandoned — no maintenance signal. |

**Time-boxed version of Option 2**: spend ≤30 min searching Pipecat releases + GitHub history. If nothing concrete turns up, drop and pick Option 1 or 4. Don't let this option become a sink.

### Option 3 — Replace EVI with STT → LLM → TTS chain

Use Pipecat's existing services: Deepgram STT (or AssemblyAI) → Anthropic LLM → Hume TTS. No custom code; all three services have first-class Pipecat integrations.

| | |
|---|---|
| ✅ | Works today. Zero custom code. R2 lands fast. |
| ✅ | Each component is independently swappable — fits "model slots" model in runtime spec §9. |
| ❌ | **No live prosody event stream.** Deepgram (and other STTs) don't emit Hume-style emotion vectors. Prosody capture moves to a post-call batch over the recorded WAV. Major design change. |
| ❌ | **No CLM webhook path.** R4's design assumes Hume calls our webhook on every EVI turn; with this chain we'd integrate Claude as a Pipecat `LLMService` directly — different code path, different system prompt plumbing. |
| ❌ | **Latency hit.** Three sequential network hops (STT WS, LLM HTTP, TTS WS) instead of one integrated EVI session. The 800ms p50 budget gets tighter. |
| ❌ | One-way door from rehearse's value prop (prosody as a training signal). Reversing later means re-doing R2 + R3 + parts of R4. |

### Option 4 — Defer the bridge; stub `/media/{id}` and proceed

Ship R2 as a no-op WebSocket that accepts and logs Twilio audio frames, then writes a final "echo" log line on hangup. Move to R3 (writers) and R4 (Claude CLM via mocked transport) on a `SimulatedTransport` that injects synthetic frames matching what EVI *would* emit. Build the real Hume bridge as a focused R2.5 PR after R3/R4 expose what frame shapes we actually need.

| | |
|---|---|
| ✅ | Unblocks the rest of the runtime build. R3 (writers, phase machine) and R4 (Claude CLM endpoint, agents) are mostly orthogonal to the audio-in path and can be developed against simulated frames. |
| ✅ | Forces the `SimulatedTransport` discipline early — same rig the eval harness needs anyway (eval spec §X). Builds the seam that lets eval validate the live system. |
| ✅ | When the Hume bridge lands (as Option 1 in a focused PR), the frame-shape contract has already been pinned by R3/R4 use, reducing churn. |
| ❌ | No end-to-end voice demo until the bridge lands. Manual voice testing waits. |
| ❌ | Risk of R3/R4 simulating frames that don't match real EVI output, requiring rework when the bridge arrives. Mitigation: write the simulation against Hume EVI's documented event schema, not against guesses. |
| ❌ | Schedule slip on the "hear yourself talking to a Hume voice" milestone the spec uses to sanity-check direction. |

## 4. Recommendation

**Option 1 (build the bridge)** if a full session of build time is acceptable now. It matches the spec's intent, keeps prosody as a first-class event, and the work is bounded.

**Option 4 (defer)** if the priority is keeping the runtime build phase-paced. It exchanges the R2 voice demo for unblocking R3/R4 against simulated frames, with the bridge landing as a focused PR later.

**Time-box Option 2 to 30 min** before committing to either of the above. If a usable `HumeEVIService` exists in Pipecat history, take it. If not, drop fast.

**Reject Option 3.** The prosody loss is a one-way door from the product's design center. Worth keeping in the back pocket only if Hume EVI itself becomes unworkable.

## 5. If Option 1 — implementation sketch

Concrete to anchor the estimate:

```
rehearse/services/hume_evi.py  (~300–500 lines)

class ProsodyEventFrame(Frame):
    """Custom Pipecat frame carrying one prosody sample."""
    utterance_id: str
    speaker: Speaker
    scores: ProsodyScores
    ts_start: float
    ts_end: float

class HumeEVIService(FrameProcessor):
    def __init__(self, api_key: str, config_id: str, sample_rate: int = 8000): ...
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # AudioRawFrame → forward bytes to EVI WS
        # EndFrame      → close WS
        # PersonaSwitchFrame (R6) → send EVI session_settings update
    async def _consume_evi_events(self) -> None:
        # async iterate over hume.empathic_voice.ChatWebSocket
        #   - audio_output → push AudioRawFrame downstream
        #   - user_message / assistant_message → push TranscriptionFrame
        #   - user_interruption → push InterruptionFrame
        #   - on every message with prosody scores → push ProsodyEventFrame
```

Tests:

- Unit: `tests/test_hume_evi.py` — fake `AsyncHumeClient` that yields a recorded event stream; assert frame sequence emitted matches spec.
- Wiring: extend `tests/test_pipeline.py` (R3 builds) — confirm the service plugs into a Pipecat `Pipeline` without import errors.
- Manual: dial the Twilio number, talk to Hume's default LLM, hear it back. Same demo the runtime spec lists for R2.

## 6. If Option 4 — what R2 ships instead

A no-op pipeline + simulated-transport scaffolding:

- `/twilio/voice/inbound` keeps returning the canned `<Say>` TwiML (no `<Stream>` yet).
- `rehearse/pipeline.py` defines `build_pipeline(transport, ...)` returning a Pipecat `Pipeline` with a `LoggingProcessor` (writes received frames to `telemetry.jsonl`) — exercises the wiring without Hume.
- `rehearse/eval/transport.py` (already exists per repo layout) gains a `SimulatedTransport` that emits synthetic `AudioRawFrame` / `TranscriptionFrame` / `ProsodyEventFrame` per a script. The frame shapes here are the contract the future Hume bridge must hit.
- Manual demo: run a unit test that drives `build_pipeline(SimulatedTransport(...))` to completion and check artifacts wrote.

R2.5 (the deferred bridge) then becomes a single-component PR matching §5.

## 7. Decision log

(to fill in)

| Date | Decision | Rationale |
|---|---|---|
| | | |
