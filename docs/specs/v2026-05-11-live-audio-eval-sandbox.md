# rehearse — Spec: Live-Audio Eval Sandbox (v1)

**Status**: draft
**Owner**: jz
**Date**: 2026-05-11
**Depends on**:
- `docs/specs/v2026-05-07-runtime-eval-alignment.md` (RuntimeHost, transport seam)
- `docs/specs/v2026-05-11-voice-participant-protocol.md` (VoiceParticipant protocol)
- `rehearse/services/hume_evi.py`
- `rehearse/eval/tts_bridge.py`
- `rehearse/eval/environments/runtime_sandbox.py`
- `rehearse/frames.py` (AudioChunk, TranscriptDelta, ProsodyEvent)
**Amends**: `docs/specs/v2026-05-06-eval-system-roadmap.md` §Mini-spec 2 (second half — sandbox TTS integration)
**Supersedes**: nothing

---

## 0. One-line summary

Replace the post-hoc TTS synthesis hack with a live-audio eval sandbox that
drives the real `HumeEVIClient` (or any `AudioCoachAdapter` implementation)
during the rollout, capturing real per-turn audio and prosody for scoring.

---

## 1. Current state

The runtime-sandbox environment (`v2026-05-07-runtime-eval-alignment.md`)
closed the eval/serve skew for the **text path**: `RuntimeHost`,
`PhaseProcessor`, `IntakeProcessor`, and the persona compiler now run
identically in eval and production.

**The voice path was explicitly deferred.** The current eval substitutes:

| Production | Current eval |
|---|---|
| `HumeEVIClient` drives the coach voice (EVI WebSocket, emotion-aware TTS, ASR) | `TextOnlyCoachAdapter` calls Claude directly, returns text |
| Audio flows bidirectionally over Twilio/Hume WebSocket | `InMemoryTwoWayChannel` carries text only |
| `AudioChunk` frames on FrameBus → `AudioRecorder` → per-turn WAVs | No audio. `AudioRecorder` is skipped |
| `ProsodyEvent` frames carry per-turn emotion scores from Hume | No prosody signal |
| `DeliveryJudgeScorer`, `AffectPerceptionJudgeScorer` score real voice | Score post-hoc synthetic audio |

After the rollout, `RuntimeSandboxEnvironment` synthesizes audio from the
transcript via `HumeOctaveProvider` and feeds it to the audio judges. This
produces what looks like audio scores but is not:

- The coach **never had voice context** when generating responses — Claude
  produced text without awareness of how it would be spoken or heard.
- Hume Octave synthesizes each turn in isolation, with no emotional
  continuity from the conversation.
- `naturalness.speech_rate_band` scores `0.00` on every run. The other
  audio scores are measuring TTS quality, not coaching quality.

**The eval currently has zero coverage of the voice stack.** A broken EVI
configuration, a regression in Hume's prosody model, or a swap to a
different voice provider would be invisible to every metric in
`voice-rollout-judges`.

---

## 2. Why this matters

rehearse's differentiated value is **the voice**. The system is not a
text chatbot that happens to run over the phone. The coach reads emotional
incongruence between what the caller says and how they say it. The
counterparty delivers stress in a way that forces the caller to practice
under realistic pressure. Naturalness, pacing, and emotional calibration are
not decorations — they are the product.

The current eval measures:

- Whether Claude's text content moved the user toward their goal (`content_quality`)
- Whether the runtime correctly captured intake fields (`intake_fidelity`)

It does not measure:

- Whether the coach correctly read the caller's emotional state from audio (`affect_perception`)
- Whether the coach's prosody and pacing matched the emotional moment (`delivery_quality`)
- Whether the conversation felt natural — no interruptions, appropriate
  silence, comfortable speech rate (`naturalness.*`)
- Any of the above when EVI is swapped for a different voice provider or an
  internal model

This means:
1. **We cannot detect regressions in the voice stack.** A Hume EVI config
   change, a model update from Hume, or a misconfigured persona could
   silently degrade call quality — no eval would catch it.
2. **We cannot compare voice providers.** The SPEC.md design commitment
   "model slots, not model choices" requires that swapping voice providers is
   driven by eval-deltas. Without a live-audio eval, there are no deltas to
   compare.
3. **We cannot generate real multimodal training data from eval.** The RL
   training loop (v2026-05-05-multimodal-trajectory-rubric-rlaif.md) needs
   artifact sets where audio, prosody, and transcript are co-produced by the
   same model call. Post-hoc TTS breaks this coupling.

**The outcome we are building for:** a single `make eval-voice-rollout` that
exercises the full production voice stack end-to-end — EVI (or any plugged-in
audio model) hears synthetic caller audio, responds with real voice, and is
scored on actual prosody and emotional calibration. The same command works
with a different `AudioCoachAdapter` to compare providers or evaluate an
internal model under development.

---

## 3. Non-goals

- Replacing Twilio in the eval (caller-side telephony is out of scope; the
  synthetic caller uses in-memory audio transport, not a real phone call).
- Training. This spec only closes the eval/serve seam for audio.
- Corpus-level stability scoring across providers (deferred to Mini-spec 8 /
  `StabilityScorer`).
- Real-time streaming audio to a human evaluator during an eval rollout.
- Cost optimisation of EVI session pricing (noted as a constraint, not a
  design target).

---

## 4. Design commitments

1. **`AudioCoachAdapter` is the voice seam.** `RuntimeHost` never imports
   `HumeEVIClient` directly. The adapter protocol is the only surface that
   changes when the voice provider changes. Hume EVI, an internal model, and
   a test stub all implement the same protocol.

2. **The synthetic caller drives audio, not text.** `AudioCustomerDriver`
   synthesizes PCM16 audio from scenario text and sends it through the
   transport as `kind="audio"` events. The runtime receives audio as it would
   from a real caller — it does not know it is speaking to a TTS engine.

3. **Per-turn audio is captured by the runtime, not reconstructed afterward.**
   `AudioChunk` frames from EVI flow through `FrameBus` to `AudioRecorder`
   exactly as in production. The audio judges read files that were written
   during the conversation, not synthesized from a transcript afterward.

4. **Parallelism is preserved.** Each rollout owns its own `HumeEVIClient`
   instance and WebSocket session. `asyncio.gather` in the runner continues
   to work; the concurrency bound is Hume's API rate limit, not the harness.

5. **Text eval path is unchanged.** `RuntimeSandboxEnvironment` with
   `TextOnlyCoachAdapter` continues to exist and is the default when
   `HUME_API_KEY` is not set. Live-audio mode is opt-in via
   `--environment live-audio-sandbox`.

6. **Provider swap is one class.** Replacing Hume EVI with an internal
   model requires implementing `AudioCoachAdapter` in one new file and
   passing it to `LiveAudioSandboxEnvironment`. No changes to `RuntimeHost`,
   `telephony.py`, or any scorer.

---

## 5. Interfaces

### 5.1 `AudioCoachAdapter` — the voice seam

Replaces the text-only `CoachVoiceAdapter` for audio-capable providers.
EVI is not request-response; it is a stateful event-driven WebSocket.

```python
# rehearse/runtime.py

class CoachTurnEvent:
    """One event emitted by the coach during or after a turn."""

@dataclass(frozen=True)
class CoachTranscriptEvent(CoachTurnEvent):
    text: str
    utterance_id: str

@dataclass(frozen=True)
class CoachAudioEvent(CoachTurnEvent):
    pcm16_16k: bytes

@dataclass(frozen=True)
class CoachProsodyEvent(CoachTurnEvent):
    scores: dict[str, float]      # emotion label → probability

@dataclass(frozen=True)
class CoachTurnComplete(CoachTurnEvent):
    pass


class AudioCoachAdapter(Protocol):
    """Stateful session: receives caller audio, yields coach events.

    Lifecycle: acquired via async context manager per RuntimeHost.run() call.
    send_audio() and end_of_turn() may be called from the _run_loop coroutine.
    events() is consumed by a parallel task on the same event loop.
    """

    async def __aenter__(self) -> AudioCoachAdapter: ...
    async def __aexit__(self, *args: object) -> None: ...

    async def send_audio(self, pcm16_16k: bytes) -> None:
        """Stream one chunk of caller PCM16 audio to the provider."""

    async def end_of_turn(self) -> None:
        """Signal that the caller's utterance is complete.
        Used in eval mode (VAD is synthetic); in serving mode
        EVI's built-in VAD handles this automatically."""

    def events(self) -> AsyncIterator[CoachTurnEvent]:
        """Async stream of coach events for the current session."""
```

**Concrete implementations in v1:**

| Class | File | Description |
|---|---|---|
| `HumeEVIAdapter` | `rehearse/runtime.py` | Wraps `HumeEVIClient`. Bridges `send_audio` → `hume.send_audio`, `events()` → FrameBus subscriber reading `TranscriptDelta`, `AudioChunk`, `ProsodyEvent`. |
| `TextOnlyCoachAdapter` | `rehearse/runtime.py` | Existing. Implements `AudioCoachAdapter` by ignoring audio; calls Claude on `end_of_turn()` using accumulated transcript. Text eval path. |
| `StubAudioCoachAdapter` | `rehearse/runtime.py` | Returns scripted responses. Used in unit tests; requires no API keys. |

**Backward compatibility:** `TextOnlyCoachAdapter` gains `send_audio` (no-op)
and `end_of_turn` (triggers Claude call). Its existing `respond(user_text)` is
kept internally but is no longer part of the public protocol.

### 5.2 `AudioCustomerDriver`

Generates caller-side audio from a scenario and sends it through the
transport. Phase-aware; sends first turn immediately without awaiting a
greeting, consistent with the initiation protocol in §5.3 of
`v2026-05-07-runtime-eval-alignment.md`.

```python
# rehearse/eval/customers/audio_customer.py

class AudioCustomerDriver:
    """TTS-backed synthetic caller. Sends PCM16 audio, not text."""

    name = "audio-customer"
    version = "v1"

    def __init__(
        self,
        scenario: dict[str, Any],
        tts: TTSProvider,
        *,
        llm_client: Any = None,
        run_dir: Path | None = None,
    ) -> None: ...

    async def run(
        self,
        *,
        transport: TwoWayChannel,
        runtime_phase: Callable[[], Phase],
    ) -> CallerResult:
        """
        Per turn:
        1. Generate caller utterance text (same LLM prompts as SyntheticCaller).
        2. Synthesize to PCM16 via tts.synthesize_pcm(text, description=emotional_state).
        3. Stream PCM16 chunks as kind="audio" transport events.
        4. Send kind="control", event="end_of_turn" to signal speech boundary.
        5. Await coach audio back (kind="audio") or phase_transition control event.
        6. Decode and buffer coach audio for CallerResult artifact write.
        """
```

`TTSProvider` gains a `synthesize_pcm()` method alongside the existing
`synthesize()` (which writes a WAV file). Returns `bytes` (PCM16, 16kHz mono).

```python
class TTSProvider(Protocol):
    name: str

    async def synthesize(self, *, text: str, out_path: Path,
                         description: str | None = None) -> float: ...

    async def synthesize_pcm(self, *, text: str,
                              description: str | None = None) -> bytes:
        """Return raw PCM16 16kHz mono bytes. Default impl decodes synthesize() WAV."""
```

### 5.3 `RuntimeHost` audio loop

`RuntimeHost._run_loop` currently drops `kind="audio"` events. In audio
mode it must:

1. Route `kind="audio"` events to `self._coach.send_audio(event.data)`.
2. Route `kind="control", event="end_of_turn"` to `self._coach.end_of_turn()`.
3. Run a parallel task consuming `self._coach.events()`:
   - `CoachTranscriptEvent` → publish `TranscriptDelta(speaker=COACH)` on bus
     and `transport.send("text", {"text": ..., "role": "coach"})`.
   - `CoachAudioEvent` → publish `AudioChunk(speaker=COACH)` on bus.
   - `CoachProsodyEvent` → publish `ProsodyEvent` on bus.
   - `CoachTurnComplete` → signal the run loop that the turn is done.
4. The existing text branch remains for `TextOnlyCoachAdapter`.

`RuntimeHost.__init__` infers mode from the adapter type:

```python
self._audio_mode = isinstance(coach, AudioCoachAdapter) and not isinstance(
    coach, TextOnlyCoachAdapter
)
```

### 5.4 `LiveAudioSandboxEnvironment`

```python
# rehearse/eval/environments/live_audio_sandbox.py

class LiveAudioSandboxEnvironment:
    name = "live-audio-sandbox"
    version = "v1"

    def __init__(
        self,
        model_slots: dict[str, str] | None = None,
        *,
        coach_adapter_factory: Callable[[], AudioCoachAdapter] | None = None,
    ) -> None:
        """
        coach_adapter_factory: callable that returns a fresh AudioCoachAdapter
        per rollout. Defaults to HumeEVIAdapter(api_key=HUME_API_KEY,
        config_id=HUME_CONFIG_ID). Pass a custom factory to evaluate a
        different voice provider.

        Preflight: raises immediately if ANTHROPIC_API_KEY or HUME_API_KEY
        are not set (unless a coach_adapter_factory is provided that does not
        require them).
        """

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        """
        1. Build AudioCustomerDriver (SyntheticCaller text prompts + HumeOctave TTS).
        2. Build coach = coach_adapter_factory() → fresh HumeEVIAdapter.
        3. async with coach:
               await asyncio.gather(
                   host.run(session_id=..., transport=transport.runtime),
                   customer.run(transport=transport.customer, ...),
               )
        4. RolloutResult includes token_usage (coach LLM calls via EVI CLM).
        """
```

Each rollout gets its own `HumeEVIClient` instance and WebSocket connection.
`asyncio.gather` in the runner parallelises across rollouts as before; the
practical concurrency bound is `HUME_API_KEY`'s concurrent-session limit
(typically 10–20 simultaneous sessions).

### 5.5 `AudioRecorder` wired in eval

`AudioRecorder` is currently in the **skip** column for eval mode (§6 of
`v2026-05-07-runtime-eval-alignment.md`). `LiveAudioSandboxEnvironment`
enables it unconditionally — its purpose is precisely to write per-turn WAVs
for the audio judges.

`RuntimeHost.__init__` accepts `enable_audio_recording: bool = False`.
`LiveAudioSandboxEnvironment` passes `enable_audio_recording=True`.

---

## 6. What's real vs. stubbed in live-audio mode

| Component | `runtime-sandbox` (text) | `live-audio-sandbox` (audio) | Serving |
|---|---|---|---|
| RuntimeHost | real | real | real |
| PhaseProcessor | real | real | real |
| IntakeProcessor | real | real | real |
| PersonaCompiler | real | real | real |
| Coach CLM (text) | TextOnlyCoachAdapter → Claude direct | HumeEVIAdapter → EVI → CLM (configured model) | HumeEVIAdapter → EVI → CLM |
| Coach TTS | none (text frames only) | **real** (EVI native TTS) | **real** (EVI native TTS) |
| Coach prosody | none | **real** (EVI ProsodyEvent) | **real** |
| Caller ASR | synthetic text | **real** (EVI ASR on PCM16) | Twilio ASR → Hume |
| Caller audio | none | AudioCustomerDriver TTS → PCM16 | live human |
| AudioRecorder | skipped | **real** (per-turn WAVs written) | real |
| per-turn WAVs | post-hoc Octave synthesis | **real** (from EVI audio output) | real |
| Audio judges | score fake audio | **score real audio** | (production) |
| Naturalness | measures synthetic artefact | **measures real conversation** | (production) |
| TelemetryLogger | skipped | skipped | real |

The `live-audio-sandbox` column is the production serving column with two
substitutions: the caller is synthetic (AudioCustomerDriver) and the
transport is in-memory. Everything else — EVI, CLM, TTS, ASR, audio
recording — is the real production stack.

---

## 7. Provider swap contract

To evaluate a different voice provider or an internal model:

```python
class MyModelAdapter:
    """Example: internal voice model implementing AudioCoachAdapter."""

    async def __aenter__(self) -> MyModelAdapter:
        # open WebSocket / gRPC / in-process connection
        return self

    async def __aexit__(self, *args: object) -> None:
        # close connection
        ...

    async def send_audio(self, pcm16_16k: bytes) -> None:
        # stream to your model
        ...

    async def end_of_turn(self) -> None:
        # signal VAD boundary
        ...

    def events(self) -> AsyncIterator[CoachTurnEvent]:
        # yield CoachTranscriptEvent, CoachAudioEvent, CoachProsodyEvent,
        # CoachTurnComplete as your model produces them
        ...


# Run eval with your model:
uv run rehearse-eval run \
    --eval voice-rollout-judges \
    --environment live-audio-sandbox \
    --model-slot coach_adapter=my_model
```

`LiveAudioSandboxEnvironment` resolves `coach_adapter_factory` from
`model_slots["coach_adapter"]` if provided, else defaults to
`HumeEVIAdapter`.

---

## 8. Migration path (4 phases)

### Phase 1 — `AudioCoachAdapter` protocol + `HumeEVIAdapter`

1. Define `CoachTurnEvent` union and `AudioCoachAdapter` protocol in
   `rehearse/runtime.py`.
2. Implement `HumeEVIAdapter`: wraps `HumeEVIClient.__aenter__/__aexit__`,
   bridges `send_audio`, `end_of_turn`, and `events()` (reads FrameBus
   subscriber populated by `run_event_loop`).
3. Implement `StubAudioCoachAdapter` for unit tests.
4. Extend `TextOnlyCoachAdapter` with no-op `send_audio` and `end_of_turn`
   (triggers existing Claude call); make it satisfy `AudioCoachAdapter`.
5. Tests: `test_runtime_host.py` gains audio-mode codepaths (stub adapter).

Verification: existing `runtime-sandbox` tests pass unchanged; new
audio-mode tests pass with `StubAudioCoachAdapter`.

### Phase 2 — Audio loop in `RuntimeHost`

1. Add `enable_audio_recording: bool = False` to `RuntimeHost.__init__`.
2. Modify `_run_loop` to handle `kind="audio"` and `kind="control",
   event="end_of_turn"` events; spawn coach events consumer task.
3. Wire `AudioRecorder` when `enable_audio_recording=True`.
4. Tests: `test_runtime_host.py` — audio frame routing, per-turn WAV write,
   `CoachTurnComplete` unblocks next customer turn.

Verification: `pytest tests/test_runtime_host.py` green; manual check that
`audio/coach/turn_0.wav` exists after a stub-adapter rollout.

### Phase 3 — `AudioCustomerDriver` + `TTSProvider.synthesize_pcm`

1. Add `synthesize_pcm()` to `TTSProvider` protocol and `HumeOctaveProvider`
   (decode WAV bytes from existing `synthesize()` output).
2. Implement `AudioCustomerDriver` in
   `rehearse/eval/customers/audio_customer.py`: same LLM prompts as
   `SyntheticCaller`, TTS per turn, sends PCM16 chunks + `end_of_turn`.
3. Tests: `test_audio_customer_driver.py` — sends audio before any runtime
   message; phase transition switches TTS description; turn cap respected;
   `customer_driver.json` written.

Verification: `AudioCustomerDriver` + `StubAudioCoachAdapter` + `RuntimeHost`
gather completes without deadlock; per-turn WAVs present.

### Phase 4 — `LiveAudioSandboxEnvironment` + eval wiring

1. Implement `LiveAudioSandboxEnvironment` in
   `rehearse/eval/environments/live_audio_sandbox.py`.
2. Register in `rehearse/eval/environments/__init__.py`.
3. Add `live-audio-sandbox` to `voice-rollout-judges`
   `supported_environments`.
4. Add `make eval-voice-rollout-audio` Makefile target.
5. Tests: `test_live_audio_sandbox_rollout.py` — preflight raises without
   keys; gather completes with stub adapter; all 5 base artifacts + per-turn
   WAVs written; audio judge scorers return non-flagged scores.

Verification: `make eval-voice-rollout-audio` (with real keys) completes;
`summary.md` shows non-zero `affect_perception` and `delivery_quality`
without `audio_missing` flags; `naturalness.speech_rate_band` is no longer
always `0.00`.

---

## 9. Repo additions

```
rehearse/
  runtime.py                           [modified] AudioCoachAdapter protocol,
                                         HumeEVIAdapter, StubAudioCoachAdapter,
                                         TextOnlyCoachAdapter audio extensions,
                                         RuntimeHost audio loop + AudioRecorder wiring
  eval/
    customers/
      audio_customer.py                [new] AudioCustomerDriver
    environments/
      live_audio_sandbox.py            [new] LiveAudioSandboxEnvironment
    tts_bridge.py                      [modified] synthesize_pcm() on TTSProvider +
                                         HumeOctaveProvider
tests/
  test_runtime_host.py                 [modified] audio-mode codepaths
  test_audio_customer_driver.py        [new]
  test_live_audio_sandbox_rollout.py   [new]
Makefile                               [add eval-voice-rollout-audio target]
```

**Test specifications:**

`tests/test_runtime_host.py` additions:
- Audio frame routed to `coach.send_audio()`, not dropped
- `end_of_turn` control event calls `coach.end_of_turn()`
- `CoachAudioEvent` → `AudioChunk` published on bus
- `CoachTranscriptEvent` → `TranscriptDelta(speaker=COACH)` published on bus
- `AudioRecorder` writes `audio/coach/turn_0.wav` when `enable_audio_recording=True`
- Text mode unchanged with `TextOnlyCoachAdapter`

`tests/test_audio_customer_driver.py`:
- Sends audio before any runtime message (initiation protocol holds for audio mode)
- `phase_transition` control event → switches TTS emotional description
- PCM16 chunks precede `end_of_turn` control event on every turn
- Hard turn cap: `customer_driver.json` shows `turns_per_phase` capped
- `CallerResult.token_usage` populated from LLM calls

`tests/test_live_audio_sandbox_rollout.py`:
- Missing `HUME_API_KEY` → raises at init (not mid-rollout)
- Gather with `StubAudioCoachAdapter` completes; all 5 base artifacts written
- Per-turn WAVs at `audio/coach/turn_N.wav` and `audio/user/turn_N.wav`
- `DeliveryJudgeScorer` returns score without `audio_missing` flag
- `provenance.json` reflects real vs. stub components accurately

---

## 10. Acceptance criteria

A change is accepted when all of the following are true:

1. `make eval-voice-rollout-audio` completes with `HUME_API_KEY` and
   `ANTHROPIC_API_KEY` set; `rehearse-eval show <run_id>` shows non-flagged
   scores for `affect_perception`, `delivery_quality`, and all
   `naturalness.*` dimensions.

2. `naturalness.speech_rate_band` is not `0.00` on at least one rollout
   (verifies real audio is being scored, not silent WAVs or synthetic artefacts).

3. `grep -r "hume\|HumeEVI" rehearse/eval/` returns no results outside
   `environments/live_audio_sandbox.py` and `tts_bridge.py` — the
   environment is the only caller site, not scattered through the harness.

4. Swapping `HumeEVIAdapter` for `StubAudioCoachAdapter` in
   `LiveAudioSandboxEnvironment` requires zero changes outside
   `live_audio_sandbox.py`. Verified by a test that passes the stub factory.

5. `runtime-sandbox` (text) eval continues to run without `HUME_API_KEY`
   and produces the same scores as before this spec ships.

6. `pytest tests/` green with no `live_api` tests required for the non-Hume
   path.

---

## 11. Open questions

1. **EVI VAD vs. eval VAD.** In production, Hume's VAD detects turn
   boundaries. In eval, `AudioCustomerDriver` sends `end_of_turn` explicitly.
   Should `HumeEVIAdapter` suppress EVI's VAD to avoid double-triggering, or
   let EVI handle it naturally from the audio stream? Recommendation: let EVI
   handle it — synthetic PCM16 will trigger EVI's VAD naturally, and
   `end_of_turn` serves as a fallback only if VAD misses the boundary.

2. **EVI concurrent session limit.** Default concurrency in the runner is 4.
   Hume's limit may be lower on some API tiers. The `LiveAudioSandboxEnvironment`
   should surface a clear error when the limit is hit rather than silently
   timing out rollouts.

3. **Cost gate.** A 12-turn eval rollout over EVI costs roughly $0.10–$0.20
   depending on audio duration. A 20-example eval run is ~$2–4. Whether to
   add a `--confirm-spend` CLI flag for `live-audio-sandbox` is left to
   implementation judgment.

---

## 12. Deferred

- **Internal voice model adapter.** `AudioCoachAdapter` is the seam; the
  adapter for rehearse's own model will be a new implementation of the
  protocol. No changes to `RuntimeHost`, scorers, or environment are needed
  when it ships.
- **Caller-side prosody scoring.** The current `AffectPerceptionJudgeScorer`
  reads caller audio to assess whether the coach read the caller's affect
  correctly. v1 uses `HumeOctave` TTS for the caller, which does not carry
  the emotional nuance of a real caller. Replacing the caller TTS with an
  emotion-conditioned voice model (or production-replay audio) is a follow-up
  spec.
- **A/B provider comparison eval.** Running `HumeEVIAdapter` and
  `YourModelAdapter` against the same scenario set and diffing scores is the
  mechanism for provider selection. The harness supports it via
  `coach_adapter_factory`; a dedicated eval workflow and comparison report
  are deferred.
- **Cost budget enforcement.** Hard spending caps, run-level cost estimates,
  and `--dry-run` cost projection for `live-audio-sandbox` are deferred.

---

## GSTACK REVIEW REPORT

| Review | Trigger | Runs | Status | Findings |
|---|---|---|---|---|
| Eng Review | `/plan-eng-review` | 0 | — | — |
| DX Review | `/plan-devex-review` | 0 | — | — |

- **VERDICT:** draft — awaiting eng review before implementation begins
