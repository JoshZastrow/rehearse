# rehearse — Development Guide

Instructions for AI coding assistants and developers working on the rehearse codebase.

## Development Environment

```bash
uv sync          # install all deps (including dev group)
source .venv/bin/activate
cp .env.example .env   # then fill in API keys
```

Required env vars for a full run: `ANTHROPIC_API_KEY`, `TWILIO_ACCOUNT_SID`,
`TWILIO_AUTH_TOKEN`, `HUME_API_KEY`. For eval-only work only `ANTHROPIC_API_KEY`
is required.

## Project Structure

```
rehearse/
├── app.py                 # FastAPI app factory — mounts telephony routes
├── telephony.py           # Live runtime entrypoint — Twilio webhooks, media stream
├── runtime.py             # RuntimeHost — single entry point for both eval and serving
├── transport.py           # RuntimeTransport / InMemoryTwoWayChannel / TwoWayChannel
├── bus.py                 # FrameBus — internal fan-out pub/sub for runtime tasks
├── frames.py              # All frame types (TranscriptDelta, PhaseSignal, IntakeComplete, …)
├── phases.py              # PhaseProcessor + PhaseBudgets — INTAKE/PRACTICE/FEEDBACK control
├── intake.py              # IntakeProcessor — extracts situation/goal from transcript
├── personas.py            # build_intake_record(), compile_character() — persona compiler
├── session.py             # SessionOrchestrator, Session model, utcnow()
├── storage.py             # LocalFilesystemStore — artifact persistence
├── types.py               # Shared types: Phase, Speaker, Session, RubricScore, …
├── config.py              # RuntimeConfig — loaded from env
├── consent.py             # ConsentGate
├── outcome.py             # OutcomeProbe
├── synthesis.py           # Text→SSML helpers
├── agents/
│   ├── clm.py             # Core LLM calls — the coach loop
│   ├── persona_router.py  # Persona routing logic
│   └── persona_swap.py    # Mid-call persona swap coordinator
├── audio/
│   ├── twilio_stream.py   # TwilioStream — reads/writes Twilio media frames
│   └── mulaw.py           # μ-law encode/decode
├── services/
│   └── hume_evi.py        # HumeEVIClient — Hume EVI websocket
├── writers/
│   └── artifacts.py       # TranscriptWriter, TimingWriter (subscribe to FrameBus)
└── eval/                  # Eval harness — see below
```

**Artifact layout (one session):**
```
evals/runs/{run_id}/sessions/{example_id}/
├── session.json           # Session manifest (phases, consent, timestamps)
├── transcript.jsonl       # One TranscriptFrame per line
├── intake.json            # IntakeRecord — situation, goal, stakes, relationship
├── persona.json           # PersonaRecord — compiled counterparty character
├── phase_timing.json      # Phase entry/exit timestamps
├── customer_driver.json   # SyntheticCaller turn counts per phase
└── provenance.json        # Which components ran real vs stub
```

## Key Classes

### `RuntimeHost` (`rehearse/runtime.py`)

Single entry point that boots one session against any transport. Same object in
serving and eval.

```python
class RuntimeHost:
    def __init__(
        self,
        store: LocalFilesystemStore,
        coach: CoachVoiceAdapter,
        *,
        budgets: PhaseBudgets | None = None,
        clock: Callable[[], datetime] = utcnow,
        phase_timeout_s: float = 60.0,
        bus: FrameBus | None = None,
    ) -> None: ...

    async def run(
        self,
        *,
        session_id: str,
        transport: RuntimeTransport,
        consent: ConsentState = ConsentState.GRANTED,
    ) -> SessionArtifacts: ...

    @property
    def current_phase(self) -> Phase: ...
```

Internally wires `PhaseProcessor`, `IntakeProcessor`, `TranscriptWriter`,
`TimingWriter` onto a `FrameBus`. Reads user text from `transport.receive()`,
publishes `TranscriptDelta` on the bus, calls `coach.respond()`, and sends
the reply via `transport.send()`. Emits `phase_transition` control events to
the transport on every `PhaseSignal`.

**In serving:** `telephony.py:mount_twilio_routes` builds a `TwilioPhoneBridge`
transport and a `HumeCoachAdapter`, then calls `RuntimeHost.run`.

**In eval:** `RuntimeSandboxEnvironment` builds an `InMemoryTwoWayChannel`
and a `TextOnlyCoachAdapter`, then `asyncio.gather`s `host.run` and
`customer.run`.

### `FrameBus` (`rehearse/bus.py`)

Fan-out pub/sub for internal runtime tasks. Every task subscribes independently
and receives every frame.

```python
bus = FrameBus(session_id)
await bus.publish(frame)
async for frame in bus.subscribe():   # each call returns a new AsyncIterator
    ...
await bus.aclose()
```

`bus.subscribe()` must be called **before** the first `publish()`. `RuntimeHost`
gives each task a one-tick grace via `await asyncio.sleep(0)` after creating
tasks and before entering the main loop.

### `PhaseProcessor` (`rehearse/phases.py`)

Subscribes to `FrameBus`, tracks transcript turns, and transitions between
`INTAKE → PRACTICE → FEEDBACK` based on cue-based heuristics and `PhaseBudgets`
time limits. Emits `PhaseSignal` frames on the bus. Awaits `IntakeComplete`
before emitting the `INTAKE→PRACTICE` signal when `wait_for_intake_complete=True`
(always True in `RuntimeHost`).

### `IntakeProcessor` (`rehearse/intake.py`)

Subscribes to `FrameBus` during INTAKE. Builds an `IntakeRecord` from
transcript turns, then calls `compile_character()` to produce a `PersonaRecord`
for the PRACTICE phase. Emits `IntakeComplete` on the bus when done (with an
`error` flag if it fails, so `PhaseProcessor` can still advance).

### `CoachVoiceAdapter` (`rehearse/runtime.py`)

```python
class CoachVoiceAdapter(Protocol):
    async def respond(self, user_text: str, session_id: str) -> str: ...

class TextOnlyCoachAdapter:   # eval mode — calls Anthropic API directly
class HumeCoachAdapter:       # serving mode — routes through HumeEVIClient
```

`RuntimeHost` never imports `HumeEVIClient` directly — it calls `coach.respond()`.

## File Dependency Chain

```
rehearse/frames.py  (no deps — imported by all runtime files)
rehearse/types.py   (no deps — imported by all)
rehearse/bus.py     (imports frames, types)
       ↑
rehearse/phases.py, intake.py, writers/  (subscribe to FrameBus)
       ↑
rehearse/runtime.py  (wires bus + processors + transport)
       ↑
rehearse/telephony.py  (serving entrypoint — builds TwilioBridgeTransport + RuntimeHost)
rehearse/eval/environments/runtime_sandbox.py  (eval entrypoint — builds InMemoryTwoWayChannel + RuntimeHost)
```

## Eval Harness (`rehearse/eval/`)

```
rehearse/eval/
├── cli.py                     # rehearse-eval CLI entry point
├── runner.py                  # EvalRunner — orchestrates rollouts, writes run artifacts
├── protocols.py               # BenchmarkExample, RolloutResult, Scorer, Environment protocols
├── evals/                     # Eval definitions (dataset + scoring plan + environments)
├── datasets/                  # Dataset loaders (only load examples)
├── environments/              # Environment implementations (run the system under test)
│   ├── runtime_sandbox.py     # ← PRIMARY eval environment (real RuntimeHost)
│   └── voice_agent_sandbox.py # DEPRECATED shim — emits DeprecationWarning, delegates
├── scorers/                   # Scorer implementations
│   ├── intake_fidelity.py     # IntakeFidelityScorer — checks intake.json vs expected
│   ├── content_judge.py       # ContentJudgeScorer via DeepEval G-Eval
│   ├── delivery_judge.py      # DeliveryJudgeScorer (audio)
│   ├── naturalness.py         # NaturalnessScorer (timing-derived, no LLM)
│   └── aggregate.py           # AggregateScorer — weighted composite
├── customers/
│   └── llm_customer.py        # SyntheticCaller — phase-aware LLM synthetic caller
├── transports.py              # Backward-compat shim → rehearse/transport.py
├── deepeval_adapter/          # Bridge between rehearse scorers and DeepEval metrics
└── tts_bridge.py              # TTSProvider / HumeOctaveProvider for post-hoc audio
```

### Running evals

```bash
# Offline smoke test (no API key needed)
uv run rehearse-eval run --eval noop --environment echo

# Full runtime eval (needs ANTHROPIC_API_KEY)
make eval-voice-rollout               # stub TTS, --limit 3
make eval-voice-rollout-live          # real Gemini judges + Hume TTS

# Watch one rollout turn-by-turn
uv run rehearse-eval run --eval coach-dialogue-smoke --limit 1 --verbose

# View a previous run
uv run rehearse-eval show <run_id>

# List registered pieces
make eval-list
```

### `live_api` tests

Tests marked `@pytest.mark.live_api` hit real provider APIs and are **deselected
by default**. Run them explicitly when you need end-to-end coverage:

```bash
ANTHROPIC_API_KEY=... uv run pytest -m live_api
```

## Adding Pieces

Register new pieces in the matching package `__init__.py`.

### Adding an eval

1. Dataset in `rehearse/eval/datasets/your_dataset.py` — loads `BenchmarkExample` rows.
2. Eval in `rehearse/eval/evals/your_eval.py` — composes dataset + scoring plan +
   `supported_environments` + `preferred_environment`.
3. Register both in their `__init__.py`.

### Adding a scorer

Implement `async def score(self, example, rollout, run_id) -> list[RubricScore]`.

**Missing-artifact convention** (match `DeliveryJudgeScorer` / `NaturalnessScorer`):
- If a required artifact is absent, return a `RubricScore` with `value=0.0`,
  `flags=["<artifact>_missing"]`, and a descriptive `rationale`. **Do not raise.**
  Other scorers must still be able to run for the same example.

### Adding a `CustomerDriver`

Add to `rehearse/eval/customers/`. Implement:

```python
class MyDriver:
    name: str
    version: str

    async def run(
        self,
        *,
        transport: TwoWayChannel,
        runtime_phase: Callable[[], Phase],
    ) -> CallerResult: ...
```

**Initiation protocol:** the customer sends the first frame in every phase.
The runtime does not send a greeting. A driver that waits for a greeting will
deadlock. Switch prompts on `phase_transition` control events from the
transport — do not poll `runtime_phase()`.

## Testing

```bash
uv run pytest tests/                     # full suite (live_api tests excluded)
uv run pytest tests/ tests/eval/ -q      # include eval tests
uv run pytest -m live_api                # live API tests only (needs .env)
```

`asyncio_mode = "auto"` is configured in `pyproject.toml` — all async tests
work without explicit `@pytest.mark.asyncio`.

The suite currently has ~370 tests covering: `RuntimeHost`, `PhaseProcessor`,
`IntakeProcessor`, `SyntheticCaller`, `RuntimeSandboxEnvironment`,
`IntakeFidelityScorer`, transport move, and all scorer/eval protocols.

### Test isolation

- Use `tmp_path` (pytest fixture) for any test that writes session artifacts.
  `LocalFilesystemStore` accepts an arbitrary base directory.
- Never write to `evals/runs/` in tests — that's the live run output directory.
- For `RuntimeHost` unit tests, inject a stub `CoachVoiceAdapter` that returns
  fixed strings; don't call the real Anthropic API unless the test is marked
  `live_api`.
- For `InMemoryTwoWayChannel`, use `transport.customer` and `transport.runtime`
  as the two endpoints — they share queues, not a shared queue.

## Known Pitfalls

### FrameBus subscribers must be created before the first publish

`bus.subscribe()` must be called before `bus.publish()` fires the first frame.
`RuntimeHost` calls `asyncio.sleep(0)` after `create_task()` to give subscriber
coroutines one tick to enter their `async for` loop. If you add a new task
subscriber, create it before `_run_loop` starts.

### `asyncio.gather` does not propagate partial results

If `host.run()` raises mid-call (e.g. `RuntimePhaseTimeoutError`), `gather`
cancels the customer coroutine. The `RuntimeSandboxEnvironment` wraps the
`gather` in a try/except and falls through to write whatever artifacts exist.
Don't assume all 5 artifact files are present if `rollout.status == "error"`.

### Transport is closed after `gather` — don't reuse it

`InMemoryTwoWayChannel.close()` is idempotent but non-reversible. Create a
fresh channel per rollout.

### `TextOnlyCoachAdapter` is stateful per `session_id`

It accumulates conversation history keyed by `session_id`. In tests, use a
fresh adapter instance per test or pass a stub instead.

### `voice-agent-sandbox` is deprecated

`get_environment("voice-agent-sandbox", ...)` emits a `DeprecationWarning` to
stderr and delegates to `runtime-sandbox`. It will be removed after `runtime-sandbox`
has run green for one week (Phase 5 of the runtime-eval alignment spec). Do not
add new code paths that reference `voice-agent-sandbox`.

### Scorer errors must not block other scorers

The runner collects scorer results independently. A scorer that raises an
exception stops only that scorer — other scorers still run. Follow the
flagged-`RubricScore` convention (see §Adding a scorer above) instead of raising.
