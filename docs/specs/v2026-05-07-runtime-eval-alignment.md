# rehearse — Spec: Runtime ↔ Eval Alignment (v1)

**Status**: draft
**Owner**: jz
**Date**: 2026-05-07
**Depends on**: `rehearse/types.py`, `rehearse/phases.py`, `rehearse/intake.py`, `rehearse/personas.py`, `rehearse/telephony.py`, `rehearse/eval/environments/voice_agent_sandbox.py`
**Supersedes**: nothing (refactors `voice-agent-sandbox` env contract)

---

## 0. One-line summary

Make the eval harness drive the **same runtime code** that ships to production, so an eval rollout and a live phone call produce the same artifacts and are scored the same way. v1 is text-only; audio synthesis and call replay are deferred to follow-up specs.

## 1. Goal

A rehearse session is a phone call with three phases:

1. A customer (an LLM stub today, a real person tomorrow) places a call carrying **a situation** they need to rehearse and **an outcome** they want from it.
2. The runtime's INTAKE phase elicits both. The persona compiler turns what was elicited into a counterparty for PRACTICE.
3. The customer rehearses against that counterparty. FEEDBACK debriefs.
4. The verifier reads the persisted artifacts and answers two questions:
   - **Did intake correctly capture the situation and outcome?**
   - **Did the rehearsal close the gap between what the customer said and what they needed to say to land their goal?**

The current eval does not do step 1–3. It runs an LLM customer talking to a stand-in coach prompt (`_COACH_SYSTEM_PROMPT` in `voice_agent_sandbox.py`) that production never sees. That means we have no signal on whether intake is working, whether the persona compiler picks the right counterparty, or whether the live coach loop produces useful rehearsal — the three things the runtime actually does.

This spec wires the eval harness to the **same runtime code** the phone path runs. After v1:

- One command tells us whether a runtime change made the system better or worse than what's deployed.
- "A runtime change" includes the coach prompt, the intake parser (`personas.py:_infer_relationship` and friends), and the phase budgets in `PhaseBudgets`. All three change how a call goes, so all three must be exercised by the eval.
- Serving and eval write the same artifact set, so anything we score offline can also be scored on a real call.

## 2. Non-goals

- Training. This spec only wires the rollout/eval seam.
- Replacing existing scorers. Composite/affect/delivery/naturalness judges keep their current contracts.
- New telephony providers. Twilio/Hume bridges stay as-is.
- Replacing the JSONL scenario format. Existing `scenarios.jsonl` rows continue to work.
- Audio in the eval rollout (deferred). v1 is text-only.
- Replaying logged production calls (deferred to follow-up spec).

## 3. Design commitments

1. **One runtime, three drivers.** The runtime is a single object with one I/O contract. Synthetic LLM customer (v1), replay (deferred), and live phone (existing) all plug into the same contract.
2. **The customer is outside the runtime.** No customer simulator code lives under `rehearse/` — only under `rehearse/eval/`.
3. **Artifacts are the contract.** Everything downstream of a call (verifier, future replay, future training data) reads only the persisted artifact set. If serving and eval produce different artifacts, the seam is wrong.
4. **No coach prompt in the eval.** The eval owns the customer; the runtime owns the coach. The eval may not contain a coach prompt string.
5. **Phase-aware customer.** The synthetic customer drives INTAKE → PRACTICE → FEEDBACK behavior, not a single static role-play.
6. **Eval artifacts land in the existing run layout.** All session artifacts persist under `evals/runs/{run_id}/sessions/{example_id}/` (already created by `rehearse/eval/runner.py`).

## 4. Architecture

### 4.1 Today (broken)

```
[ scenarios.jsonl ]
        │
        ▼
[ LLMSandboxAgent(customer) ] ◀──text──▶ [ LLMSandboxAgent(coach, static prompt) ]
        │                                              │
        └──────────────► transcript.jsonl ◀────────────┘
                                │
                                ▼
                          [ verifiers ]
```

The runtime is bypassed. No phase controller, no intake processor, no persona compiler.

### 4.2 Target (v1, text-only)

```
                       ┌─────────── Eval Harness ───────────────────┐
                       │                                            │
[ scenarios.jsonl ] ──▶│  ┌──────── LLMCustomerDriver ────────┐     │
                       │  │  phase-aware: INTAKE / PRACTICE / │     │
                       │  │  FEEDBACK behavior switching      │     │
                       │  └────────────────┬──────────────────┘     │
                       │                   │ text frames             │
                       │                   ▼                         │
                       │       ┌── InMemoryDuplexTransport ──┐       │
                       │       └────────────┬─────────────────┘      │
                       │                    │                        │
                       │                    ▼                        │
                       │      ┌────── RuntimeHost ─────────────┐     │
                       │      │  PhaseProcessor                │     │
                       │      │  IntakeProcessor               │     │
                       │      │  persona compiler              │     │
                       │      │  coach loop (text frames out)  │     │
                       │      │  LocalFilesystemStore          │     │
                       │      └─────────────────┬──────────────┘     │
                       │                        │                    │
                       │                        ▼                    │
                       │  ┌── evals/runs/{run_id}/sessions/{id}/ ─┐  │
                       │  │ session.json · transcript.jsonl       │  │
                       │  │ intake.json  · persona.json           │  │
                       │  │ phase_timing.json                     │  │
                       │  └────────────────┬──────────────────────┘  │
                       │                   ▼                         │
                       │       [ verifier suite + IntakeFidelity ]   │
                       └────────────────────────────────────────────┘
```

The same `RuntimeHost` box is used in serving (with a Twilio transport instead of in-memory).

## 5. Interfaces

### 5.1 `RuntimeHost`

Single entry point that boots the runtime against a transport.

```python
class RuntimeHost:
    """Boot one rehearse session against a transport. Same object in
    serving and eval. No knowledge of who is on the other end."""

    def __init__(
        self,
        store: LocalFilesystemStore,
        budgets: PhaseBudgets | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None: ...

    async def run(
        self,
        *,
        session_id: str,
        transport: RuntimeTransport,
        consent: ConsentState = ConsentState.GRANTED,
    ) -> SessionArtifacts:
        """Drive one call to completion. Wires PhaseProcessor +
        IntakeProcessor + persona compiler + coach loop onto the bus.
        Returns paths to the persisted artifacts."""
```

`SessionArtifacts` is a Pydantic model carrying paths to `session.json`, `transcript.jsonl`, `intake.json`, `persona.json`, `phase_timing.json`. (Prosody is added when audio is re-introduced in a follow-up spec.)

**Artifact write timing**: artifacts are written incrementally as each phase completes, not only at the end of `run()`. A rollout that crashes mid-INTAKE still produces a partial `session.json`. This makes failures in `failures/{example_id}/` debuggable: the presence or absence of `intake.json` tells you whether `IntakeProcessor` ran to completion.

**Phase timeout**: `RuntimeHost.__init__` accepts `phase_timeout_s: float = 60.0`. If any phase (INTAKE, PRACTICE, FEEDBACK) exceeds this budget, `run()` raises `RuntimePhaseTimeoutError("INTAKE phase exceeded 60s budget")` and cancels both coroutines cleanly.

### 5.2 `RuntimeTransport`

Lift `RuntimeDuplexEndpoint` from `rehearse/eval/transports.py` to `rehearse/transport.py`. **v1 is text-only** — no audio frame extension.

```python
class RuntimeTransport(Protocol):
    async def send(self, kind: Literal["text", "control"],
                   payload: dict) -> None: ...
    async def receive(self, timeout_s: float | None = None) -> TransportEvent: ...
    async def close(self) -> None: ...
```

Concrete implementations in v1:
- `InMemoryDuplexTransport` (already exists, used as-is).
- `TwilioBridgeTransport` — extracted from existing telephony pump code so live calls go through the same `RuntimeHost.run` entry point. (Audio frames stay inside this transport; the runtime sees text events from ASR.)

### 5.3 `CustomerDriver`

```python
class CustomerDriver(Protocol):
    name: str
    version: str

    async def run(
        self,
        *,
        transport: RuntimeTransport,
        runtime_phase: Callable[[], Phase],
    ) -> CustomerDriverResult: ...
```

#### What it is

A small async object that plays the role of the person on the other end of the line. It receives runtime utterances from the transport and sends customer utterances back. It is the **only** component in the eval harness that knows about scenarios — the runtime sees nothing but a text stream.

#### Why we need it

The runtime has one I/O contract: a duplex stream plus session metadata. It cannot tell whether the bytes on the other end come from a real human, a recording, or an LLM. `CustomerDriver` is the abstraction that lets us swap who is on the other end without touching runtime code. Without it, eval and serving diverge: today the eval has its own coach prompt because the customer/runtime boundary is drawn around the wrong thing.

**`CustomerDriverResult`** — the return value of `CustomerDriver.run()`:

```python
@dataclass
class CustomerDriverResult:
    turns_sent: int
    turns_per_phase: dict[str, int]   # e.g. {"INTAKE": 2, "PRACTICE": 4, "FEEDBACK": 1}
    error: str | None = None
    metadata: dict = field(default_factory=dict)
```

Persisted to `evals/runs/{run_id}/sessions/{example_id}/customer_driver.json`.

#### Conversation initiation protocol

**The customer sends the first frame in every phase. The runtime does not send a greeting.** This matches the behavior of `ScriptedCustomerAgent` and the live Twilio path (the caller dials in; the IVR prompts). An `LLMCustomerDriver` implementation that waits for a runtime greeting will deadlock.

#### What it does, turn by turn (v1: `LLMCustomerDriver`)

1. Subscribes to phase transitions on the transport (or polls `runtime_phase()`).
2. **INTAKE phase.** Generates 1–2 short user-style turns describing `scenario.situation`, `scenario.goal`, `scenario.stakes`, `scenario.emotional_state` in plain speech, the way a real caller would over the phone. *Does not* dump the JSON fields verbatim — the runtime's `IntakeProcessor` has to recover those fields from prose, just as it would from a real caller.
3. **PRACTICE phase.** Role-plays as the user trying lines on the counterparty. Reads runtime replies (which are now in-character as the counterparty, driven by the compiled persona) and reacts in character as the user. `scenario.counterparty_style` is *not* in the customer's prompt — that's the runtime's job to enact, and the verifier's job to check it did.
4. **FEEDBACK phase.** Brief, in-character reactions to the coach's debrief.
5. Stops when the runtime emits an end-of-call control event or after a hard turn cap.

#### Inputs

- `scenario` (from the dataset row).
- `transport` (the customer's endpoint of the duplex).
- `runtime_phase()` callback (so it knows when to switch behavior).

#### Outputs

- **Side effect**: text frames sent through the transport, one per turn.
- **Return value**: `CustomerDriverResult` (see schema above). Persisted to `evals/runs/{run_id}/sessions/{example_id}/customer_driver.json` for debugging.

#### Implementations

- `LLMCustomerDriver` — **v1 scope.**
- `ReplayCustomerDriver` — deferred to follow-up spec.
- `LiveTwilioCustomerDriver` — already implicit in `telephony.py`; factored into shape during Phase 1 of the migration.

### 5.4 Environment

Replace `VoiceAgentSandboxEnvironment` with `RuntimeSandboxEnvironment`:

```python
class RuntimeSandboxEnvironment:
    name = "runtime-sandbox"
    version = "v0"

    async def rollout(self, example, run_dir, rng_seed) -> RolloutResult:
        # run_dir = evals/runs/{run_id}/sessions/{example_id}/
        transport = InMemoryDuplexTransport()
        host = RuntimeHost(store=LocalFilesystemStore(run_dir))
        customer = LLMCustomerDriver(scenario=example.payload["scenario"])
        async with transport.lifecycle():
            await asyncio.gather(
                host.run(session_id=..., transport=transport.runtime_endpoint()),
                customer.run(
                    transport=transport.customer_endpoint(),
                    runtime_phase=host.current_phase,
                ),
            )
        return RolloutResult(... artifacts_dir=run_dir ...)
```

No coach prompt anywhere in this file.

**Preflight validation**: `RuntimeSandboxEnvironment.__init__` must call a preflight check before any rollout begins. If `ANTHROPIC_API_KEY` is not set, raise immediately:

```
RuntimeSandboxEnvironment: ANTHROPIC_API_KEY is required (used by IntakeProcessor
and persona compiler). Set it in .env before running --environment runtime-sandbox.
```

Fail at startup, not 3 minutes into a 20-rollout batch.

**Provenance block in `summary.md`**: Each run's `summary.md` must include a provenance section listing which components ran real vs stub:

```
Runtime provenance:
  RuntimeHost          real
  IntakeProcessor      real (ANTHROPIC_API_KEY set)
  PersonaCompiler      real
  CoachVoice           stubbed (Hume TTS deferred in v1)
```

This makes the magical moment visible: you can confirm the real runtime ran without digging into artifact dirs.

#### Why no audio in v1

The v1 reward signal is **content + intake fidelity**, both of which are computable from text alone. Adding audio would require either:
- live Hume EVI in the coach loop (expensive at eval scale), or
- post-hoc TTS through `rehearse/eval/tts_bridge.py:HumeOctaveProvider` (possible but adds a synthesis step and obscures the simpler text loop).

Both paths are explicitly deferred. Re-introducing audio is a strict superset of v1 — the transport, host, and customer driver are designed so that adding `kind="audio"` events later does not change the v1 surface.

### 5.5 Verifier addition

One new scorer in v1 under `rehearse/eval/scorers/`:

- `IntakeFidelityScorer` — reads `intake.json`, compares against optional `expected.intake_*` fields on the dataset row (e.g. `expected.intake_relationship`, `expected.intake_stakes`). Catches `_infer_relationship` regressions. Scored only when the `expected.intake_*` block is present, so existing rows without it produce no signal change.

Existing composite scorers (content, affect, delivery, naturalness) keep their contracts but the audio-derived ones (affect, delivery, naturalness) will degrade because there is no audio in v1. That's expected — they re-normalize when audio is reintroduced. The v1 reward leans on content + intake fidelity.

## 6. What's mocked, what's real (v1)

| Component | Eval (synthetic, text) | Serving |
|---|---|---|
| `RuntimeHost` | real | real |
| `PhaseProcessor` | real | real |
| `IntakeProcessor` | real | real |
| persona compile | real | real |
| coach loop | real (text out) | real (text + audio out) |
| storage | real (`evals/runs/{run_id}/sessions/{id}/`) | real (prod) |
| transport | `InMemoryDuplexTransport` | `TwilioBridgeTransport` |
| customer | `LLMCustomerDriver` | live human |
| Hume TTS | **deferred** (no audio in v1) | live |
| Hume EVI (coach voice) | **deferred** (text frames only in v1) | live |

The runtime core is real in both columns; only the edges (transport and customer) differ. Re-introducing audio is purely additive — it does not change the v1 surface.

**Stub mode definition**: When `HUME_API_KEY` is not set, `runtime-sandbox` runs in stub mode: Hume TTS and Hume EVI are skipped, and coach turns are produced as text frames only. `ANTHROPIC_API_KEY` is **always required** (IntakeProcessor + persona compile use it). There is no all-stub mode that requires no credentials — the eval's signal depends on real LLM calls for intake and persona.

## 7. Migration path (v1 — 5 phases)

### Phase 1 — Extract `RuntimeHost` (no behavior change)

1. Read `telephony.py:mount_twilio_routes` and identify the runtime-wiring block (bus + phase processor + intake processor + persona compiler + coach loop).
2. Move that block into `rehearse/runtime.py` as `RuntimeHost`.
3. `mount_twilio_routes` becomes a thin adapter: build `TwilioBridgeTransport`, call `RuntimeHost.run`.
4. Existing tests under `tests/test_telephony_*.py` and `tests/test_phases.py` continue to pass unchanged.

Verification: `pytest tests/` green; manual smoke of one Twilio session.

### Phase 2 — Lift transport to `rehearse/transport.py` (text-only)

1. Move `rehearse/eval/transports.py` to `rehearse/transport.py`.
2. Update imports across `rehearse/eval/` to point at the new location.
3. No new transport types in v1 — `InMemoryDuplexTransport` and `TwilioBridgeTransport` are sufficient.

Verification: existing `tests/test_telephony_r1.py` green; eval imports still resolve.

### Phase 3 — Phase-aware `LLMCustomerDriver`

1. Create `rehearse/eval/customers/llm_customer.py`.
2. Three system prompts keyed by current `Phase`. The driver polls `runtime_phase()` and switches prompt at transitions.
3. INTAKE prompt: monologue the situation in plain speech. PRACTICE prompt: rehearse with the runtime's counterparty in character. FEEDBACK prompt: react.
4. Persist `customer_driver.json` with turn counts and errors per phase to the run dir.

Verification: unit test that a fixed `scenario` produces an `IntakeRecord` whose `situation`/`relationship`/`stakes` match expectation within tolerance.

### Phase 4 — `runtime-sandbox` env + `IntakeFidelityScorer`

1. New `rehearse/eval/environments/runtime_sandbox.py` per §5.4.
2. Update `voice_rollout_judges.py` eval to point at `runtime-sandbox` (add to `supported_environments`, set `preferred_environment`).
3. Delete `_COACH_SYSTEM_PROMPT` and `LLMSandboxAgent(role="coach")` paths.
4. Keep `voice-agent-sandbox` as a full deprecated shim (not yet removed) that prints a warning to stderr and delegates to `runtime-sandbox`. Both environments are registered and runnable.
5. Add `rehearse/eval/scorers/intake_fidelity.py` and wire it into `voice-rollout-judges` `scoring_plan()` with weight 0.05 (existing weights re-normalize).
6. Add optional `expected.intake_*` fields to seed scenarios in `evals/datasets/voice-rollout-judges/v1/scenarios.jsonl` (purely additive).

Verification: full `voice-rollout-judges` run on existing scenarios completes with **both** environments registered; spot-check that `intake.json` and `persona.json` are populated under `evals/runs/{run_id}/sessions/{example_id}/`; deliberate regression in `_infer_relationship` lowers `intake_fidelity` on at least one scenario; `rehearse-eval run --environment voice-agent-sandbox` prints deprecation warning.

### Phase 5 — Remove `voice-agent-sandbox` shim

Remove `voice-agent-sandbox` from the environment registry after at least one full week of `runtime-sandbox` running green on all scenarios.

Verification: `rehearse-eval list-environments` no longer shows `voice-agent-sandbox`. CI passes.

## 8. Repo additions

```
rehearse/
  runtime.py                          [new] RuntimeHost
  transport.py                        [moved from eval/transports.py]
  eval/
    customers/
      __init__.py                     [new]
      llm_customer.py                 [new]
    environments/
      runtime_sandbox.py              [new]
      voice_agent_sandbox.py          [deprecated shim, removed in Phase 5]
    scorers/
      intake_fidelity.py              [new]
docs/specs/v2026-05-07-runtime-eval-alignment.md   [this file]
tests/
  test_runtime_host.py                [new]
  test_llm_customer_phase_aware.py    [new]
  test_runtime_sandbox_rollout.py     [new]
  test_intake_fidelity_scorer.py      [new]
Makefile                              [add eval-voice-rollout, eval-voice-rollout-live targets]
rehearse/eval/README.md               [update: add CustomerDriver to "Adding Pieces"; update
                                       "What's There Today" table to reference runtime-sandbox]
.github/workflows/ (or CI config)     [add schema-diff check: diff model_json_schema() of
                                       Session/IntakeRecord/PersonaRecord between serving and
                                       eval artifact paths on every PR]
```

**Makefile targets to add:**

```makefile
eval-voice-rollout: ## run voice-rollout-judges with runtime-sandbox, stub judges (no Hume key needed)
    uv run rehearse-eval run --eval voice-rollout-judges --limit 3

eval-voice-rollout-live: ## run voice-rollout-judges with real Gemini judges (needs GOOGLE_API_KEY + ANTHROPIC_API_KEY)
    REHEARSE_AUDIO_JUDGE=live uv run rehearse-eval run --eval voice-rollout-judges --limit 3
```

**scorer error message requirement**: Any scorer that reads an artifact file must raise a `ScorerArtifactError` (not a raw `FileNotFoundError`) with message format:
```
{ScorerName}: {artifact}.json not found in {run_dir}.
This means RuntimeHost did not complete the {phase} phase.
Check failures/{example_id}/ for the error that stopped the rollout.
```

## 9. Acceptance criteria

A change is accepted when all of the following are true:

1. `voice-rollout-judges` runs the real `RuntimeHost` for every example. `grep -R "system.*coach\|COACH_SYSTEM" rehearse/eval/` returns no results.
2. Each rollout produces `session.json`, `transcript.jsonl`, `intake.json`, `persona.json`, `phase_timing.json` under `evals/runs/{run_id}/sessions/{example_id}/`.
3. A serving call and an eval rollout of the same scenario produce artifact sets with identical Pydantic schemas (diff of `Session.model_json_schema()` etc. is empty).
4. A deliberate regression in `IntakeProcessor._infer_relationship` lowers `intake_fidelity` on at least one scenario in the seed set.
5. `make eval-voice-rollout` (stub mode, `HUME_API_KEY` not set, `ANTHROPIC_API_KEY` set) completes in under 5 minutes for `--limit 1`.
6. `rehearse-eval show <run_id>` output includes a provenance block listing which components ran real vs stub.
7. CI schema diff check: a GitHub Actions job diffs `model_json_schema()` for `Session`, `IntakeRecord`, and `PersonaRecord` between a serving-path artifact and an eval-path artifact. Job fails if the schemas diverge.

## 10. Open questions

1. **Verifier weight re-normalization.** Adding `intake_fidelity` and removing audio judges from the v1 reward shifts the composite. Recommend: ship `intake_fidelity` at weight 0.05; keep audio-judge weights in the rubric but flag scores as "audio-deferred" so dashboards can mask them until audio is re-introduced.

## 11. Deferred to follow-up specs

These were in earlier drafts of this spec but are explicitly deferred so v1 is shippable in a week, not a month. Each is purely additive on top of v1:

- **Audio in eval rollouts.** Either post-hoc TTS via the existing `rehearse/eval/tts_bridge.py:HumeOctaveProvider` (cheap), or live Hume EVI in the coach loop (expensive). Re-enables affect/delivery/naturalness audio judges in the composite.
- **`ReplayCustomerDriver` and replay manifests.** Streams logged caller-side audio/text from production sessions through the same `RuntimeHost`, so we can A/B runtime versions on real call traces. Requires a replay manifest format and audio-frame transport.
- **`PersonaGroundingScorer`.** LLM judge: does `persona.personality_prompt` reflect `scenario.counterparty_style`?
- **`PhaseTimingScorer`.** Reads `phase_timing.json`. Penalizes budget-driven (timeout) transitions vs cue-driven ones in INTAKE; flags PRACTICE phases that never transitioned.
- **`LiveTwilioCustomerDriver` formalization.** Today the live customer is implicit in `telephony.py`; a follow-up spec can make it a first-class `CustomerDriver` so the symmetry with eval is total.

## 12. Out of scope (will not be specced here)

- Curriculum / scenario generation (covered by `scripts/generate_scenarios.py` and downstream specs).
- Reward-model training, RL update steps, replay buffers.
- Persona swapping mid-call.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Voice | `/plan-devex-review` | Independent 2nd opinion | 1 | issues_found | 6 findings (extraction risk, stub mode, Phase 4 sequencing, schema identity) — 3 accepted, 3 deferred |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 0 | — | — |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 1 | CLEAR | score: 4/10 → 8/10, TTHW: unknown → 5min |

- **UNRESOLVED:** 0 decisions
- **VERDICT:** DX CLEARED — eng review required before shipping
