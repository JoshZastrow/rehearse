# Runtime Mirror Eval — Streaming Scores to a Dashboard

**File:** `docs/specs/v2026-05-08-runtime-mirror-eval-streaming.md`
**Status:** Draft
**Supersedes:** parts of `v2026-05-07-runtime-eval-alignment.md` (audio, dashboard, queue)
**Extends:** Phase 4b/4c/4-stretch from `v2026-05-07`

---

## 0. One-line summary

Run N parallel sandboxed rollouts of the **same runtime** that ships to production, capture the **full artifact bundle including audio**, score each rollout, stream the scored results to a queue, and render a live aggregate readout in the terminal that updates as scores land.

Each change to `RuntimeHost` and its dependencies (`PhaseProcessor`, `IntakeProcessor`, `PersonaCompiler`, writers) reflects in eval automatically because the eval boots the same object the phone path boots. The eval substitutes only the **edges**: coach voice transport (Hume EVI ↔ Anthropic), human counterparty (synthetic caller), and live audio capture (post-hoc TTS in the cheap tier; live Hume in the release-gate tier).

---

## 1. Context

### Where we are today

`v2026-05-07-runtime-eval-alignment.md` landed a `RuntimeSandboxEnvironment` that boots the real `RuntimeHost` against an `InMemoryDuplexTransport` and an `LLMCustomerDriver`. The kernel is correctly shared with telephony.

What still doesn't work:
- **No audio in eval rollouts.** `RuntimeSandboxEnvironment.rollout` returns immediately after the runtime/customer gather; no WAV files, no `timing.jsonl`. The audio judges (`StrictAffectPerceptionJudgeScorer`, `StrictDeliveryJudgeScorer`) and timing judges (`NaturalnessScorer`) all degrade to `audio_missing` / zero-signal. That's 0.42 of the composite weight producing no useful signal.
- **No streaming results.** `runner.py` collects all scores in-memory, writes `summary.md` once at the end. Nothing tails per-rollout progress; nothing recomputes aggregates as rollouts land. Run a 100-row eval and you stare at silence for 20 minutes.
- **No mirror enforcement.** Nothing detects if `RuntimeHost`'s artifact contract drifts between serving and eval. A field added to `IntakeRecord` in serving could silently break eval scoring.

### What we want

Three concrete behaviors, layered:

1. **Full artifact mirror.** A `runtime-sandbox` rollout produces the same artifact bundle a live phone call produces — `session.json`, `transcript.jsonl`, `intake.json`, `persona.json`, `phase_timing.json`, `customer_driver.json`, `audio/{user,coach}/turn_<N>.wav`, `timing.jsonl`. Audio comes from post-hoc TTS in v2 (kept cheap), with an opt-in `tier=mirror-full` mode that uses real Hume EVI for full Hume-integration fidelity.

2. **Streaming scoring.** As each rollout completes and gets scored, its `RubricScore` rows append to a per-run `scores.jsonl` queue file. A separate `rehearse-eval watch <run_dir>` process tails the file, recomputes the weighted aggregate on each new line, and re-renders a terminal table. The harness writes; the dashboard reads. Decoupled.

3. **Mirror enforcement.** A CI job boots `RuntimeHost` once via the serving path and once via `runtime-sandbox`, diffs the JSON schemas of `Session`, `IntakeRecord`, and `CounterpartyPersona`, and fails on any divergence.

---

## 2. Goals

1. Sandboxed rollouts produce a complete artifact bundle that includes per-turn audio WAVs and `timing.jsonl`.
2. Audio judges (`affect_perception`, `delivery_quality`) and timing judges (`naturalness.*`) produce real, non-degraded scores when `HUME_API_KEY` is set.
3. `make eval-voice-rollout` runs N rollouts in parallel and returns a complete `summary.md` with all dimensions populated.
4. A separate `rehearse-eval watch <run_dir>` process renders a live aggregate that updates as each rollout's scores land, without polling or coupling to the runner.
5. CI catches any drift between the production-runtime artifact contract and what `runtime-sandbox` produces.
6. The substitution boundary is documented and enforced: only `Transport`, `CoachVoiceAdapter`, and the human counterparty (now `CustomerDriver`) are eval-replaceable. Everything else inside `RuntimeHost` is identical between paths.

## 3. Non-goals

- **No web dashboard.** Terminal-only renderer in v2. The streaming queue is the seam for a future web UI; we don't build the UI now.
- **No distributed execution.** Single-host parallelism via `asyncio.Semaphore` is enough. No Ray, no Celery, no remote workers.
- **No PR comments / GitHub Action surfacing.** The streaming queue can feed those later; not in v2.
- **No replay against production audio.** That's `production-replay`'s job; orthogonal.

### What the cheap tier deliberately doesn't catch

The default tier (`text-plus-tts`) is the only tier we run in normal CI. It is **not a full mirror of production**. We're choosing this trade-off knowingly. The default tier will miss:

- **Hume EVI integration regressions.** A change that breaks our Hume client, message format, or session lifecycle will pass the cheap tier and ship.
- **Live ASR fidelity.** Production audio is transcribed by Hume EVI's ASR; the cheap tier uses LLM-generated text directly. Mistranscription patterns, word-error-rate effects, and ASR latency aren't visible.
- **Interruption / barge-in dynamics.** The cheap tier is strict turn-taking. The production runtime handles the user interrupting the coach mid-sentence; we don't exercise that path here.
- **Live prosody coupling.** In production the coach can react to user prosody mid-call. Post-hoc TTS produces a voice for each turn after the fact; nothing in the eval loop reacts to it.
- **Real Hume voice character.** The coach's actual production voice is the one Hume's TTS produces given the EVI persona config. Post-hoc Octave TTS with our description string is in the same family but not identical.

The release-gate tier (`mirror-full`, §5.5) covers all of these. It runs before each production deploy; the cheap tier is not allowed to be the only signal before a release.

---

## 4. Architecture

### 4.1 The substitution boundary

```
                    ┌─────────────────────────────────────────┐
                    │           Production runtime kernel     │  ← identical in
                    │  RuntimeHost                            │    serving + eval
                    │   ├── PhaseProcessor                    │    (enforced by
                    │   ├── IntakeProcessor                   │    schema-diff CI)
                    │   ├── PersonaCompiler                   │
                    │   ├── TranscriptWriter                  │
                    │   ├── TimingWriter                      │
                    │   ├── ProsodyWriter                     │
                    │   └── AudioRecorder                     │
                    └─────────┬──────────────────┬────────────┘
                              │                  │
                  CoachVoice  │                  │  Transport
                  Adapter     │                  │
                              │                  │
                ┌─────────────┴────┐    ┌────────┴───────────┐
   serving →   │ HumeCoachAdapter │    │ TwilioBridgeTransp.│  ← real edges
                └──────────────────┘    └────────────────────┘
                              │                  │
                ┌─────────────┴────┐    ┌────────┴───────────┐
   eval (default) │ TextOnlyCoach- │    │ InMemoryDuplex-    │  ← cheap edges
                  │ Adapter        │    │ Transport          │     (post-hoc TTS)
                ┌─┴────────────────┐    └────────────────────┘
   eval (mirror-full)│ HumeCoach- │    │ HumeAudioBridge-   │  ← full-fidelity
                     │ Adapter    │    │ Transport          │     edges (real Hume)
                     └────────────┘    └────────────────────┘
                                              │
                                       ┌──────┴──────────┐
                                       │ CustomerDriver  │  ← LLM-driven user
                                       │ (LLMCustomer)   │     (always synthetic
                                       │ (AudioCustomer) │      in eval)
                                       └─────────────────┘
```

**Rule:** anything inside the kernel box is the same import path, same code, same tests in serving and eval. New runtime features land once and propagate. Anything outside the box is an eval-replaceable edge.

### 4.2 Streaming pipeline

```
  runner.py ─┬──> [rollout 1] ──> [scorer 1] ──┐
             │                                  ▼
             ├──> [rollout 2] ──> [scorer 2] ──> scores.jsonl  ←── persistent queue
             │                                  ▲                    (run_dir/scores.jsonl)
             └──> [rollout N] ──> [scorer N] ──┘
                                                 │
                                                 │  inotify / poll-tail
                                                 ▼
                              rehearse-eval watch <run_dir>
                                                 │
                                                 ▼
                                       [terminal renderer]
                                       re-aggregates each tick
```

The runner doesn't know about the watcher. The watcher doesn't know about the runner. Both share the file. JSONL is the protocol.

---

## 5. Functional requirements

### FR-1 — Post-hoc TTS in `RuntimeSandboxEnvironment`

After the runtime/customer gather completes, synthesize per-turn WAVs from `transcript.jsonl`:
- User turns: WAV at `audio/user/turn_<N>.wav`. Hume description = `scenario.emotional_state`.
- Coach turns: WAV at `audio/coach/turn_<N>.wav`. Hume description = `_DEFAULT_COACH_DESCRIPTION = "warm, steady, present"`, override via `example.payload["coach_description"]`.
- Compute `timing.jsonl` from real WAV durations using the `_timing_from_frames` pattern.
- Surface `tts_provider` in `RolloutResult.payload`.

When `HUME_API_KEY` unset: silent WAVs sized by a duration heuristic; audio judges still run, just produce lower scores. Add an `audio_stub` flag to the run's metadata so dashboards can mask noise.

### FR-2 — Streaming scores via `scores.jsonl`

Each `RubricScore` produced by the scoring plan appends a JSON line to `<run_dir>/scores.jsonl` **immediately** when the scorer returns it, not when the whole run finishes. Append is atomic (one write per line, line-buffered).

Schema:
```json
{
  "run_id": "...",
  "example_id": "...",
  "dimension": "content_quality",
  "value": 0.73,
  "scorer": "llm_judge",
  "rationale": "...",
  "modality": "text",
  "judge_prompt_version": "strict-content-v1",
  "flags": [],
  "emitted_at": "2026-05-08T12:34:56Z"
}
```

`emitted_at` is a new field added at write time; it's not on `RubricScore` itself.

### FR-3 — `rehearse-eval watch <run_dir>` command

A new CLI subcommand that tails `<run_dir>/scores.jsonl` and re-renders a terminal aggregate on each new line.

Behavior:
- On startup, read the existing `scores.jsonl` to build the current aggregate state.
- Then tail (block on new lines).
- On each new score, recompute weighted aggregates using `AggregateScorer._weights` from the eval's scoring plan.
- Re-render: clear screen, print a table of dimensions × mean × n × confidence.
- Exit on EOF if the run is finished (a sentinel line `{"event": "done"}` is appended by the runner) or on Ctrl-C.

Renderer is plain ANSI / pure stdout. No curses, no rich tables (avoid TTY assumption breakage in CI). One `\033[2J\033[H` clear, plain markdown table, refresh.

### FR-4 — Mirror enforcement via schema-diff CI

`.github/workflows/schema-diff.yml` runs on every push:
1. Boot `RuntimeHost` via a serving-path fixture (telephony test harness, scripted Twilio events).
2. Boot `RuntimeHost` via `runtime-sandbox` rollout against a single dataset row.
3. Diff the JSON schemas of `Session`, `IntakeRecord`, and `CounterpartyPersona` produced by both paths.
4. Fail on any divergence. Print a diff.

This is the structural assertion that supports goal 6: "the kernel is the same."

### FR-5 — `tier=mirror-full` release-gate mode

A second mode of `RuntimeSandboxEnvironment` that uses the real `HumeCoachAdapter` and a real audio transport, not text + post-hoc TTS. Selected via `--environment-config tier=mirror-full` or `REHEARSE_RUNTIME_TIER=mirror-full`.

**This tier is the release gate.** A production deploy is blocked unless mirror-full has run green within the last 24 hours on the deploying commit. Triggers:
- Manual: `make eval-mirror-full` before opening a release PR.
- Scheduled: nightly cron on `main` for trend detection.
- Pre-merge (optional, post-v2): a label like `needs-mirror-full` triggers it on PRs that touch `rehearse/runtime.py`, `rehearse/telephony.py`, `rehearse/runtime/hume_*`, or any file flagged in the schema-diff workflow.

In `tier=mirror-full`:
- `SyntheticCaller` (the renamed `LLMCustomerDriver`) produces audio frames inline during the rollout — TTS the same way coach turns are synthesized, but for user turns at send-time, not post-hoc.
- The transport carries audio. We add a `HumeAudioBridge` (audio-capable two-way channel) that owns one Hume EVI session per rollout.
- `HumeCoachAdapter` is used unchanged (same code as production).
- `AudioRecorder` and `ProsodyWriter` run live (no post-hoc step), producing `audio/` and `prosody.jsonl` exactly as production does.

This is the strict mirror tier. Default tier (`tier=text-plus-tts`) remains the cheap, fast loop. **Default tier is not allowed to be the only signal before a release** — see Non-goals §3 for what it can't catch.

### FR-6 — Composite scorer wires the queue

`CompositeScorer` (and `AggregateScorer`) emit each child dimension's `RubricScore` to the queue as the child returns. Today the composite collects everything and emits the aggregate at the end; we change it so individual dimensions hit the queue first, then the aggregate hits last with a known dimension name (`weighted_reward`).

This means the watcher sees `content_quality` for example_id `vrj-s01` land before `affect_perception` for the same example, and re-aggregates as each lands.

---

## 6. Non-functional requirements

### NFR-1 — Performance

- Default tier rollout completes in ≤ 60s for a 6-turn coaching session (`make eval-voice-rollout --limit 1`).
- 10-row default-tier batch with concurrency=4 completes in ≤ 5 min wall clock.
- `mirror-full` tier rollout: ≤ 180s per rollout (Hume real-time).
- Watch renderer refresh latency: ≤ 200ms from `scores.jsonl` append to re-render.

### NFR-2 — Cost

- Default tier per rollout: ≤ $0.10 (Anthropic Sonnet for coach + customer + content judge; Hume Octave for ~12 short TTS calls; Gemini for two audio judges).
- `mirror-full` per rollout: ≤ $0.50 (real Hume EVI session).
- Stub mode (no `HUME_API_KEY`): $0.

### NFR-3 — Reproducibility

- Same input + same RNG seed must produce same `transcript.jsonl` modulo LLM nondeterminism. We don't fight LLM nondeterminism in v2; we accept it and use repetitions for stability (Spec 8 already exists).
- `customer_driver.json` records the seed and model used, so runs are reproducible to the limits of the model's determinism settings.

### NFR-4 — Reliability

- A crashed rollout produces partial artifacts and a clearly-flagged `RolloutResult.status=error`. Other rollouts in the same batch continue.
- A scorer that raises produces a flagged zero-score (existing `_zero` pattern); other scorers run.
- `scores.jsonl` writes are atomic per line (if a writer crashes mid-line, the next reader skips the malformed line and continues).

### NFR-5 — Observability

- Every rollout writes a `provenance.json` to its run dir documenting: which `CoachVoiceAdapter`, which `Transport`, which TTS provider, which API keys were available (boolean only — never the keys themselves), the runtime tier.
- The provenance block already in `summary.md` reflects this.
- `runtime_sandbox.py` does not emit a coach prompt anywhere on disk (the existing grep gate stays green).

### NFR-6 — Boundary discipline

- No imports from `rehearse.eval.*` inside `rehearse/runtime.py`, `rehearse/intake.py`, `rehearse/phases.py`, or any file in `rehearse/writers/`. The runtime kernel does not know it's being evaluated.
- No imports from `rehearse.telephony` inside the eval harness. Twilio is a serving-only edge.

---

## 7. Inputs / outputs / data sources

### Inputs

| Source | Path | Purpose |
|---|---|---|
| Dataset rows | `evals/datasets/voice-rollout-judges/v1/scenarios.jsonl` | One row per rollout: `scenario` (situation, goal, stakes, emotional_state, counterparty_role/style), `expected.intake_*`, `rubric_weights` |
| Anthropic API | `ANTHROPIC_API_KEY` | Coach LLM (`TextOnlyCoachAdapter`) and customer LLM (`LLMCustomerDriver`) and content judge (`StrictContentJudgeScorer`) |
| Hume Octave TTS | `HUME_API_KEY` | Per-turn audio synthesis (post-hoc in default tier; live in `mirror-full`) |
| Gemini multimodal | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Audio judges (`StrictAffectPerceptionJudgeScorer`, `StrictDeliveryJudgeScorer`) |

### Outputs (per rollout, in `<run_dir>/<example_id>/`)

| Artifact | Producer | Read by |
|---|---|---|
| `session.json` | `RuntimeHost` (kernel) | All scorers, persistence |
| `transcript.jsonl` | `TranscriptWriter` (kernel) | `StrictContentJudgeScorer`, post-hoc TTS, audio judges |
| `intake.json` | `IntakeProcessor` (kernel) | `IntakeFidelityScorer` |
| `persona.json` | `PersonaCompiler` (kernel) | Operator review (no scorer reads it yet) |
| `phase_timing.json` | `TimingWriter` (kernel) | `NaturalnessScorer` (phase dwell), summary rendering |
| `customer_driver.json` | `LLMCustomerDriver` (eval edge) | Operator review, debugging |
| `audio/user/turn_<N>.wav` | post-hoc TTS (default) / `AudioRecorder` (mirror-full) | `StrictAffectPerceptionJudgeScorer`, `StrictDeliveryJudgeScorer` |
| `audio/coach/turn_<N>.wav` | post-hoc TTS (default) / `AudioRecorder` (mirror-full) | Same |
| `timing.jsonl` | `_timing_from_frames` (default) / `TimingWriter` (mirror-full) | `NaturalnessScorer` |
| `prosody.jsonl` | none (default) / `ProsodyWriter` (mirror-full) | `NaturalnessScorer` (silence after affect) |
| `provenance.json` | `RuntimeSandboxEnvironment` | Summary rendering, audit |

### Outputs (per run, in `<run_dir>/`)

| Artifact | Producer | Read by |
|---|---|---|
| `scores.jsonl` | each scorer (live) | `rehearse-eval watch`, summary rendering |
| `summary.md` | `runner._render_summary` (end of run) | Human review |
| `run.json` | `runner.execute_run` (end of run) | Run registry |
| `aggregates.json` | `AggregateScorer` (end of run) | Comparison across runs |

---

## 8. Interfaces

### 8.1 `CoachVoiceAdapter` (existing, no change)

```python
class CoachVoiceAdapter(Protocol):
    async def respond(self, user_text: str, session_id: str) -> str: ...
```

Both `TextOnlyCoachAdapter` and `HumeCoachAdapter` implement this. `mirror-full` tier uses the latter unchanged.

### 8.2 `Transport` (existing, may need audio bridge addition)

`RuntimeDuplexEndpoint` already supports `kind="audio"` events with `data: bytes | None`. Default tier uses `InMemoryDuplexTransport` and only sends `kind="text"` and `kind="control"`. `mirror-full` adds `HumeAudioBridgeTransport` (new, in `rehearse/transport.py` or `rehearse/eval/transports/`).

### 8.3 `CustomerDriver` (existing for text, extend for audio)

```python
class CustomerDriver(Protocol):
    name: str
    version: str
    async def run(
        self,
        *,
        transport: RuntimeDuplexEndpoint,
        runtime_phase: Callable[[], Phase],
    ) -> CustomerDriverResult: ...
```

Default tier uses `LLMCustomerDriver` (text). Mirror-full tier adds `AudioLLMCustomerDriver` that wraps the text driver and TTSes user turns inline before sending them.

### 8.4 `Scorer` queue interface (new)

```python
class StreamingScorer(Protocol):
    """Optional addition: scorers can publish scores incrementally."""
    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
        publish: Callable[[RubricScore], None] | None = None,
    ) -> list[RubricScore]: ...
```

The runner injects `publish` so scorers that produce multiple dimensions (`CompositeScorer`) can stream each as it's ready. Scorers without a `publish` parameter still work — the runner publishes their returned list at end.

### 8.5 Watch protocol (file-based)

`<run_dir>/scores.jsonl`: append-only JSONL. Format defined in FR-2.
`<run_dir>/scores.jsonl.done`: empty sentinel file written when the run finishes. Watch exits on its appearance.

No need for a richer protocol than file + sentinel. If we ever go networked, swap the implementation behind the same contract.

---

## 9. Implementation gaps (audit by file)

| File | Gap | Effort |
|---|---|---|
| `rehearse/eval/environments/runtime_sandbox.py` | No post-hoc TTS, no `timing.jsonl`, no `provenance.json` | 1d |
| `rehearse/eval/environments/_audio.py` | Doesn't exist; need to extract `_synthesize`, `_timing_from_frames`, `_silent_wav` from `live_rollout_audio.py` | 0.5d |
| `rehearse/eval/runner.py` | No streaming publish; `RubricScore` writes happen at end only; no `scores.jsonl` | 0.5d |
| `rehearse/eval/scorers/composite.py` | Doesn't accept a `publish` callback | 0.25d |
| `rehearse/eval/cli.py` | No `watch` subcommand | 0.5d |
| `rehearse/eval/watch.py` | Doesn't exist; new file (file tail + re-render) | 0.75d |
| `rehearse/runtime.py` | `HumeCoachAdapter` skeleton may be incomplete; verify against `telephony.py` | 0.5d |
| `rehearse/transport.py` | `HumeAudioBridgeTransport` doesn't exist (mirror-full tier only) | 1d (deferred) |
| `rehearse/eval/customers/audio_llm_customer.py` | Doesn't exist (mirror-full tier only) | 0.75d (deferred) |
| `.github/workflows/schema-diff.yml` | Doesn't exist | 0.5d |
| `tests/test_runtime_sandbox_audio.py` | Doesn't exist | 0.25d |
| `tests/test_eval_streaming.py` | Doesn't exist | 0.5d |
| `tests/test_watch_renderer.py` | Doesn't exist | 0.25d |
| `docs/specs/v2026-05-07-runtime-eval-alignment.md` | Audio-deferred language still present (§0/§2/§5.2/§5.4/§5.5/§6/§7/§11) | 0.25d |
| `Makefile` | `eval-voice-rollout-watch` target missing | 0.1d |

**Default tier (Phase B + C):** ~5.5 dev days sequential, ~3 days parallelized.
**Renaming pass (Phase E):** +0.75 dev days. Lands inside v2.
**Mirror-full release gate (Phase D):** +2 dev days. Lands in v2.1 (within 2 weeks of v2 merge — required before any deploy past v2).

---

## 10. Implementation plan

### Phase A — Spec edit (prerequisite, ~0.25d, owner only)

Update `v2026-05-07-runtime-eval-alignment.md` per the divergence list. Owner edit; cannot proceed without this because reviewers reading the spec during T-A.2 review would be misled.

### Phase B — Default-tier audio + streaming (the core of v2)

| Task | File(s) | Spec ref |
|---|---|---|
| B.1 Extract audio helpers | `rehearse/eval/environments/_audio.py` | FR-1 |
| B.2 Wire post-hoc TTS into `runtime-sandbox` | `rehearse/eval/environments/runtime_sandbox.py` | FR-1 |
| B.3 Add `provenance.json` writer | same | NFR-5 |
| B.4 Add `publish` to scorer interface | `rehearse/eval/protocols.py`, `rehearse/eval/scorers/composite.py` | FR-6, 8.4 |
| B.5 Wire streaming in runner | `rehearse/eval/runner.py` | FR-2 |
| B.6 Add `watch` subcommand | `rehearse/eval/cli.py`, `rehearse/eval/watch.py` | FR-3 |
| B.7 Tests | `tests/test_runtime_sandbox_audio.py`, `tests/test_eval_streaming.py`, `tests/test_watch_renderer.py` | — |
| B.8 Makefile + Docs | `Makefile`, `rehearse/eval/README.md` | — |

**Phase B acceptance gate:**
- `make eval-voice-rollout` produces a complete artifact bundle including audio.
- Audio judges produce non-degraded scores when `HUME_API_KEY` is set.
- `make eval-voice-rollout-watch` (new target) runs the rollout in one shell and `rehearse-eval watch <run_dir>` in another; the watcher updates as scores land.
- All tests green.

### Phase C — Mirror enforcement (parallel with B, ~0.75d)

| Task | File(s) | Spec ref |
|---|---|---|
| C.1 Schema-diff workflow | `.github/workflows/schema-diff.yml` | FR-4 |
| C.2 Verify `HumeCoachAdapter` parity with telephony | `rehearse/runtime.py`, `rehearse/telephony.py` | NFR-6 |

**Phase C acceptance gate:**
- A field added to `IntakeRecord` and pushed to a branch causes the CI job to fail with a readable diff.
- Reverting the field returns CI to green.

### Phase D — Mirror-full tier (release-gate, lands after Phase B+C+E green)

| Task | File(s) | Spec ref |
|---|---|---|
| D.1 `HumeAudioBridge` (audio-capable two-way channel) | `rehearse/transport.py` | FR-5 |
| D.2 `SyntheticAudioCaller` (audio-producing customer) | `rehearse/eval/customers/audio_caller.py` | FR-5 |
| D.3 Tier selection in `runtime-sandbox` | `rehearse/eval/environments/runtime_sandbox.py` | FR-5 |
| D.4 `make eval-mirror-full` target + nightly cron workflow | `Makefile`, `.github/workflows/mirror-full-nightly.yml` | FR-5 |
| D.5 Release-gate doc + checklist | `docs/release-checklist.md` | FR-5 |

**Phase D acceptance gate:**
- `make eval-mirror-full` completes one rollout end-to-end with a real Hume EVI session and produces `prosody.jsonl` populated by the live `ProsodyWriter`.
- A documented release procedure requires a green mirror-full run on the deploy commit within 24h.

### Phase E — Renaming pass (in scope for v2; lands after Phase B)

The codebase carries telecom-era names that aren't grounded in product language. We're building AI products for people, not phone switches. Rename in one pass with shims for one release cycle, then remove the shims.

| Old name | New name | Reason |
|---|---|---|
| `InMemoryDuplexTransport` | `InMemoryTwoWayChannel` | "duplex" is telecom jargon; "two-way channel" says what it is |
| `RuntimeDuplexEndpoint` | `TwoWayChannel` | same |
| `RuntimeTransport` (alias) | `Channel` (alias) | same |
| `LLMCustomerDriver` | `SyntheticCaller` | "driver" is test-automation jargon; we're simulating a human caller |
| `CustomerDriver` (Protocol) | `Caller` (Protocol) | same |
| `CustomerDriverResult` | `CallerResult` | same |
| `TwilioBridgeTransport` | `TwilioPhoneBridge` | "bridge" is grounded; drop "transport" |
| `HumeAudioBridgeTransport` (planned) | `HumeAudioBridge` | same |

Implementation:
1. Rename the class/protocol/alias in its definition file.
2. Add a re-export with the old name (`RuntimeDuplexEndpoint = TwoWayChannel`) and a `DeprecationWarning` on import.
3. Update all callers in the same PR.
4. Update tests, type hints, and docs.
5. Old aliases removed in the next eval-system PR (one release cycle later).

**Phase E acceptance gate:**
- All production and eval call sites use the new names.
- Old names still importable but warn.
- Docs and `README.md` use new names exclusively.

---

## 11. Test plan

### Unit (in scope for v2)

- **`tests/test_runtime_sandbox_audio.py`** — runtime-sandbox produces audio WAVs (real or stub) and `timing.jsonl` matching turn count; `coach_description` override applied; provenance.json content correct.
- **`tests/test_eval_streaming.py`** — `scores.jsonl` is written line-by-line as scores land; sentinel file created on completion; malformed line tolerated by reader.
- **`tests/test_watch_renderer.py`** — given a synthetic `scores.jsonl` stream, the renderer produces the expected aggregate table after each new line.
- **`tests/test_runtime_sandbox_provenance.py`** — provenance.json fields are correct under each combination of (`HUME_API_KEY` set/unset, `tier=text-plus-tts`).

### Integration (single-host)

- `make eval-voice-rollout --limit 3` produces three runs, each with full audio bundle.
- `rehearse-eval watch <run_dir>` against a 10-row run prints incremental aggregates; no duplicate or missing scores.

### Regression (the existing safety net)

- All 377 existing tests stay green.
- The `grep -R "system.*coach\|COACH_SYSTEM" rehearse/eval/environments/runtime_sandbox.py` returns no hits (no static coach prompt creep).

---

## 12. End-to-end flow (after Phase B + C land)

```
$ export ANTHROPIC_API_KEY=... HUME_API_KEY=... GEMINI_API_KEY=...
$ make eval-voice-rollout              # in shell 1
  → spawns 4 parallel rollouts
  → each rollout: RuntimeHost runs, customer drives, post-hoc TTS, scoring
  → each scorer emits to scores.jsonl as it returns
  → final summary.md written

$ rehearse-eval watch <run_dir>        # in shell 2 (started anytime)
  ┌──────────────────────────────────────┐
  │ Eval run abc123 — 7/10 examples done │
  │ ┌─────────────────────┬──────┬───┐  │
  │ │ Dimension           │ Mean │ N │  │
  │ ├─────────────────────┼──────┼───┤  │
  │ │ content_quality     │ 0.73 │ 7 │  │
  │ │ affect_perception   │ 0.81 │ 7 │  │
  │ │ delivery_quality    │ 0.69 │ 7 │  │
  │ │ naturalness.*       │ 0.85 │ 7 │  │
  │ │ intake_fidelity     │ 0.92 │ 7 │  │
  │ │ weighted_reward     │ 0.77 │ 7 │  │
  │ └─────────────────────┴──────┴───┘  │
  └──────────────────────────────────────┘
  (refreshes ~once per scored example)
```

When the run finishes, `scores.jsonl.done` is created and the watcher exits cleanly.

---

## 13. Decisions

1. **Watch renderer:** `rich`. Pretty terminal table, manageable dep, already in the Python ecosystem.
2. **Composite recompute:** wait-for-completion per example. Show `pending` until every child dimension for an example has landed; only then add it to the aggregate.
3. **Crash-resume:** out of scope for v2. If `runner.py` crashes mid-run, the run is lost. Flag for v3.
4. **Mirror-full triggering:** manual `make eval-mirror-full` is the canonical trigger; nightly cron on `main` is secondary. **Required before every production deploy.**
5. **Naming:** drop telecom and test-automation jargon (`Duplex`, `Driver`). New names land in Phase E. Old names re-exported with `DeprecationWarning` for one release cycle.

## 14. Still uncertain

- **`HumeCoachAdapter` parity with `telephony.py`.** I haven't audited it line-by-line. Phase C.2 does; if it's a stub, costs +0.5d to flesh out.
- **LLM nondeterminism budget.** With Anthropic-as-coach and Anthropic-as-customer and Anthropic-as-judge, the noise floor on any single rollout's `weighted_reward` is real. Spec 8 (repetitions=N) helps but doesn't quantify. We should pick a target stddev and report it on every run.

---

## 15. What this spec does NOT change

- `RuntimeHost` API: stays exactly as in v2026-05-07.
- `Scorer` protocol: backwards-compatible — `publish` is optional.
- `voice-rollout-judges` weights: unchanged (still re-normalized for `intake_fidelity`).
- Storage format on disk: append-only additions (`scores.jsonl`, `provenance.json`, `audio/`, `timing.jsonl`); no existing artifacts removed or renamed.
- Telephony path: untouched apart from the rename to `TwilioPhoneBridge`. Production phone calls keep working.

---

## 16. Decision log

- **Default tier uses post-hoc TTS, not real Hume.** Cost (≤$0.10/rollout) and CI runtime drove this. Mirror-full release-gate preserves full-fidelity coverage.
- **Default tier alone is not sufficient to ship.** Mirror-full must run green on the deploy commit within 24h. Cheap eval is for hill-climbing iteration speed, not for release confidence.
- **File-based queue (JSONL), not Redis/Kafka.** Simplest decoupling that meets the requirement; swappable behind the watch contract.
- **Terminal renderer with `rich`, not web dashboard.** Fastest path to "I can see scores land" without scope creep.
- **Schema-diff CI, not unit tests for parity.** The schemas are derived from Pydantic models — a diff check is more durable than per-field unit tests, and it catches additions automatically.
- **Drop telecom and test-automation jargon.** `InMemoryDuplexTransport` → `InMemoryTwoWayChannel`; `LLMCustomerDriver` → `SyntheticCaller`; etc. Code names should describe what the thing does in product language.

---

## 17. Surfaced assumptions (per "the developer's assumptions ship to production")

1. **Post-hoc TTS approximates live Hume voice.** Not always true. Live Hume reacts to user prosody mid-call; post-hoc TTS is per-turn after the fact. Mirror-full release gate (Phase D) catches the divergence; default tier doesn't.
2. **Synthetic caller behavior approximates real callers.** Real users stammer, sigh, escalate, interrupt. Our caller speaks fluently with one emotional state per phase. We should evolve this by mining production audio (with consent) for caller archetypes.
3. **Sharing the `RuntimeHost` object means "mirroring the runtime."** True only at the kernel. The Hume EVI integration is a real edge that's only exercised in mirror-full. Schema-diff CI catches contract drift; it does not catch behavioral drift in edges.
4. **Three Pydantic schemas captures the contract.** `Session`, `IntakeRecord`, `CounterpartyPersona`. Doesn't cover frame formats, `transcript.jsonl` shape, audio file naming. Necessary, not sufficient — we extend the schema-diff over time as new contracts emerge.
5. **LLM nondeterminism is tolerable for hill-climbing.** Coach + customer + judge are all LLMs. We need a stddev budget per dimension before claiming a regression. Treat single-run deltas under the budget as noise.
6. **Single-host parallelism is enough.** True up to ~100 examples × 4 concurrency. If we 10x dataset size we'll need distributed execution.
7. **The spec author isn't editing telephony's coach loop without checking.** `HumeCoachAdapter` parity is a known gap (§14). It must be audited before mirror-full is trusted as a release gate.
