# rehearse — Eval System Roadmap

**Status**: draft (planning) — decomposes
`v2026-05-05-multimodal-trajectory-rubric-rlaif.md` into shippable units.
**Owner**: jz
**Depends on**:
- `docs/specs/v2026-04-29-mme-seeded-rl-sandbox-eval.md`
- `docs/specs/v2026-05-01-consent-and-outcome-capture.md`
- `docs/specs/v2026-05-05-multimodal-trajectory-rubric-rlaif.md`

This roadmap is not itself an implementation spec. It is a sequencing
document. Each numbered mini-spec below should be expanded into its own
implementation spec when work is about to start on it.

---

## 0. Where we are now

The current eval system measures:

- **One eval** (`mme-sandbox-rollout`) on **one rubric** of two text-only
  dimensions (`emotion_responsiveness`, `coaching_trajectory_quality`),
  scored by **one judge** (`TrajectoryJudgeScorer`, Claude over the
  transcript).
- **One environment** (`voice-agent-sandbox`), LLM-vs-LLM, no audio. The
  runtime captures real audio and writes `transcript.jsonl` from the live
  bus, but no eval reads from production sessions today.
- **One source of preference data**: paired sandbox rollouts, currently
  hypothetical — we have not generated any.

What this measurement system *cannot* see:

- How a candidate response is *delivered* (prosody, pacing, expressiveness).
- Whether the coach correctly read the user's *audio* affect (vs the
  transcript-readable text).
- Naturalness pathologies (interruptions, dead air, rushed speech).
- Stability — variance across repeated rollouts of the same scenario.
- Anything about real production conversations.

The judges are uncalibrated against humans; their scores cannot be safely
admitted to RL training data.

Framework-wise, evals today are written against our home-grown
`Eval`/`Environment`/`Scorer` protocols (`rehearse/eval/protocols.py`).
There is no third-party eval framework in use — every metric, test runner
hook, and dataset adapter is custom. This works but raises the cost of
adding standard text metrics (faithfulness, toxicity, role adherence,
hallucination) and means new contributors learn a bespoke API.

## 1. End state this roadmap delivers

When all mini-specs ship, the eval system will:

1. **Measure delivery (voice) alongside content (text) with parity.**
   Every trajectory is scored on three judge dimensions — `content_quality`
   (text), `delivery_quality` (audio in + audio out), `affect_perception`
   (audio in + text) — plus three deterministic naturalness sub-metrics
   (interruption, silence-after-affect, speech rate). No dimension is
   optional; voice and text are first-class measurement surfaces.
2. **Run on DeepEval as the test/metric framework where it fits.**
   Text-side metrics are written as DeepEval `BaseMetric`/`G-Eval`
   instances. Test runs use DeepEval's pytest integration. Audio,
   timing, and meta scorers stay on our custom protocol but are
   exposed to the DeepEval runner as custom metrics so a single
   command runs the full eval suite.
3. **Be calibrated**: each judge's scores have a known Spearman ρ vs
   humans, gate-checked at 0.6, recorded in a `data_card.json` per run.
4. **Run against real production calls** through a replay environment,
   gated on consent, so live behavior is measurable against the same
   rubric as sandbox.
5. **Use Best-of-N at runtime, with candidate sets persisted as part of
   the production session trace** (not sidecar files). The session bus
   that already records transcript + audio + timing also records each
   turn's candidate set, selector decision, and rationale — so a session
   trace is the complete record of what was offered, what was chosen,
   and why.
6. **Convert runtime BoN data into multimodal preference pairs** via
   offline audio re-scoring of non-chosen candidates — RLAIF-ready DPO
   input, no human labeling per pair.
7. **Detect regressions on a nightly cadence** via stability meta-metrics
   over repeated rollouts.

What this **does not** deliver, and is the next macro-spec after this
roadmap: **online training + deployment** — the actual loop that consumes
`preference_pairs.jsonl`, fine-tunes a model, deploys it behind a feature
flag, watches for regressions, and either promotes or rolls back. The
roadmap delivers the data and measurement substrate that makes online RL
*possible*; it does not deliver online RL itself.

## 2. Sequencing principles

Five principles drove the order:

1. **Measurement before action.** Build the rubric, the judges, and the
   calibration first. Don't start collecting preferences from a selector
   we don't trust yet.
2. **Cheap and deterministic before expensive and judged.** Naturalness is
   arithmetic; ship it early. Audio judges and BoN cost money and risk;
   ship later.
3. **Aligned with production before leaving sandbox.** A production-replay
   environment lands before BoN, so we can verify judges against real
   calls before building the runtime data flywheel.
4. **Runtime risk last.** BoN modifies the live phone-call path. Defer
   until eval can detect regressions in it.
5. **Stability is diagnostic, not gating.** Run nightly on a subset; never
   block PRs on it.

## 3. Wave structure

Five waves, ~16–18 weeks total. Items within a wave are independent and
can run in parallel.

```
Wave A (foundation)         Wave B (multimodal + production)    Wave C (trust)
├─ Spec 0: DeepEval           ├─ Spec 2: Audio-In Judges         └─ Spec 3: Calibration +
│  Adapter Layer              │  + Sandbox Audio                    Voice Rating UI
├─ Spec 1: Schema + Content   └─ Spec 5: Production Replay
│  Judge (DeepEval-backed)       Environment
└─ Spec 4: Timing +
   Naturalness

Wave D (runtime re-implementation + flywheel)   Wave E (diagnostic)
├─ Spec 6: Runtime Re-impl with BoN              └─ Spec 8: Stability + Repetitions
│  as Session-Trace Citizen
└─ Spec 7: Preference Pairs +
   Audio Re-scoring
```

Spec 0 lands first within Wave A so Spec 1's `ContentJudgeScorer` can be
written as a DeepEval metric directly, instead of a custom scorer that
gets ported later.

---

## 4. Mini-spec 0 — DeepEval Adapter Layer

**Wave**: A (lands first within the wave). **Estimated size**: 1–2 weeks.
**Risk**: medium (new dependency, partial-fit framework).

### Outcome
DeepEval is the framework for writing and running evals where it fits
(text-side metrics, pytest test runner, golden-set evaluation). Audio,
timing, and meta scorers continue to use our `Scorer` protocol, but are
exposed to the DeepEval runner as custom metrics so a single
`pytest evals/` command covers the full suite. New evals can be authored
in DeepEval idioms (`assert_test`, `evaluate`, `LLMTestCase`) without
losing access to our voice-aware scorers.

### What goes to DeepEval, what stays custom

| Concern | Framework | Why |
|---|---|---|
| Text content scoring (faithfulness, role adherence, toxicity, hallucination, custom G-Eval) | DeepEval | Their bread and butter; well-trodden, low-cost |
| Conversational metrics (knowledge retention, completeness) | DeepEval | Built-in `ConversationalTestCase` already fits |
| Pytest test running, CI integration | DeepEval | Standard, low-friction |
| Audio judges (`AffectPerception`, `Delivery`) | Custom `Scorer` | DeepEval has no first-class audio in/out |
| Timing-derived `NaturalnessScorer` | Custom `Scorer` | Pure arithmetic, not an LLM judgment |
| Meta-scorers (`StabilityScorer` over `repetitions`) | Custom `MetaScorer` | DeepEval has no concept of grouping rollouts by example_id |
| Rollout orchestration, sandbox env, executors | Custom | Voice-shaped; no DeepEval analogue |
| `RolloutResult` → DeepEval `LLMTestCase` shape | Adapter | Bridge layer (this spec) |

### Deliverables
- `rehearse/eval/deepeval_adapter/` package:
  - `to_test_case.py` — converts `(BenchmarkExample, RolloutResult)` to
    `LLMTestCase` / `ConversationalTestCase` with full transcript +
    expected output.
  - `from_metric.py` — wraps a DeepEval `BaseMetric` as a rehearse
    `Scorer` so DeepEval-authored metrics flow through the existing
    runner.
  - `to_metric.py` — wraps a rehearse `Scorer` as a DeepEval custom
    `BaseMetric` so audio/timing/meta scorers run inside a DeepEval
    `evaluate()` call.
- DeepEval added to `pyproject.toml` with pinned version.
- `evals/tests/` directory — pytest-based eval test files using
  DeepEval idioms, runnable via `pytest evals/`.
- One reference test: `test_mme_sandbox_rollout_content_quality.py`
  using DeepEval's G-Eval on transcripts, demonstrating the full path.
- Documentation in `rehearse/eval/README.md` updated: when to write a
  DeepEval metric vs a custom `Scorer`.

### Dependencies
None.

### Tests
- Round-trip: a `RolloutResult` → `LLMTestCase` → DeepEval metric → score
  → `RubricScore` produces the same output as the metric run directly.
- Wrapped custom scorer: `NaturalnessScorer` exposed via `to_metric` runs
  inside `evaluate()` and emits scores identical to the standalone path.
- `pytest evals/` discovers and runs the reference test.
- Backwards compatibility: every existing eval (today: `mme-sandbox-rollout`,
  `mme-emotion`, smoke evals) still runs through `rehearse-eval run`
  unchanged.

### Gates
- Cost spike check: confirm DeepEval's metric overhead (per-test LLM
  calls for `G-Eval` if used) doesn't materially inflate run cost vs the
  current judge.
- Confident AI cloud opt-out: explicit config that the framework runs
  fully local; no telemetry to a third-party dashboard unless we choose
  to enable it.

### Honest fit notes
DeepEval is text-first. Three friction points to plan around:
1. **Audio in/out has no native support** — adapter is unidirectional
   (we expose our scorers; we don't get DeepEval audio metrics for free).
2. **`LLMTestCase` assumes a single input/output pair**; coaching
   trajectories are multi-turn. `ConversationalTestCase` is the right
   primitive but its built-in metrics are narrower.
3. **The runner has its own concurrency model**; we keep our `Executor`
   for rollouts and only use DeepEval's runner for *scoring* completed
   rollouts, not for orchestrating new ones.

If the partial-fit cost grows during integration (e.g. their conversational
metric API changes break our adapter), the fallback is to keep DeepEval
as a *supplementary* metric provider only, not the test runner. That
fallback is named in §14.

---

## 5. Mini-spec 1 — Schema + Content Judge Decomposition

**Wave**: A. **Estimated size**: 1 week. **Risk**: low.

### Outcome
Today's `mme-sandbox-rollout` produces near-identical numerical results,
but the score record carries the new schema fields, scoring is decomposed
into a content-only judge plus a pure aggregator, and the content judge is
implemented as a DeepEval `G-Eval` metric. Foundation for adding more
judges.

### Deliverables
- `RubricScore` extensions: `modality`, `confidence`, `judge_prompt_version`,
  `flags`. Backwards-compatible defaults.
- `MetaScorer` protocol added (used later by Spec 8). No implementations yet.
- `ContentJudgeScorer` — implemented as a DeepEval `G-Eval` metric with
  steps + criteria narrowed to *what was said*. Wrapped via the
  `from_metric` adapter from Spec 0 so it satisfies our `Scorer` protocol.
- `AggregateScorer` — pure function, emits `weighted_reward`, writes
  `judge.json` with provenance (judge prompt version, DeepEval metric
  name + version, model IDs, confidences).
- The existing `TrajectoryJudgeScorer` is retired.

### Dependencies
- Mini-spec 0 (DeepEval adapter layer must exist).

### Tests
- Schema round-trips through JSON; old artifacts deserialize with defaults.
- `mme-sandbox-rollout` numerical regression test: scores within rounding
  of pre-change baseline on the existing dataset.
- `judge_prompt_version` populated on every emitted `RubricScore`.

### Gates
None — pure refactor.

---

## 6. Mini-spec 2 — Audio-In Judges + Sandbox Audio Capture

**Wave**: B. **Estimated size**: 2–3 weeks. **Risk**: medium (provider
integration, audio I/O).

### Outcome
Sandbox rollouts produce per-turn coach + user audio; two new judges score
audio-aware dimensions; full three-dimension rubric is alive on sandbox
runs.

### Deliverables
- Sandbox env wires `tts_bridge.py` into the coach turn loop, persists
  `audio/coach/turn_<N>.wav`.
- For user turns: MME clip on opening; TTS for subsequent user turns
  (cheapest sandbox audio path).
- `AffectPerceptionJudgeScorer` (Gemini 2.5, audio-in over user audio +
  transcript). Emits `affect_perception` ∈ [0,1] + per-turn affect flags
  (consumed later by `NaturalnessScorer`).
- `DeliveryJudgeScorer` (Gemini 2.5, user audio + coach audio). Emits
  `delivery_quality` per Appendix A anchors of the parent spec.
- Both judges respect Appendix A degradation rules
  (`audio_missing` flag, etc).

### Dependencies
- Mini-spec 1 (schema, AggregateScorer).

### Tests
- Each judge: scores ∈ [0, 1]; emits `judge_prompt_version`; degrades
  cleanly when audio missing.
- Sandbox produces `audio/coach/turn_<N>.wav` for every coach turn.
- High-affect turn flags from `AffectPerceptionJudgeScorer` are persisted
  in a stable location for downstream scorers.

### Gates
- Cost gate: per-rollout audio judge cost measured and below an agreed
  ceiling before this becomes the default eval scorer.

---

## 7. Mini-spec 3 — Calibration Harness + Voice Rating UI

**Wave**: C. **Estimated size**: 2 weeks. **Risk**: low.

### Outcome
Each judge's Spearman ρ vs humans is computed and recorded. No judge's
scores are admitted to training data until ρ ≥ 0.6 on its dimension.
Humans rate trajectories by speaking structured scores; an LLM parser
converts the speech to scalars (it does *not* interpret reasoning).

### Deliverables
- New routes in `rehearse/viewer.py`:
  - `GET /viewer/{session_id}/rate` — renders trajectory + record button.
  - `POST /viewer/{session_id}/rate` — accepts rater audio, transcribes,
    parses to `{dimension: float}`, stores `HumanRating` JSON.
- Rating storage at `evals/golden/v1/ratings/{session_id}__{rater}.json`.
- Initial 25-trajectory golden set selected (script that samples across
  the affect distribution from existing sandbox runs).
- `CalibrationHarness` — joins judge scores against human ratings,
  computes per-dimension Spearman ρ.
- `data_card.json` writer in the runner; `calibration_status` populated
  per dimension.
- `excluded_flags` filter applied to preference data (used by Spec 7).

### Dependencies
- Mini-spec 1 (schema with `flags`).
- Mini-spec 2 (judges to calibrate).

### Tests
- Voice rating round-trip: known audio recording → expected scalar scores.
- Parser rejection: ambiguous speech → re-record prompt, no partial save.
- Spearman ρ computation against fixture data.
- `data_card.calibration_status[dim].passed` flips correctly at the 0.6
  threshold.

### Gates
- Initial calibration set rated by jz before any downstream spec consumes
  judge scores as ground truth.
- Selector calibration (per parent spec §8.2) is gated on this same UI.

---

## 8. Mini-spec 4 — Timing Instrumentation + Naturalness Scorer

**Wave**: A (parallel with Spec 1). **Estimated size**: 2 weeks. **Risk**:
medium (touches runtime audio path).

### Outcome
Every rollout — sandbox and production — emits `timing.jsonl`. A
deterministic `NaturalnessScorer` produces three sub-metric `RubricScore`
rows on every eval run. Dashboard signal lights up immediately; no
calibration burden.

### Deliverables
- Runtime: emit per-turn `{turn_index, role, event, t_ms, duration_ms}`
  records to `timing.jsonl` from the live audio bus. Wired into
  `rehearse/writers/artifacts.py` alongside `TranscriptArtifactWriter`.
- Sandbox: emit equivalent records from TTS metadata + simulated user-turn
  pacing.
- `NaturalnessScorer` — pure arithmetic, three sub-metrics
  (`interruption_rate`, `silence_after_affect`, `speech_rate_band`),
  banded against the v1 thresholds in Appendix A of the parent spec.
- `thresholds_version` field on `NaturalnessScorer` `RubricScore` rows.

### Dependencies
- Mini-spec 1 (schema with `modality="timing"`).
- `silence_after_affect` band depends on Mini-spec 2's high-affect flags
  *if* you want to restrict the metric to high-affect turns only;
  otherwise compute on all turns.

### Tests
- Fixture timing → expected metric values (one test per sub-metric).
- Missing `timing.jsonl` → `timing_missing` flag, no crash.
- Threshold change requires `thresholds_version` bump (test asserts both
  versions produce different scores on the same fixture).

### Gates
- None.

---

## 9. Mini-spec 5 — Production-Replay Environment

**Wave**: B. **Estimated size**: 1–2 weeks. **Risk**: medium (consent
boundary, schema parity with sandbox).

### Outcome
Real consented production calls can be scored by every existing judge.
This is the first time the eval system measures something other than
sandbox rollouts; it closes the train-serve gap on the *measurement* side.

### Deliverables
- `ProductionSessionsDataset` — enumerates completed sessions from
  `rehearse/storage.py` with `Session.consent == "granted"`.
- `ProductionReplayEnvironment` — `rollout()` resolves the session's
  artifacts directory and returns a `RolloutResult` pointing at the
  on-disk transcript + audio + timing.
- `mme-sandbox-rollout` eval gains `"production-replay"` to its
  `supported_environments`. Scoring plan unchanged.
- Verify schema parity: production `transcript.jsonl` shape matches what
  the offline judges expect (a small adapter may be needed).

### Dependencies
- Mini-spec 1 (schema).
- Mini-spec 2 (judges that will score the production calls).
- Mini-spec 4 (production runtime emits `timing.jsonl`).
- `v2026-05-01-consent-and-outcome-capture.md` must be live in the
  runtime, since this spec is gated on `Session.consent`.

### Tests
- Non-consented session is excluded from the dataset; assertion in tests.
- A stored production session can be replayed end-to-end; produces a
  valid `RolloutResult` with `artifacts_dir` set.
- Schema parity: production `transcript.jsonl` parses with the same
  loader the sandbox uses.

### Gates
- Consent capture (05-01) shipped to production. Without that, this spec
  has no input data.

---

## 10. Mini-spec 6 — Runtime Re-implementation: BoN as Session-Trace First-Class Citizen

**Wave**: D. **Estimated size**: 4–5 weeks. **Risk**: high (live phone-call
path, latency budget, streaming → buffered fork, session-trace schema
change).

### Outcome
The runtime is re-implemented so that every coach turn on a consented BoN
session produces N=2 candidates in parallel, a selector picks one, the
chosen candidate is streamed to the user, and **the full candidate set
(all N candidates, selector decision, selector rationale, selector scores,
latencies) is recorded as a first-class event on the session trace** —
written through the same `Bus` and `Storage` mechanisms as transcript and
audio, not as a sidecar file.

A "session trace" after this spec is the complete, replayable record of
what was offered, what was chosen, why, and what was heard. There is no
separate "candidates" directory; the candidate sets live alongside
transcript turns in the bundle that the existing `viewer.py` already
renders.

This is intentionally a re-implementation, not a wrapper. The current
`CLMResponder.stream_reply` interface streams tokens directly to TTS,
which is incompatible with selecting after generation. We replace the
single-stream path with a **buffered-then-streamed path** as the *new
default for BoN-eligible sessions*, with the old streaming path
preserved as a fallback for non-BoN sessions and for failure recovery.

### Deliverables

**Runtime core changes:**
- `CoachResponder` protocol (replaces `CLMResponder` for the BoN path):
  - `generate_candidates(prompt_state, n) -> list[Candidate]` — parallel
    via `asyncio.gather`.
  - `select(context, candidates) -> SelectorDecision` — runs the
    selector.
  - `stream(chosen) -> AsyncIterator[bytes]` — re-streams the chosen
    candidate to the existing SSE contract Hume EVI expects.
- The OpenAI-compatible `/v1/chat/completions` endpoint (`clm.py:131`) is
  re-implemented to drive `CoachResponder` instead of `CLMResponder`
  directly. The SSE byte-stream contract to Hume is unchanged; the
  internals are not.
- `Selector` — Haiku-class text judge over recent context + N candidate
  texts; emits `chosen_index`, per-dimension scalar scores, rationale;
  800ms timeout with `chosen_index=0` fallback.
- Single-candidate fallback path: both candidates fail, selector times
  out, or BoN disabled at the session level → fall through to the
  existing single-stream `CLMResponder`. Recorded as a `bon_fallback`
  event in the session trace.

**Session-trace integration:**
- New event type on the runtime `Bus`: `CandidateSetEvent`. Carries the
  full `TurnCandidateSet` payload.
- `Session` and `SessionPhase` types extended to include candidate-set
  events in the trace timeline alongside transcript turns and audio
  segments. (No new top-level storage path; uses existing session bundle.)
- `TranscriptArtifactWriter` companion: a `CandidateSetWriter` subscribes
  to `CandidateSetEvent`s and writes them into the session bundle in the
  same JSONL append-only style — `candidates.jsonl` colocated with
  `transcript.jsonl`. Replayable via the same loader.
- `viewer.py` renders candidate sets inline in the trace view: for each
  coach turn, show the chosen response highlighted, alternates collapsed,
  selector rationale on hover.

**Gating and lifecycle:**
- Per-session BoN eligibility computed at session start:
  `consent == "granted"` AND BoN flag on AND fallback healthy.
- Session record carries `bon_enabled: bool` so traces are
  self-describing — a non-BoN session has no candidate events; a BoN
  session has one per coach turn.
- A session-level config snapshot is written into the trace at start:
  `selector_model`, `selector_prompt_version`, `n_candidates`,
  `rubric_version`. Reproducibility for any historical session.

### Dependencies
- Mini-spec 1 (schema for `RubricScore`, `Candidate`, `TurnCandidateSet`).
- Mini-spec 3 (selector calibration must pass before live ship).
- Mini-spec 4 (timing instrumentation already part of the trace; BoN
  builds on it).

### Tests (load-bearing)
- `test_generate_candidates_dispatches_in_parallel` — two 200ms mock
  calls complete in <350ms.
- `test_runtime_bon_latency_within_budget` — full BoN turn ≤ 1.2× single-
  call baseline against a fixture LLM client.
- `test_time_to_first_audio_within_budget` — buffered → re-stream first
  byte arrives ≤ 1.5× the single-stream baseline. *This is separate from
  the overall latency test* and matters more for user perception.
- `test_selector_timeout_falls_back_to_zero` — selector mock exceeding
  800ms produces `chosen_index=0` and a `selector_timeout` flag on the
  trace event.
- `test_candidate_set_event_appears_in_session_trace` — a BoN session's
  loaded trace contains one `CandidateSetEvent` per coach turn,
  interleaved with transcript turns at the correct timestamps.
- `test_non_bon_session_emits_no_candidate_events` — control case;
  schema is opt-in.
- `test_session_trace_replay_round_trip` — load → re-serialize → reload
  preserves all candidate events; `data_card.json` derived from the
  reloaded trace matches the original.
- `test_bon_disabled_for_unconsented_session` — session record shows
  `bon_enabled=false`; no candidate events written.
- `test_viewer_renders_candidate_set` — viewer route returns HTML with
  chosen response highlighted and alternates accessible.

### Gates
- Selector calibration (Spec 3) before BoN ships to production.
- Latency tests must pass against a realistic Hume EVI session, not just
  mock fixtures. If Hume's first-token timeout is below our buffered-path
  TTFT, BoN cannot ship on the EVI transport without more work — that
  becomes its own spec.
- Trace-replay test: a session bundle written under the new schema must
  still load on the existing viewer for non-BoN sessions (additive
  compatibility).

---

## 11. Mini-spec 7 — Preference Pair Generation + Offline Audio Re-scoring

**Wave**: D. **Estimated size**: 2 weeks. **Risk**: low (mostly
orchestration over existing pieces).

### Outcome
Every consented BoN turn — sourced by reading `candidates.jsonl` from the
session trace produced in Spec 6 — produces a `PreferencePair` with
audio-aware dimension scores on both sides. DPO-shaped training data
ready to consume.

### Deliverables
- Session-trace reader: enumerates `CandidateSetEvent`s from completed
  session bundles and yields `TurnCandidateSet` records.
- `turn_candidate_set_to_pairs` — pure function with deterministic UUIDs.
- Margin floor (default 0.05; configurable).
- Offline TTS step for non-chosen candidates (batched after call ends or
  on a nightly job).
- Audio re-scoring driver: invokes `DeliveryJudgeScorer` over synthesized
  audio paired with user audio.
- `preference_pairs.jsonl` writer per eval run.
- Source field tags pairs by origin (`runtime_bon`, `offline_rescored`).

### Dependencies
- Mini-spec 6 (`TurnCandidateSet` exists).
- Mini-spec 2 (`DeliveryJudgeScorer`).
- Mini-spec 3 (calibration gate — `delivery_quality` ρ ≥ 0.6).

### Tests
- Pure function: same `TurnCandidateSet` → same `PreferencePair[]` (UUIDs
  derived deterministically).
- Margin floor: pairs below threshold excluded.
- Audio re-scoring: synthesized non-chosen TTS produces a delivery score;
  the resulting pair carries `delivery_quality` for both sides.
- Idempotency: re-running re-scoring on the same TCS produces identical
  output.
- Traceability: every pair traces back to a stored `TurnCandidateSet` ID.

### Gates
- `delivery_quality` past calibration floor before pairs are written to
  the canonical training-data path.

---

## 12. Mini-spec 8 — Stability via Repetitions + Meta-Scorer

**Wave**: E. **Estimated size**: 1 week. **Risk**: low.

### Outcome
Nightly job runs `repetitions=5` on a 10-trajectory golden subset and
emits per-dimension stability scores. Trends visible on the dashboard;
spikes alert.

### Deliverables
- `RunConfig.repetitions: int = 1`.
- Runner schedules `repetitions` rollouts per example with distinct
  `rng_seed`.
- `StabilityScorer` — meta-scorer; computes
  `1 - normalized_stddev(scores_per_dim)`.
- Records sampling temperature on each stability `RubricScore`; flags
  `mixed_temperature` if rollouts disagree.
- Cron entry that runs `--eval golden-stability --repetitions 5` nightly.

### Dependencies
- Mini-spec 1 (`MetaScorer` protocol).
- Mini-spec 2 (per-dim judges to vary across).

### Tests
- Three fixture rollouts with known variance → expected stability scores.
- `repetitions=1` → `stability_unmeasurable` flag.
- Mixed temperatures across rollouts → `mixed_temperature` flag.

### Gates
- None — diagnostic only, never PR-gating.

---

## 13. After this roadmap (out of scope here)

The roadmap delivers a measurement substrate and a preference-data
flywheel. It does not deliver:

1. **Online training loop.** A worker that consumes `preference_pairs.jsonl`
   on a cadence, fine-tunes a candidate model (DPO/IPO), and produces a
   versioned weight artifact.
2. **Model deployment + canary.** A flag-controlled rollout that swaps in
   a candidate model for a fraction of traffic, monitors for regressions
   on the calibrated rubric, and either promotes or rolls back.
3. **Drift / regression alerting.** Watches `weighted_reward` and
   stability trend lines per model version; pages on degradation.
4. **Continuous evaluation cron.** Runs the full golden + production
   eval on every promoted model, automatically.

These four are the *online-RL* macro-spec, written next, after this
roadmap is in flight.

## 14. Risks and what would force a re-sequencing

- **Hume EVI latency ceiling**: if Spec 6's buffered TTFT is incompatible
  with EVI, BoN must move to a non-EVI transport, which becomes its own
  spec ahead of Spec 6.
- **Judge cost at scale**: if Spec 2's audio judges are prohibitively
  expensive per-rollout, we may need a sampling strategy or a cheaper
  judge before Spec 5 (production replay) becomes affordable.
- **Calibration ρ misses**: if Spec 3 shows ρ < 0.6 for a dimension, we
  cannot use that dimension's scores for training. Spec 7 may ship with
  fewer dimensions than planned; downstream training spec adjusts.
- **Consent capture delays**: Spec 5 is gated on 05-01 shipping to
  production. If 05-01 slips, Spec 5 shifts right.
- **DeepEval partial-fit growth**: DeepEval is text-first. Spec 0 plans
  for an adapter layer that bridges our voice-shaped scorers to
  DeepEval's metric/runner model. If their `ConversationalTestCase` API
  shifts or their multimodal support stagnates, we may have to demote
  DeepEval from "the framework" to "a metric provider" — keeping our
  custom runner and using DeepEval only for individual text metrics.
  Detect early: Spec 0 ships one reference test against the real
  framework before Spec 1 commits to DeepEval-backed `ContentJudgeScorer`.
- **Session-trace schema churn**: Spec 6 extends the session record with
  candidate events. Any in-flight work that reads the session bundle
  (viewer, replay, exports) must be updated atomically. Coordinate with
  whoever holds the session-trace contract before Spec 6 lands.

## 15. Manifest update

Add to `docs/specs/MANIFEST.md`:

| Spec | Status | Policy | Applies to | Notes |
|---|---|---|---|---|
| `v2026-05-06-eval-system-roadmap.md` | `acknowledged` | `implementation` | Eval system sequencing | Roadmap that decomposes the 05-05 spec into eight mini-specs. Each mini-spec gets its own implementation spec when work begins. |
