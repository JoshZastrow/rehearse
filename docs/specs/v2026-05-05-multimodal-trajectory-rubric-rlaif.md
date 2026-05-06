# rehearse — Spec: Multimodal Rubric + Runtime BoN + RLAIF Pipeline

**Status**: draft (design-facing) — revised 2026-05-06 to add runtime
Best-of-N candidate generation, plus voice-quality measurement
(expressiveness, naturalness, stability).
**Owner**: jz
**Depends on**:
- `docs/specs/v2026-04-27-eval-harness.md`
- `docs/specs/v2026-04-29-mme-seeded-rl-sandbox-eval.md`
- `docs/specs/v2026-05-01-consent-and-outcome-capture.md`

**Amends**: §6.3, §7 of `v2026-04-29-mme-seeded-rl-sandbox-eval.md`.

---

## 0. Outcome (What success looks like)

When this spec is delivered, the rehearse runtime and eval harness will
together produce a continuously growing, calibrated, multimodal preference
dataset suitable for DPO-style training:

1. **Every consented coach turn produces N=2 candidate responses in parallel,
   a selector picks one, and the chosen response is what the user hears.**
   Best-of-N ships an inference-time quality lift before any model fine-tune.
2. **Each turn's candidate set is persisted as a `TurnCandidateSet` record**,
   which converts deterministically to a `PreferencePair` for training. No
   reward model is trained; we go straight to pairwise preference data.
3. **Three per-dimension judges score trajectories**: `ContentJudge` (text),
   `AffectPerceptionJudge` (audio in), `DeliveryJudge` (audio in + audio out).
   Each judge is independently calibrated against humans before its scores
   may enter training data.
4. **Offline audio re-scoring fills in delivery dimensions** that the
   text-only runtime selector can't see, producing audio-aware preference
   pairs without making the live phone call any slower.
5. **Latency budget held**: parallel candidate generation keeps end-to-end
   coach-turn latency within ~1.2× of single-candidate latency, not 2×.
6. **A `data_card.json` per eval run** records modality coverage, calibration
   status per dimension, and consent — so any training run is traceable to a
   known-quality slice of data.
7. **Voice-quality measurement** surfaces failure modes the per-trajectory
   judges can't see: a deterministic `NaturalnessScorer` over per-turn timing
   data (interruption rate, silence-after-affect, speech rate), sharpened
   `delivery_quality` anchors that interrogate expressiveness directly, and a
   `StabilityScorer` meta-metric over repeated rollouts of the same scenario
   seed (catches policy non-determinism that a single-rollout judge misses).

Doing nothing means RL training on a text-only rubric and producing a coach
that says the right empathetic phrases in any tone of voice. This spec
prevents that failure mode, and along the way ships a meaningful quality
improvement to live calls.

## 1. Why This Exists

The current rubric reads only the transcript. Production runtime is
multimodal. Training a reward signal on a text-only rubric and serving a
multimodal product is **train-serve skew at the reward-model layer**: the
policy will optimize for *saying* emotionally intelligent things, not *being*
emotionally attuned. Three concrete failure modes the existing rubric cannot
detect:

1. **Tonal mismatch** — right words, flat delivery.
2. **Misread affect** — coach text response keys on the wrong emotion.
3. **Prosodic non-attunement** — coach prosody doesn't track user prosody.

Separately, today's only path to preference data is paired sandbox rollouts.
That is synthetic, expensive, and detached from real users. **Runtime BoN**
collects preferences from real conversations as a free side effect of an
inference-time quality lift, replacing sandbox-paired rollouts as the primary
source.

## 2. Scope

### In Scope

- A three-dimension rubric (§3): `affect_perception`, `delivery_quality`,
  `content_quality`.
- Per-dimension scorers, each independently calibrated (§5).
- Runtime Best-of-N candidate generation with parallel execution (§4).
- A runtime selector that scores candidates and picks one (§4.2).
- `TurnCandidateSet` persistence per turn (§6).
- Conversion from `TurnCandidateSet` to `PreferencePair[]` (§7).
- Offline audio re-scoring of stored candidates (§5.4).
- Calibration protocol, shrunk for v0 (§8).
- Multi-process / async parallelism plan (§4.1).
- Deterministic `NaturalnessScorer` over per-turn timing data (§5.6).
- `StabilityScorer` meta-metric over repeated rollouts (§5.7).
- `RunConfig.repetitions` and a `MetaScorer` protocol category (§6.5).
- Sharpened expressiveness anchors on `delivery_quality` (Appendix A).
- Test plan (§9).

### Out of Scope

- Choice of RL training algorithm. The spec produces DPO-shaped data; PPO/RM
  workflows can backfill from preferences later.
- A judge-scored `turn_dynamics` dimension. Naturalness is *deterministic
  only* (§5.6); a contextual judge over naturalness signals (e.g. "was 4s
  of silence right *here*?") may follow but is not in scope.
- Per-segment (sub-turn) rewards. Trajectory- and turn-level only.
- Production-replay environment as a *training data* source. Replay is for
  re-scoring trajectories on new rubrics (eval), not for generating
  preferences (BoN does that).
- Online RL or live policy updates.

## 3. Rubric

The rubric has three *shapes* of measurement, each with different cost and
calibration profiles. Treat them as separate signals, not as one collapsed
score.

### 3.1 Per-trajectory judge dimensions (three)

Each independently observable, scored 0.0–1.0 by an offline judge, calibrated
separately against humans.

| Dimension | Modality | Question | Default weight |
|---|---|---|---|
| `affect_perception` | audio in (user) + text | Did the coach correctly read the user's state? | 0.35 |
| `delivery_quality` | audio in + audio out | Did the coach's delivery match the moment? Includes prosodic *attunement* (matches user) and *expressiveness* (range/animation). | 0.30 |
| `content_quality` | text | Was what the coach said useful, safe, trajectory-positive? | 0.35 |

`weighted_reward = Σ w_i * score_i`. Weights live on the eval payload
(`rubric_weights`) and override defaults; matches existing 04-29 mechanism.

Expressiveness is folded into `delivery_quality` rather than split out.
Calibration may force a split later — if the judge cannot reliably
distinguish "tonal match to user" from "expressive range" — but absent that
evidence, one dimension keeps the calibration burden down.

### 3.2 Naturalness sub-metrics (deterministic)

Naturalness is computed directly from `timing.jsonl` — **no judge, no
calibration ρ.** Cheap, reproducible, banded against pinned thresholds.
Three sub-metrics, each emitted as its own `RubricScore`:

| Sub-metric | Source | Bands |
|---|---|---|
| `naturalness.interruption_rate` | timing | events / user turn — 0.0 ideal, ≤0.2 acceptable, >0.5 pathological |
| `naturalness.silence_after_affect` | timing + affect flags | seconds after high-affect turn — 1.5–4.0s ideal, <1.0s rushed, >6.0s dead air |
| `naturalness.speech_rate_band` | timing | words/minute — 130–170 paced, >200 rushed, <100 slow |

Thresholds are pinned and versioned. They do not enter the
`weighted_reward` directly; they are reported separately on the dashboard
and feed an optional naturalness gate in the data card.

### 3.3 Stability (meta-metric)

Stability is **not** a property of a single trajectory. It is the *variance*
of per-trajectory dimension scores when the same scenario seed is rolled out
M times. A policy that scores 0.9 once and 0.3 the next time on the same
input is unusable in production, regardless of the mean.

```
seed S ──┬─► rollout 1 ──► judges ──► scores_1[dim]
         ├─► rollout 2 ──► judges ──► scores_2[dim]
         ├─► ...
         └─► rollout M ──► judges ──► scores_M[dim]

stability(dim) = 1 - normalized_stddev(scores_1[dim], ..., scores_M[dim])
```

Properties:

- **Per-dimension** — you can be stable on `content_quality` and unstable
  on `delivery_quality`; report each separately.
- **Temperature-sensitive** — measured stability depends on sampling
  temperature. Always report the temperature it was measured at.
- **Cost-sensitive** — M× rollouts means M× judge cost. Stability runs
  *nightly on a subset*, not on every PR.

Stability scores are emitted as `RubricScore` rows with `dimension`
prefixed `stability.<dim>` and `modality="meta"`.

Anchors and degradation rules for §3.1 and §3.2: see Appendix A.

## 4. Runtime Best-of-N

### 4.1 Candidate Generation (parallel)

For each coach turn on a consented session:

```
user_turn  ─►  CandidateGenerator (N=2)  ─►  [candidate_0, candidate_1]
                       │
                       └── two LLM calls dispatched in parallel
```

**Parallelism choice: `asyncio.gather`, not `multiprocessing`.**

Candidate generation is **I/O-bound** — the wall time is network round-trip
to the LLM provider, not local CPU. `asyncio.gather` runs N requests
concurrently in one event loop, single process, no serialization overhead,
no IPC. `multiprocessing.Pool` would buy nothing here and add cost (process
startup, pickling the request, cross-process state).

```python
async def generate_candidates(prompt_state, n: int = 2) -> list[Candidate]:
    coros = [_one_candidate(prompt_state, idx=i) for i in range(n)]
    candidates = await asyncio.gather(*coros, return_exceptions=True)
    return [c for c in candidates if not isinstance(c, Exception)]
```

**Where multiprocessing is the right answer**: any *CPU-bound* step in the
turn pipeline. The realistic candidate today is **local TTS rendering** if
we ever bring TTS in-process. Then `concurrent.futures.ProcessPoolExecutor`
across N candidates is correct, because TTS is CPU-bound and blocks the
event loop. For now TTS is provider-side; only the chosen candidate is
synthesized at runtime, so this doesn't apply yet.

**Latency math (N=2, asyncio.gather)**:

```
single-candidate latency:  T_llm
N=2, sequential:           2 * T_llm                  (worst case)
N=2, asyncio.gather:       max(T_llm_0, T_llm_1)      (target)
N=2, end-to-end with sel:  max(T_llm_0, T_llm_1) + T_selector + T_tts
```

Selector adds one short LLM call (~150 tokens of rubric scoring). Target
end-to-end overhead vs single-candidate path: **≤ 1.2×**. Tested in §9.3.

**Failure handling**: if a candidate generation fails or times out, the
remaining candidate(s) proceed; if both fail, fall back to a single
non-BoN call (existing path). No user-visible failure surface.

### 4.2 Selector

A lightweight, low-latency LLM judge that:

- Reads recent dialogue context + the N candidate texts.
- Scores each candidate per dimension on a coarse rubric (text-readable
  signals only — `content_quality` directly, `affect_perception` partially
  from word choice).
- Picks `chosen_index` and emits a one-line rationale.

**Constraints:**

- Text-only at runtime. Audio dimensions cannot be judged before TTS without
  unacceptable latency. Audio re-scoring happens offline (§5.4).
- Smaller/faster model than offline judges (Haiku-class). The selector is a
  lower-stakes judge — it picks between similar candidates from the same
  policy, not absolute scoring.
- Times out fast (e.g. 800ms). On timeout, falls back to `chosen_index = 0`.

**The selector is itself a judge** and subject to the calibration gate (§8).

### 4.3 Persistence

Every BoN turn writes one `TurnCandidateSet` row to durable storage,
gated on consent (per 05-01). Schema in §6.

## 5. Scoring Pipeline (Offline)

Three per-dimension scorers run on completed trajectories. Per-dimension
isolation lets each calibrate independently and lets us swap models per
dimension (text → Claude, audio → Gemini).

### 5.1 `ContentJudgeScorer`

- Input: `transcript.jsonl`.
- Model: Claude (text).
- Output: `content_quality` ∈ [0,1] + rationale + key turn indices +
  confidence.

Direct evolution of today's `TrajectoryJudgeScorer`, prompt narrowed to
*what was said*, not *how it landed*.

### 5.2 `AffectPerceptionJudgeScorer`

- Input: `transcript.jsonl` + `audio/user/turn_<N>.wav`.
- Model: Gemini 2.5 (audio in).
- Output: `affect_perception` ∈ [0,1].

### 5.3 `DeliveryJudgeScorer`

- Input: `audio/user/turn_<N>.wav` + `audio/coach/turn_<N>.wav`.
- Model: Gemini 2.5 (audio in).
- Output: `delivery_quality` ∈ [0,1].

### 5.4 Audio Re-scoring of Stored Candidates

To turn runtime BoN data into audio-aware preference pairs, we **re-score
non-chosen candidates offline** by:

1. Generating TTS for each non-chosen candidate text (cheap, batchable,
   runs after the call ends).
2. Running `DeliveryJudgeScorer` over the synthesized audio paired with
   the user audio.
3. Updating the `TurnCandidateSet` with offline scores.

This is the move that lets a text-only runtime selector still produce
audio-aware training data. Latency cost is offline; user experience is
unaffected.

### 5.5 `NaturalnessScorer` (deterministic)

Reads `timing.jsonl` from `rollout.artifacts_dir`; emits one `RubricScore`
per sub-metric (§3.2). No LLM call, no calibration ρ — but threshold values
are pinned and versioned. Works in degraded form when timing is missing
(emits a `timing_missing` flag rather than failing).

```python
class NaturalnessScorer:
    name = "naturalness"
    thresholds_version = "v1"
    async def score(example, rollout, run_id) -> list[RubricScore]:
        timing = load_jsonl(rollout.artifacts_dir / "timing.jsonl")
        return [
            _interruption_rate(timing),
            _silence_after_affect(timing, rollout),
            _speech_rate_band(timing),
        ]
```

The rollout must produce `timing.jsonl` (per §6 of this spec). That is the
load-bearing instrumentation work; the scorer itself is arithmetic. High-
affect turn detection for `silence_after_affect` is a side product of
`AffectPerceptionJudgeScorer` — it flags the turns; this scorer reads the
flags.

### 5.6 `StabilityScorer` (meta)

A *meta-scorer*: operates on a list of `RolloutResult`s sharing an
`example_id`, after the per-trajectory scorers have run. Emits one
`RubricScore` per dimension with `dimension="stability.<dim>"`.

```python
class StabilityScorer:
    name = "stability"
    async def score_meta(
        example: BenchmarkExample,
        rollouts: list[RolloutResult],
        per_rollout_scores: list[list[RubricScore]],
        run_id: str,
    ) -> list[RubricScore]: ...
```

Run only when `RunConfig.repetitions > 1` (see §6.5). On nightly stability
runs the harness sets `repetitions=5`. Stability scores live alongside the
mean per-dimension scores in `summary.md`.

### 5.7 `AggregateScorer`

Pure function: per-dimension scores → `weighted_reward`. Emits `judge.json`
per rollout with provenance (judge prompt versions, model IDs, confidences,
naturalness thresholds version). Stability scores are *not* aggregated into
`weighted_reward` — they are reported separately because they describe the
policy, not a single trajectory.

## 6. Schema

### 6.1 `RubricScore` extensions (additive)

```python
class RubricScore(BaseModel):
    # existing fields unchanged
    ...
    modality: Literal["text", "audio", "audio+text", "timing", "meta", "aggregate"] = "text"
    confidence: float | None = None
    judge_prompt_version: str | None = None
    flags: list[str] = []   # e.g. ["audio_missing", "uncalibrated"]
```

No `Segment` field; trajectory- and turn-level only.

### 6.2 `TurnCandidateSet` (new, runtime-emitted)

```python
class Candidate(BaseModel):
    candidate_index: int
    text: str
    audio_path: Path | None        # set after TTS (chosen now; others later via re-scoring)
    generation_metadata: dict[str, Any]   # model, temperature, seed, latency_ms

class TurnCandidateSet(BaseModel):
    candidate_set_id: str
    session_id: str
    turn_index: int

    candidates: list[Candidate]
    chosen_index: int
    selector_model: str
    selector_prompt_version: str
    selector_rationale: str
    selector_scores: dict[str, list[float]]   # dim -> [score per candidate]
    selector_latency_ms: int

    rubric_version: str
    consent: Literal["granted", "declined", "unknown"]
    created_at: datetime
```

Stored under `sessions/{session_id}/candidates/turn_{NN}.json`.

### 6.3 `PreferencePair` (new, derived)

```python
class PreferencePair(BaseModel):
    pair_id: str
    source: Literal["runtime_bon", "sandbox_paired", "offline_rescored"]
    session_id: str | None
    turn_index: int | None
    example_id: str | None

    chosen: Candidate
    rejected: Candidate
    dimension: str            # which dim drove the preference, or "weighted_reward"
    margin: float

    chosen_scores: dict[str, float]
    rejected_scores: dict[str, float]
    judge_prompt_versions: dict[str, str]   # dim -> version
    created_at: datetime
```

Stored under `evals/runs/{run_id}/preference_pairs.jsonl`.

### 6.4 `data_card.json`

Per eval run, summarizes the slice safe to train on:

```python
{
  "run_id": "...",
  "rubric_version": "v2",
  "min_confidence": 0.6,
  "min_modalities": 2,
  "excluded_flags": ["uncalibrated", "audio_missing"],
  "consent_filter": "granted_only",
  "calibration_status": {
    "content_quality": {"rho": 0.71, "passed": true},
    "affect_perception": {"rho": 0.58, "passed": false},
    "delivery_quality": {"rho": 0.64, "passed": true}
  },
  "counts": {"in": 12450, "out": 9120}
}
```

### 6.5 `RunConfig.repetitions` and `MetaScorer` protocol

Adds two pieces to the existing eval harness:

```python
class RunConfig:
    ...
    repetitions: int = 1   # rollouts per example; >1 enables stability scoring

@runtime_checkable
class MetaScorer(Protocol):
    name: str
    async def score_meta(
        self,
        example: BenchmarkExample,
        rollouts: list[RolloutResult],
        per_rollout_scores: list[list[RubricScore]],
        run_id: str,
    ) -> list[RubricScore]: ...
```

The runner schedules `repetitions` rollouts per example with distinct
`rng_seed` values, runs the per-trajectory scorers as today, and then runs
any `MetaScorer` instances over the grouped results. With `repetitions=1`,
meta-scorers receive a single-element list and either no-op or emit
`stability_unmeasurable` flags. The protocol change is additive — existing
scorers and existing `Eval.scoring_plan()` outputs are unaffected.

### 6.6 Rollout artifact: `timing.jsonl`

Required for `NaturalnessScorer`. One JSON object per turn boundary event:

```json
{"turn_index": 4, "role": "coach", "event": "audio_start",
 "t_ms": 18420, "duration_ms": null}
{"turn_index": 4, "role": "coach", "event": "audio_end",
 "t_ms": 21105, "duration_ms": 2685}
```

Both runtime and sandbox rollouts must emit it. Runtime sources timestamps
from the live audio bus; sandbox sources them from TTS metadata + simulated
user-turn pacing. When `timing.jsonl` is absent, `NaturalnessScorer` emits
`timing_missing` and skips its sub-metrics.

## 7. Conversion: BoN → Preferences

Pure function, deterministic, replayable:

```python
def turn_candidate_set_to_pairs(
    tcs: TurnCandidateSet,
    *,
    margin_floor: float = 0.05,
) -> list[PreferencePair]:
    chosen = tcs.candidates[tcs.chosen_index]
    pairs = []
    for cand in tcs.candidates:
        if cand.candidate_index == tcs.chosen_index:
            continue
        margin = _compute_margin(tcs, chosen, cand)
        if margin < margin_floor:
            continue   # near-tie, skip
        pairs.append(PreferencePair(...))
    return pairs
```

For N=2 every non-tie turn produces exactly one pair. Margin floor
prevents low-information pairs from polluting the training set.

When offline audio re-scoring lands, we recompute pairs with audio
dimensions weighted in. The conversion stays a pure function of the stored
`TurnCandidateSet`, so re-deriving training data from production logs is
always possible without re-running production.

## 8. Calibration

### 8.1 Per-Judge Calibration

Each of the three offline judges + the runtime selector is calibrated
independently:

- **25 trajectories** sampled across the affect distribution.
- **One human rater** scoring all dimensions, with a spot-check from a
  second rater on 5 trajectories to flag drift.
- **Acceptance floor**: judge-vs-human Spearman ρ ≥ 0.6 per dimension
  before that dimension's scores enter training data.

(Shrunk from the v1 spec's 50 trajectories / 2 raters. Cost-down per
the simplification pass.)

### 8.2 Selector Calibration

The selector is calibrated against the offline judges, not directly against
humans: pick 50 production turns, run all three offline judges, run the
selector, measure agreement. Floor: top-1 agreement ≥ 0.7 with the offline
judges' weighted-reward ordering.

### 8.3 Recalibration Triggers

Re-run the calibration set on:
- Judge prompt change.
- Judge model change.
- Anchor change.
- Selector prompt or model change.

Bump `judge_prompt_version` on every change. Old scores stay valid for
historical eval runs but cannot mix with new scores in training data.

## 9. Tests

Tests are listed in execution order. Each test is the verification step for
one or more outcomes from §0.

### 9.1 Schema & Conversion (verifies §0.2)

- `test_turn_candidate_set_validates` — schema round-trips through JSON.
- `test_candidate_set_to_pairs_n2_emits_one_pair` — N=2 with non-tie produces
  exactly one `PreferencePair`.
- `test_candidate_set_to_pairs_skips_below_margin_floor` — pairs at margin
  < 0.05 are excluded.
- `test_pairs_are_pure_function_of_tcs` — running the converter twice on the
  same `TurnCandidateSet` yields identical pairs (fixed UUIDs derived from
  `candidate_set_id` + `candidate_index`).

### 9.2 Parallel Candidate Generation (verifies §0.1, §0.5)

- `test_generate_candidates_dispatches_in_parallel` — mock LLM client with
  artificial 200ms delay per call; assert `generate_candidates(n=2)`
  completes in < 350ms (would be ≥ 400ms if sequential). This is the
  load-bearing latency claim.
- `test_generate_candidates_partial_failure` — one of two coros raises;
  function returns the surviving candidate without bubbling the error.
- `test_generate_candidates_total_failure_falls_back` — both coros fail;
  caller receives a sentinel that triggers single-call fallback.
- `test_generate_candidates_respects_timeout` — slow coro is cancelled at
  the per-candidate timeout; fast coro is returned.

### 9.3 Latency Budget (verifies §0.5)

- `test_runtime_bon_latency_within_budget` — full BoN turn (generate + select
  + TTS) against a fixture LLM client measured at < 1.2× the single-call
  baseline. This is the gate that BoN doesn't ruin call latency.
- `test_selector_timeout_falls_back_to_zero` — selector mock that exceeds
  800ms; assert `chosen_index == 0` and a `selector_timeout` flag on the
  `TurnCandidateSet`.

### 9.4 Selector Behavior (verifies §0.1)

- `test_selector_picks_higher_scoring_candidate` — fixture pair where one
  candidate is obviously stronger; selector picks it.
- `test_selector_emits_per_dimension_scores` — output includes
  `selector_scores` keyed by dimension with one float per candidate.
- `test_selector_records_rationale` — non-empty `selector_rationale` string.

### 9.5 Consent Gate (verifies §0.2)

- `test_bon_disabled_for_unconsented_session` — a non-consented session
  takes the single-call path; no `TurnCandidateSet` written.
- `test_data_card_excludes_unconsented_pairs` — synthetic pairs marked
  `consent != "granted"` are not counted in `counts.in`.

### 9.6 Per-Dimension Judges (verifies §0.3)

- `test_content_judge_scores_in_unit_interval` — output ∈ [0, 1].
- `test_affect_judge_returns_audio_missing_flag_when_no_audio` — graceful
  degradation per Appendix A.
- `test_delivery_judge_runs_only_with_both_audios` — refuses to score
  without coach audio; emits `audio_missing` flag.
- `test_judges_emit_prompt_version` — every `RubricScore` carries
  `judge_prompt_version`.

### 9.7 Offline Audio Re-scoring (verifies §0.4)

- `test_rescore_synthesizes_tts_for_nonchosen` — given a stored
  `TurnCandidateSet`, re-scoring synthesizes TTS for `candidates[i]` where
  `i != chosen_index`.
- `test_rescore_updates_pair_dimensions` — after re-scoring, the derived
  `PreferencePair` includes `delivery_quality` for both sides.
- `test_rescore_is_idempotent` — running re-score twice on the same TCS
  produces the same outputs.

### 9.8 Calibration Harness (verifies §0.6)

- `test_calibration_set_loader` — loads 25 trajectories + human ratings
  from a fixture path, validates schema.
- `test_spearman_rho_computation` — given fixture (judge_scores,
  human_scores), computes ρ to spec.
- `test_data_card_marks_uncalibrated_dimensions` — when ρ < 0.6 for a
  dimension, `data_card.calibration_status[dim].passed == false` and
  `excluded_flags` includes `uncalibrated`.

### 9.9 End-to-End (verifies §0 holistically)

- `test_e2e_sandbox_rollout_emits_turn_candidate_sets` — sandbox dialogue
  with BoN enabled writes one `TurnCandidateSet` per coach turn.
- `test_e2e_run_emits_data_card` — eval run produces `data_card.json` with
  populated calibration and consent fields.
- `test_e2e_preference_pairs_are_traceable` — every pair traces back to
  a `TurnCandidateSet` ID, and that ID exists.

### 9.10 Naturalness (verifies §0.7)

- `test_naturalness_scorer_computes_interruption_rate` — fixture
  `timing.jsonl` with a known coach-overlap event; assert the computed rate
  matches.
- `test_naturalness_scorer_silence_after_affect` — fixture pairing a
  high-affect user turn (flagged) with a 3.2s coach pause; assert the band
  resolves to "ideal".
- `test_naturalness_scorer_speech_rate_band` — coach turn at 215 wpm
  resolves to `rushed`.
- `test_naturalness_scorer_handles_missing_timing` — no `timing.jsonl`;
  scorer emits zero `RubricScore`s and a `timing_missing` flag, does not
  raise.
- `test_naturalness_thresholds_versioned` — `RubricScore` carries
  `thresholds_version` for traceability across threshold changes.

### 9.11 Repetitions Runner (verifies §0.7)

- `test_runner_supports_repetitions` — `repetitions=3` on one example
  produces three `RolloutResult`s with distinct `rng_seed`s.
- `test_runner_groups_results_by_example_id` — meta-scorers receive a list
  keyed correctly when `repetitions > 1`.
- `test_runner_meta_scorer_skipped_when_repetitions_one` — `repetitions=1`
  produces results without stability rows.

### 9.12 Stability (verifies §0.7)

- `test_stability_scorer_aggregates_across_runs` — three fixture rollouts
  with known per-dimension variance produce expected stability scores.
- `test_stability_emitted_per_dimension` — separate `stability.<dim>` row
  per `delivery_quality`, `affect_perception`, `content_quality`.
- `test_stability_records_temperature` — sampling temperature is recorded
  on each stability `RubricScore`; mixing temperatures across rollouts
  emits `mixed_temperature` flag.
- `test_stability_unmeasurable_with_one_rollout` — `repetitions=1` produces
  `stability_unmeasurable` flag on each dimension's stability row.

### 9.13 Backwards Compatibility

- `test_existing_mme_sandbox_rollout_still_runs` — the v0 eval path with
  BoN disabled produces identical results.jsonl as before this spec.
- `test_old_rubric_scores_load` — `RubricScore` records without the new
  fields deserialize with defaults.

## 10. Sequencing

Seven phases. Each gates on the previous unless noted. Phases 1, 2, and 4
are independent and can run in parallel.

1. **Schema** (§6). Add `RubricScore` extensions, `TurnCandidateSet`,
   `PreferencePair`, `data_card`, `MetaScorer`, `RunConfig.repetitions`.
   Backwards-compatible. Tests 9.1, 9.13.
2. **Per-dimension offline scorers + sharpened delivery anchors**
   (§5.1–5.3, 5.7). Decompose existing `TrajectoryJudgeScorer`; bump
   `delivery_quality` prompt for expressiveness (Appendix A). No runtime
   change. Tests 9.6, 9.13.
3. **Runtime BoN with parallel generation** (§4). Selector, candidate
   generator, persistence. Default off; opt-in via flag. Tests 9.2–9.5.
4. **Naturalness + timing instrumentation** (§5.5, §6.6). Runtime and
   sandbox rollouts emit `timing.jsonl`; `NaturalnessScorer` runs on every
   eval. Cheap, deterministic, no calibration burden. Tests 9.10, 9.13.
   Independent of phase 3 — can land in parallel.
5. **Calibration harness + 25-trajectory set** (§8). Voice-recorded
   ratings via the rating UI, parsed to scalars. Tests 9.8.
6. **Offline audio re-scoring** (§5.4) + **preference pair generation**
   (§7). Tests 9.7, 9.9. Gated on at least `delivery_quality` passing the
   calibration floor.
7. **Stability via repetitions** (§5.6, §6.5). Runs nightly only on a
   10-trajectory subset, `repetitions=5`. Not PR-gating. Tests 9.11, 9.12.

## 11. Open Questions

1. **Selector model.** Haiku is the cheap default. Worth measuring against
   a tiny self-hosted model before locking in.
2. **Sample rate.** BoN on every turn or every Kth turn? Cost vs data-rate
   tradeoff. Start every-turn behind a flag; revisit at scale.
3. **Margin floor value.** 0.05 is a guess. Tune empirically once we have a
   week of pairs.
4. **What if the runtime selector is wrong but the offline judges are
   right?** The user heard the wrong candidate, but we still get a clean
   training pair (chosen=offline-correct). Worth tracking divergence rate
   between selector and offline judges as a quality KPI.
5. **TTS for non-chosen candidates** — cheaper to batch overnight or
   stream as calls end? Affects re-scoring latency, not user latency.
6. **Naturalness thresholds.** The bands in §3.2 are defensible defaults,
   not validated. Pin v1, then revisit after a week of production data —
   in particular `silence_after_affect`, where ideal duration probably
   varies by affect type (grief vs frustration vs joy).
7. **Stability sample size.** M=5 nightly runs is a guess. M=3 is cheaper
   and may suffice for catching gross instability; M=10 is the rigorous
   answer. Tune empirically.
8. **Should expressiveness split out from `delivery_quality`?** The spec
   keeps it folded in for v0. If calibration shows judges can't reliably
   distinguish "tonal match" from "expressive range," split into
   `prosodic_attunement` and `prosodic_expressiveness` and recalibrate.

## 12. Acceptance Criteria

This spec is satisfied when:

1. `RubricScore` carries the new fields and old artifacts still load.
2. The three per-dimension offline judges exist and each produce scores in
   [0,1] with `judge_prompt_version` populated.
3. Runtime BoN is implemented with `asyncio.gather` parallel candidate
   generation; latency overhead vs single-call ≤ 1.2× on the fixture
   benchmark in test 9.3.
4. `TurnCandidateSet` is persisted per coach turn on consented sessions
   and the consent gate test (9.5) passes.
5. `turn_candidate_set_to_pairs` is a pure function and round-trips per
   test 9.1.
6. Offline audio re-scoring updates non-chosen candidates with
   `delivery_quality` per test 9.7.
7. Calibration harness exists, the 25-trajectory set has been rated, and
   `data_card.json` reflects per-dimension calibration status per test 9.8.
8. End-to-end sandbox run with BoN on emits valid `TurnCandidateSet` +
   `PreferencePair` + `data_card` artifacts (test 9.9).
9. Runtime and sandbox rollouts emit `timing.jsonl`; `NaturalnessScorer`
   runs on every eval and emits `interruption_rate`,
   `silence_after_affect`, and `speech_rate_band` `RubricScore` rows with
   `thresholds_version` populated (tests 9.10).
10. `RunConfig.repetitions > 1` produces grouped `RolloutResult`s and the
    `MetaScorer` protocol delivers them to `StabilityScorer`, which emits
    one `stability.<dim>` row per dimension per example (tests 9.11, 9.12).
11. `delivery_quality` judge prompt has been bumped to interrogate
    expressiveness directly per Appendix A; calibration ρ is reported on
    the new prompt version.

## 13. Manifest Update

Add to `docs/specs/MANIFEST.md`:

| Spec | Status | Policy | Applies to | Notes |
|---|---|---|---|---|
| `v2026-05-05-multimodal-trajectory-rubric-rlaif.md` | `acknowledged` | `amendment` | Eval rubric + runtime BoN + RLAIF data shape | Amends §6.3, §7 of the 04-29 spec. Introduces runtime preference collection. |

---

## Appendix A — Rubric Anchors and Degradation

### Anchors

**`affect_perception`**
- 1.0 — Coach demonstrably reads the user's affect (named, mirrored, or
  acted on) and updates as the user shifts.
- 0.5 — Generic empathy; not grounded in the user's actual state.
- 0.0 — Misreads, escalates, dismisses, or ignores affect.

**`delivery_quality`** (covers attunement *and* expressiveness)
- 1.0 — Coach prosody has expressive range that meets the moment: warmer
  on disclosure, grounded on escalation, animated when celebrating. Pitch,
  rate, energy, and pause all track emotional shifts. Holds space.
- 0.5 — Audible delivery, flat range. Neutral, not jarring, not
  load-bearing. Doesn't make things worse, doesn't help.
- 0.0 — Monotone, tonally miscalibrated (cheery on grief, flat on joy),
  rushed, talks over the user, or fills every silence.

**`content_quality`** — inherits 04-29 §7.2 anchors for
`coaching_trajectory_quality`.

### Naturalness sub-metric thresholds (v1)

Pinned. Versioned via `thresholds_version`. Banded values reported as
0.0/0.5/1.0 against these bands.

| Sub-metric | Ideal (1.0) | Acceptable (0.5) | Pathological (0.0) |
|---|---|---|---|
| `interruption_rate` | 0.0 events / user turn | ≤ 0.2 | > 0.5 |
| `silence_after_affect` | 1.5 – 4.0 s | 1.0 – 1.5 s or 4.0 – 6.0 s | < 1.0 s or > 6.0 s |
| `speech_rate_band` | 130 – 170 wpm | 100 – 130 or 170 – 200 wpm | < 100 or > 200 wpm |

Notes: `interruption` requires ≥ 250 ms of overlap (excludes brief
backchannels like "mhm"). `silence_after_affect` is measured only on user
turns flagged high-affect by `AffectPerceptionJudgeScorer`.

### Degradation

| Missing | Effect |
|---|---|
| Coach audio | `delivery_quality` not emitted; weights renormalize; `partial_modality` flag. |
| User audio | `affect_perception` falls back to text-only; `audio_missing` flag. |
| `timing.jsonl` | `NaturalnessScorer` emits zero rows + `timing_missing` flag; trajectory still scored on §3.1 dimensions. |
| Both audio + timing | Trajectory excluded from training data by default (`min_modalities` filter); still surfaced on dashboards. |
| `repetitions == 1` | Stability not computable; `StabilityScorer` emits one row per dimension with `score=null` and `stability_unmeasurable` flag. |
