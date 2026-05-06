# rehearse — Spec: Multimodal Trajectory Rubric + RLAIF Pipeline

**Status**: draft (design-facing)
**Owner**: jz
**Depends on**:
- `docs/specs/v2026-04-27-eval-harness.md`
- `docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md`
- `docs/specs/v2026-04-29-mme-seeded-rl-sandbox-eval.md`
- `docs/specs/v2026-05-01-consent-and-outcome-capture.md`

**Amends**: §6.3, §7 of `v2026-04-29-mme-seeded-rl-sandbox-eval.md` (rubric and
scorer output). The dataset, environment, and rollout artifact contracts from
the 04-29 spec remain authoritative.

---

## 0. One-line Summary

Replace the text-only two-dimension rubric (`emotion_responsiveness`,
`coaching_trajectory_quality`) with a four-dimension multimodal rubric scored
over both transcript and audio, and define the data shapes that pipeline
rollout judgments into RLAIF — both pointwise reward modeling and pairwise
preference learning.

## 1. Why This Exists

The current rubric has a load-bearing flaw: production runtime is multimodal
(audio in, audio out, prosody on both sides) but the rubric reads only the
transcript. This is train-serve skew at the reward-model layer. An RL signal
trained on the existing rubric optimizes for *saying* emotionally intelligent
things, not *being* emotionally attuned. The judge can't tell whether "I hear
you" was delivered warmly, flatly, or over the user still speaking — so the
policy can't either.

Three concrete failures the current rubric cannot detect:

1. **Tonal mismatch.** Coach says the right words in a flat or rushed delivery.
2. **Turn-taking violations.** Coach interrupts, leaves dead air, or rushes the
   user through escalated affect.
3. **Prosodic non-attunement.** Coach prosody stays neutral while the user's
   affect shifts; or coach amplifies arousal when the user needs grounding.

These failures are the difference between a coach who *sounds* like they get
it and one who *does*. The rubric must distinguish them, or the RL loop will
collapse the distinction.

## 2. Scope

### In Scope

- A four-dimension rubric (§4) that separates content from delivery.
- An audio-aware scoring pipeline (§5) that runs alongside the existing text
  judge and emits per-dimension `RubricScore` rows.
- Schema extensions to `RubricScore` for modality and segment provenance (§7).
- A `PreferencePair` artifact for pairwise RLAIF (§8).
- A reward-shape contract — dense (per-segment) and terminal (per-trajectory)
  — suitable for both reward modeling and direct preference optimization (§8).
- Compatibility with both sandbox rollouts and production-replay rollouts (§9).
- Human calibration protocol for the audio-aware dimensions (§10).

### Out of Scope

- Choice of RL algorithm (PPO / GRPO / DPO / IPO). The spec defines the data
  the trainer consumes; algorithm selection is downstream.
- Online RL or any policy update loop.
- Replacement of the sandbox environment. Sandbox audio fidelity is its own
  workstream (see §9.1).
- Production-replay environment implementation. This spec defines the rubric
  contract that environment must satisfy; the environment itself is a separate
  doc.
- Real-time feedback to a live caller.

## 3. Why a New Spec, Not an Edit to 04-29

The 04-29 spec stands for the *rollout* and *dataset* contracts. Those are
correct. What needs to change is downstream of the rollout: the rubric, the
scorer protocol, and the artifacts the trainer consumes. Keeping those changes
in a new spec lets 04-29 remain the source of truth for sandbox topology and
keeps the RLAIF data contract reviewable in one place.

## 4. Rubric

### 4.1 Design Principle

Each dimension must be (a) independently observable, (b) cheap-or-tractable to
score, and (c) actionable for an RL policy — meaning a policy update could
plausibly improve it without destroying another. Collapsing dimensions makes
the reward model load-bearing in ways the policy can't disentangle.

### 4.2 Dimensions

| Dimension | Modality | Score Range | Question |
|---|---|---|---|
| `affect_perception` | audio in (user) + text | 0.0–1.0 | Did the coach correctly read what the user was feeling? |
| `prosodic_attunement` | audio in (user) + audio out (coach) | 0.0–1.0 | Did the coach's *delivery* match what the moment needed? |
| `content_quality` | text | 0.0–1.0 | Was what the coach *said* useful, safe, and trajectory-positive? |
| `turn_dynamics` | timing + audio | 0.0–1.0 | Pacing, interruptions, silence — did the coach hold space well? |

`weighted_reward` is the trajectory-level aggregate (§4.4).

### 4.3 Anchors

Anchors are written for the audio-aware dimensions because those are new.
`content_quality` inherits the `coaching_trajectory_quality` anchors from
04-29 §7.2.

**`affect_perception`**

- **1.0** — Coach demonstrably reads the user's affect (named, mirrored, or
  acted on) and updates as the user shifts. Evidence is in both word choice
  and the moments the coach chose to slow, soften, or probe.
- **0.5** — Generic empathy. Acknowledges feelings in the abstract; not
  clearly grounded in the user's actual state. Doesn't update on shifts.
- **0.0** — Misreads, escalates, dismisses, or ignores the affect.

**`prosodic_attunement`**

- **1.0** — Coach prosody (pitch range, speaking rate, energy, pause length)
  meets the moment: grounded when the user is escalated, warmer when the user
  is flat, paced down when the user is rushed.
- **0.5** — Neutral delivery. Not jarring, not load-bearing.
- **0.0** — Tonally wrong: cheery when user is grieving, rushed when user is
  spiraling, flat when user needs warmth.

**`turn_dynamics`**

- **1.0** — No interruptions when the user is mid-thought; comfortable with
  silence after emotional disclosures; doesn't rush escalated affect.
- **0.5** — A few minor pacing issues; nothing damaging.
- **0.0** — Talks over the user, fills every silence, or rushes through
  high-affect moments.

### 4.4 Weights

Default weights (subject to calibration in §10):

```
affect_perception      0.30
prosodic_attunement    0.25
content_quality        0.30
turn_dynamics          0.15
```

Weights live in the eval payload's `rubric_weights` and override the default,
matching the existing 04-29 mechanism. Weights are documented per eval and
versioned with the eval.

`weighted_reward = Σ w_i * score_i`.

### 4.5 Degradation Rules

Not every rollout has every modality. Rules:

- **No coach audio** (e.g. text-only sandbox path, or replay where TTS wasn't
  captured): `prosodic_attunement` is *not* emitted; weights renormalize over
  remaining dimensions; `weighted_reward` carries a `partial_modality` flag.
- **No user audio** (e.g. transcript-only production replay): `affect_perception`
  drops to a text-only inference and is flagged `audio_missing`.
- **No timing data**: `turn_dynamics` is not emitted; flagged.

A trajectory that loses two or more dimensions is excluded from RLAIF training
data by default (a `min_modalities` filter in the trainer config), but is
still surfaced in eval dashboards.

## 5. Scoring Pipeline

### 5.1 Topology

```
RolloutResult
  ├── transcript.jsonl              ──► text judge (Claude)            ─► RubricScore[content_quality]
  ├── audio/coach/turn_*.wav        ──┐
  ├── audio/user/turn_*.wav         ──┼► audio judge (Gemini 2.5)      ─► RubricScore[affect_perception,
  └── timing.jsonl                  ──┘                                                prosodic_attunement,
                                                                                       turn_dynamics]
                                                                       ─► aggregator   ─► RubricScore[weighted_reward]
                                                                                       ─► PreferencePair (when paired)
```

### 5.2 Scorer Decomposition

Three scorers, composed in the eval's `scoring_plan()`:

1. **`ContentJudgeScorer`** — text judge, scores `content_quality`. Direct
   evolution of today's `TrajectoryJudgeScorer` with the prompt narrowed to
   *what was said*, not *how it landed*.
2. **`AudioJudgeScorer`** — audio judge, scores `affect_perception`,
   `prosodic_attunement`, and `turn_dynamics`. Reads aligned per-turn audio
   segments (coach + user) and the transcript for grounding.
3. **`AggregateScorer`** — pure function over the per-dimension scores.
   Emits `weighted_reward` and writes `judge.json` with provenance.

Decomposition rationale: text-only rollouts can run scorer 1 alone and still
land on the dashboard. Cost control: the audio judge is the expensive call;
keeping it isolated lets us cache, batch, and ablate.

### 5.3 Per-Segment vs Per-Trajectory

The audio judge emits *both*:

- **Per-segment scores** at the level of conversational turns (or smaller
  windows for prosody). Segments are addressed by `(turn_index, t_start_ms,
  t_end_ms)`. These become the dense reward signal.
- **Per-trajectory scores** rolled up per dimension. These become the
  terminal reward and the comparison target for preference pairs.

Both are persisted; the trainer chooses which to consume.

### 5.4 Judge Prompts

Judge prompts are versioned (`judge_prompt_version` on every `RubricScore`)
and live alongside the scorer code. A prompt change requires a version bump
and re-running calibration before scores from old and new prompts can be
mixed in training data.

### 5.5 Judge Self-Consistency

Each audio-judge call returns a confidence ∈ [0, 1] per dimension, plus key
moments (segment indices that drove the score). Trajectories below a
configurable confidence floor are routed to a human review queue rather than
into RLAIF training data.

## 6. Inputs the Pipeline Requires

This spec assumes the following are produced by the rollout — environment
implementations are responsible for them:

- `transcript.jsonl` — already produced (04-29 §6.1).
- `audio/coach/turn_<N>.wav` — coach TTS output, per turn.
- `audio/user/turn_<N>.wav` — user audio (synthesized in sandbox; real in
  production replay, gated on consent per 05-01).
- `timing.jsonl` — per-event records `{turn_index, role, t_start_ms,
  t_end_ms, t_first_audio_ms}` with enough resolution to compute interruption
  and silence metrics.

If a rollout cannot produce a field, it omits it; the degradation rules in
§4.5 apply.

## 7. Schema Changes

### 7.1 `RubricScore` extensions

Add fields (all optional, all defaulting in a way that keeps existing
artifacts valid):

```python
class RubricScore(BaseModel):
    # existing fields unchanged
    ...
    modality: Literal["text", "audio", "audio+text", "timing", "aggregate"] = "text"
    segment: Segment | None = None     # None = trajectory-level
    confidence: float | None = None    # judge-reported, 0..1
    judge_prompt_version: str | None = None
    flags: list[str] = []              # e.g. ["partial_modality", "audio_missing"]

class Segment(BaseModel):
    turn_index: int
    t_start_ms: int
    t_end_ms: int
```

### 7.2 New artifact: `PreferencePair`

Persisted at `evals/runs/{run_id}/preference_pairs.jsonl`, one per line:

```python
class PreferencePair(BaseModel):
    pair_id: str
    example_id: str               # the dataset row both rollouts used
    chosen_rollout_id: str
    rejected_rollout_id: str
    dimension: str                # which dimension drove the preference,
                                  # or "weighted_reward" for aggregate
    margin: float                 # |chosen - rejected| at trajectory level
    chosen_scores: dict[str, float]
    rejected_scores: dict[str, float]
    judge_rationale: str
    judge_prompt_version: str
    created_at: datetime
```

Pairs are produced by either: (a) running the same example through two
different policies and taking the rubric's verdict, or (b) running the same
example through the same policy at higher temperature and taking the
rubric's verdict. Both are valid RLAIF data sources.

### 7.3 New artifact: `judge.json` (replaces the existing one)

Per rollout, persists the full audio + text judge output: per-segment scores,
key moments, rationales, confidences, and prompt versions. This is the
replayable record — given `judge.json` and the audio, a human can audit any
score.

## 8. RLAIF Data Contract

The trainer consumes one of two shapes from a finished eval run:

### 8.1 Pointwise (Reward Modeling)

```
(prompt_state, trajectory, scalar_reward, per_dimension_scores)
```

Sourced from `results.jsonl` joined with the `transcript.jsonl` and audio
manifest for the rollout. Suitable for training a reward model (Bradley-Terry
or scalar regression) that a policy can later optimize against.

### 8.2 Pairwise (Direct Preference)

```
(prompt_state, chosen_trajectory, rejected_trajectory, dimension, margin)
```

Sourced from `preference_pairs.jsonl`. Suitable for DPO/IPO-style direct
preference optimization without an explicit reward model.

The trainer chooses; the eval harness produces both.

### 8.3 Filters

A `data_card.json` is emitted with each run summarizing what is safe to train
on:

- `min_confidence` floor applied
- `min_modalities` floor applied
- `excluded_flags` (e.g. trajectories flagged `audio_missing`)
- consent status (production replay only; gated per 05-01)
- counts in vs counts out

The trainer is required to record which `data_card` it consumed, so any
training artifact can be traced back to a known-quality slice.

## 9. Compatibility With Both Rollout Sources

### 9.1 Sandbox Rollouts

The current sandbox is text-only (LLM-vs-LLM). To produce the audio inputs
this rubric requires, sandbox needs:

- **Coach TTS** through `rehearse/eval/tts_bridge.py` — already exists; needs
  to be wired into the sandbox loop and persist per-turn `wav` files.
- **User audio** — three options, sequenced:
  1. **MME clip** for the opening turn only (already grounded, real human
     affect). Subsequent user turns: TTS-synthesized from the customer agent
     transcript.
  2. **TTS for all user turns** — cheaper, less realistic, sufficient for
     dimensions that score *the coach's response* rather than user affect
     fidelity.
  3. **Held-out human-recorded customer turns** — most realistic, most
     expensive, longest tail. Defer.

Sandbox audio fidelity is its own workstream; this spec just declares what
the rubric needs.

### 9.2 Production Replay

A production-replay environment (separate spec) emits a `RolloutResult`
pointing at the on-disk artifacts of a completed phone call. It must produce
the same audio + timing files §6 requires, sourced from the live runtime.
The runtime already writes `transcript.jsonl` (`rehearse/writers/artifacts.py`);
the audio and timing writers are the gap.

Consent is non-negotiable: production-replay scoring runs only on sessions
where `Session.consent == "granted"` per the 05-01 spec. The `data_card`
records this.

## 10. Calibration

The audio-aware dimensions are new and the judge is new. Before any of these
scores enter training data, we calibrate against humans.

### 10.1 Calibration Set

- 50 trajectories, sampled across the affect distribution of the MME seeds.
- Two human raters per trajectory, scoring all four dimensions on the same
  anchors as the judge.
- Inter-rater agreement (Krippendorff's α or Cohen's κ on banded scores)
  must clear a floor before the set is used to score the judge.

### 10.2 Judge Acceptance Floor

Per dimension, judge-vs-human Spearman ρ ≥ 0.6 on the calibration set before
that dimension's scores may be admitted to RLAIF training data. Dimensions
that miss the floor are still emitted and dashboarded — flagged
`uncalibrated` in the data_card and excluded by the trainer.

### 10.3 Recalibration Triggers

- Judge prompt version bump.
- Judge model change.
- Anchor change.

Re-run the calibration set; re-evaluate the floor; bump the data_card schema
version.

## 11. Sequencing

Order matters; each step is gated on the previous.

1. **Schema** (§7). Land `RubricScore` extensions and the `PreferencePair`
   type. Backwards-compatible defaults — no behavior change.
2. **Decompose the existing scorer.** Split `TrajectoryJudgeScorer` into
   `ContentJudgeScorer` + `AggregateScorer`. Rubric still text-only; new
   shape, same scores. Validates the decomposition without taking on audio
   risk.
3. **`AudioJudgeScorer` v0** — single dimension, `prosodic_attunement`. Run
   on sandbox rollouts that have coach TTS captured. Prove the signal is
   non-degenerate before adding dimensions.
4. **Sandbox audio coverage** — wire TTS through user turns; produce timing.
5. **`AudioJudgeScorer` v1** — add `affect_perception` and `turn_dynamics`.
6. **Calibration set** (§10). Until this exists, no audio scores enter
   training data.
7. **`PreferencePair` generation** — once at least one dimension is past the
   acceptance floor, start producing pairs.
8. **Production-replay environment** (separate spec). Gated on 05-01 consent
   capture being live.
9. **Trainer integration.** Out of scope here.

## 12. Open Questions

1. **Audio judge model.** Gemini 2.5 (multimodal in, text out) is the working
   assumption. Worth spiking GPT-4o audio and a self-hosted vLLM audio model
   before locking in. Cost-per-trajectory at expected eval volume is the
   deciding factor.
2. **Per-segment granularity for prosody.** Per-turn is the obvious unit, but
   prosody often shifts mid-turn. Sub-turn windows (e.g. 2-second slices) are
   more faithful, more expensive, and harder to align with transcript turns.
   Start per-turn; revisit if signal is too coarse.
3. **Pair generation strategy.** Same-policy-different-temperature pairs are
   cheap and immediately available; cross-policy pairs require running two
   policies through the same dataset row. Both are RLAIF-valid; the question
   is which produces a stronger preference signal at our scale.
4. **Margin floor for preference pairs.** Some `chosen`/`rejected` pairs will
   be near-ties. Recording all pairs is fine; the trainer probably wants a
   minimum margin (e.g. 0.1 on trajectory aggregate) to avoid noise. Decide
   in the trainer spec, not here.
5. **Reward model vs DPO.** Out of scope per §2, but the choice influences
   whether the eval should optimize for high-quality pointwise scores or
   high-quality preference pairs. Currently producing both; revisit if we
   need to specialize.
6. **Should `weighted_reward` be the only reward signal at trajectory level?**
   A multi-objective trainer could consume the per-dimension vector instead
   and learn its own scalarization. The schema already supports this; the
   default contract assumes scalar.
7. **Backfill of historical sandbox runs.** Existing `mme-sandbox-rollout`
   runs were scored on the old rubric. They have no audio. Treat them as
   `partial_modality` and let the data_card filter them out, rather than
   trying to reconstruct.

## 13. Acceptance Criteria

This spec is satisfied when:

1. `RubricScore` carries modality, segment, confidence, prompt-version, and
   flags fields and existing artifacts still load.
2. `ContentJudgeScorer` and `AggregateScorer` exist and reproduce today's
   `mme-sandbox-rollout` numbers within rounding on the existing dataset.
3. `AudioJudgeScorer` emits at least `prosodic_attunement` against a sandbox
   rollout that has coach TTS captured.
4. `preference_pairs.jsonl` is generated for any run where at least two
   rollouts share an `example_id`.
5. `data_card.json` is emitted per run and records modality coverage,
   confidence floor, consent status, and calibration status per dimension.
6. The calibration set exists, has been scored by two humans, and the
   judge-vs-human Spearman ρ has been computed for every dimension that ever
   enters training data.

## 14. Manifest Update

Add to `docs/specs/MANIFEST.md`:

| Spec | Status | Policy | Applies to | Notes |
|---|---|---|---|---|
| `v2026-05-05-multimodal-trajectory-rubric-rlaif.md` | `acknowledged` | `amendment` | Eval rubric + RLAIF data shape | Amends §6.3, §7 of the 04-29 spec. |

Workstream Map row: replace the RL-style sandbox eval row's spec list with
`v2026-04-27-eval-harness.md`, `v2026-04-29-mme-seeded-rl-sandbox-eval.md`,
`v2026-05-05-multimodal-trajectory-rubric-rlaif.md`.
