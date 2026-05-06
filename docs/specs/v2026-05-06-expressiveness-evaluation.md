# rehearse — Expressiveness Evaluation (Vibe Check)

**Status**: draft (options) — not yet a decided implementation plan.
**Owner**: jz
**Depends on**:
- `docs/specs/v2026-05-05-multimodal-trajectory-rubric-rlaif.md` (defines
  `delivery_quality` and the calibration harness this spec extends)
- `docs/specs/v2026-05-06-eval-system-roadmap.md` (Mini-spec 3 — calibration +
  voice rating UI; this spec lives downstream of it)

---

## 1. Why this needs its own spec

The 05-05 rubric folds expressiveness into `delivery_quality` and assumes the
audio judge can score it directly. Three properties of expressiveness make
that assumption fragile, in a way that's distinct from the other rubric
dimensions:

1. **Automated metrics don't correlate with human judgment.** Pitch
   variance, energy, F0 std-dev — none of them predict whether a listener
   thinks the response *felt right*. Prior work on TTS naturalness MOS,
   on emotional speech synthesis, and our own informal listening sessions
   all converge on the same finding: expressiveness is the dimension where
   surrogate metrics fail hardest. Audio LLM judges may inherit the same
   ceiling — they're trained on data labelled by the same automated
   metrics that don't correlate.
2. **It is multi-dimensional and the dimensions interact.** Tone, pacing,
   and emphasis aren't independent: a well-emphasized phrase delivered at
   the wrong pace registers as worse than a flat phrase at the right pace.
   A scalar `delivery_quality` score collapses this in ways that hide
   real failure modes.
3. **Judgment is context-anchored.** The same audio output rated against
   "user just disclosed something hard" reads as warm and grounded;
   against "user is celebrating" it reads as flat and missing the moment.
   Raters must hear or see the prior turn before they can score the next.
4. **Preference is industry- and situation-conditioned.** A grief
   counselor and a sales coach both want "expressive," but in opposite
   directions. Aggregating ratings across scenarios produces a regression
   to a bland mean that doesn't represent any real user.

Together these mean: human evaluation is load-bearing here, and the
calibration harness in 05-06 Mini-spec 3 (a single-rater 0–1 voice rating
on each dimension) is too coarse to carry it. This spec proposes options
for what the expressiveness measure should actually look like.

## 2. What "the measure" must produce

Whichever option we pick, the measure must produce, per coach turn on a
scored trajectory:

- A scalar (or vector) score consumable by the existing `RubricScore` /
  `AggregateScorer` pipeline.
- A calibration story: how the score relates to human preference, and at
  what sample size we trust it.
- A story for *industry conditioning*: either the score is conditional on
  scenario metadata, or we explicitly accept a globally-averaged signal
  and document the loss.
- An audit trail: which humans (or judge versions) produced the rating,
  on what context, with what reference material.

Non-goals: this spec does not propose a new model. It proposes a
measurement protocol whose output drives the existing RLAIF data flow.

## 3. Options

Five options, ordered roughly cheapest → most rigorous. They are not
mutually exclusive — a viable path stacks two or three.

### Option A — Pairwise audio preference UI

Replace (or supplement) the 0–1 voice rating with a forced-choice
**A vs B** judgment over two coach audio renderings of the same prior
context. Rater hears the prior turns once, then A and B back-to-back, and
picks one. Scalar score is derived via Bradley-Terry or Elo over the
collected pairs.

- **Strengths**: forced choice eliminates anchor drift between raters,
  reproduces the actual decision the runtime selector makes, requires no
  numeric calibration, integrates directly with BoN candidate sets (every
  consented BoN turn is already a natural pair).
- **Weaknesses**: one number, no decomposition into tone / pacing /
  emphasis. Doesn't say *why* A beat B. Needs N pairs per scenario for
  Bradley-Terry to converge — sample-hungry.
- **Cost**: low (rater time per pair ≈ 30s). Reuses the `viewer.py` rating
  routes from Mini-spec 3 with a different render template.
- **Calibration story**: judge ρ is computed against the Bradley-Terry
  posterior, not raw scalars. Calibration gate becomes "judge picks the
  same winner as the human ≥ X% of the time on held-out pairs."

### Option B — Multi-dimensional rubric with anchored audio examples

Keep the multi-dimensional rubric (tone, pacing, emphasis — and possibly
warmth / grounding / animation as separate sub-dimensions). For each
dimension, pre-record a small set of **anchor clips** at score 0.0, 0.5,
and 1.0. Rater hears the candidate, then the three anchors for the
dimension being scored, and picks the closest. Score is the anchor's
value.

- **Strengths**: decomposes the signal — when expressiveness regresses
  we can see whether tone, pacing, or emphasis moved. Anchor-relative
  judgment dramatically reduces inter-rater drift compared to
  open-ended 0–1 sliders.
- **Weaknesses**: anchors must themselves be vetted and may need to be
  scenario-specific (a "grounded" anchor for grief coaching is different
  from one for sales coaching). Anchor maintenance is real ongoing work.
- **Cost**: medium upfront (recording anchors, validating them); medium
  per-rating (rater hears 1 candidate + 3 anchors per dimension × N
  dimensions).
- **Calibration story**: Spearman ρ per dimension, same as 05-05 — but
  now ρ is computed on a sharper signal because the human input is
  anchor-relative.

### Option C — Context-anchored rating with mandatory prior-turn playback

Cross-cutting with A or B: enforce that the rating UI **never** shows a
candidate audio without first playing the prior coach + user turns. Add
a structured pre-rating prompt: "Before rating, write one sentence about
what you'd want the coach's next turn to feel like." This forces the
rater to commit to an expectation before judging — preventing post-hoc
rationalization and giving us a free signal to sanity-check rater
attention.

- **Strengths**: addresses the context-anchoring problem head-on. Cheap
  to add to whichever rating mode we pick.
- **Weaknesses**: increases rating time per item ~2×. Some raters will
  resent the typed pre-prompt — needs to be very lightweight.
- **Cost**: low (UI work only).
- **Calibration story**: same as A or B; this is a quality lever on the
  human signal, not a different measure.

### Option D — Industry-conditioned preference panel + bucketed scores

Define explicit scenario buckets (e.g. `cold-outreach-sales`,
`grief-support`, `technical-1on1`, `escalation-deescalation`). Recruit
a small panel of raters with domain familiarity per bucket. Every rating
carries the bucket; aggregation and judge calibration are computed
**within** bucket. The reported expressiveness score is a vector indexed
by bucket, not a scalar. Aggregate scorers reduce to a scalar only when
producing training data, using the bucket distribution of the target
session.

- **Strengths**: directly addresses preference-conditioning. Prevents
  the bland-mean failure. Makes it possible to ship a model that's
  excellent at one industry without first being mediocre at all of them.
- **Weaknesses**: requires panel recruiting (real cost, real ongoing
  ops). Buckets must be enumerated and maintained; mis-bucketing
  silently corrupts data. Sample size per bucket is the binding
  constraint — 25 trajectories total becomes 25 / N_buckets per bucket.
- **Cost**: high (recruiting + ongoing rater management).
- **Calibration story**: per-bucket ρ. Judges that pass globally may
  fail on specific buckets; surfaced in `data_card.json`.

### Option E — Layered LLM-judge with disagreement-triggered human triage

Run the audio LLM judge on **every** turn, but treat its score as
provisional. Compute a confidence signal — either via re-roll variance
(score the same turn N times with judge temperature > 0) or via an
explicit confidence head — and route only the **uncertain** turns to
human rating. Humans rate via Option A or B. Judge gets re-trained or
re-prompted periodically against the human verdicts, with provenance
recorded.

- **Strengths**: scales human cost roughly with the *interesting* slice
  of data, not the full dataset. Builds a virtuous loop: as the judge
  improves, fewer turns need human review.
- **Weaknesses**: failure mode is silent — if the judge is *confidently
  wrong* in a systematic way (e.g. always rates monotone neutral as
  "fine"), it never escalates and we never see the drift. Requires a
  disagreement-detection mechanism that is itself non-trivial.
- **Cost**: low ongoing once the loop is built; medium upfront to build
  the triage and re-training pipeline.
- **Calibration story**: ρ on the held-out human-rated subset, recomputed
  on each judge revision. Plus a separately-tracked **systematic-error
  audit** — sample a small fraction of *non-flagged* turns each week for
  human rating, to detect confident-wrong drift.

## 4. Provisional recommendation (open for revision)

Stack **A + C + E**, with **D** as a follow-on:

- **A** (pairwise audio) is the load-bearing primitive. It directly maps
  to BoN candidate selection, which is where expressiveness preference
  data already exists for free. The `AggregateScorer` pipeline can
  consume Bradley-Terry scalars without schema change.
- **C** (mandatory prior-turn playback + pre-rating prompt) is a near-zero
  cost quality multiplier on every human rating we collect. Add it
  unconditionally.
- **E** (disagreement triage) is how we keep human cost bounded as
  volume grows. Defer until we have enough human-labelled pairs from
  A to bootstrap the disagreement signal.
- **D** (bucketing) is a strict superset of A — defer until we have a
  working A pipeline, then add a `scenario_bucket` field and refit
  per-bucket. Don't pay panel-recruiting cost until we know the
  measurement primitive works.

**B** (anchored multi-dimensional rubric) is attractive but expensive to
maintain and conflicts with A's "one judgment per pair" simplicity. The
question it answers — "*why* did expressiveness regress?" — is real but
likely better answered by qualitative spot-listening on regressions than
by a permanent N-dimensional rating burden.

## 5. Open questions (for you, before this becomes an implementation spec)

1. **Is the right primary signal pairwise (A) or scalar+anchored (B)?**
   This is the largest fork. Pairwise is cheaper and matches BoN
   naturally; scalar gives interpretable per-dimension regressions.
2. **Bucketing now or later?** D is real ongoing work. Are there
   already named scenarios in the product that map to buckets, or is
   bucketing premature?
3. **Who are the humans?** jz alone is the calibration set in Mini-spec 3.
   Does this spec assume the same — or do we need a recruited panel
   from day one for industry coverage? (D presupposes the latter.)
4. **What's the v1 sample size we can actually rate per week?** This
   sets the binding constraint on every option. Bradley-Terry needs
   ~N log N pairs to converge per bucket; if the budget is 50
   ratings/week and we want 4 buckets, D is gated by months of data
   collection before per-bucket ρ is meaningful.
5. **Do we accept a scalar from the start, or is a vector score
   acceptable to the downstream pipeline?** `RubricScore` is per-
   dimension already; adding more dimensions is cheap. But
   `AggregateScorer` weights need values — if expressiveness becomes a
   vector, the weights schema changes.
6. **Failure mode of E ("confidently wrong" drift)**: are we OK with a
   weekly random-sample audit as the safety net, or does that itself
   need human bandwidth we don't have?

## 6. Manifest update (when this graduates from draft)

Add to `docs/specs/MANIFEST.md`:

| Spec | Status | Policy | Applies to | Notes |
|---|---|---|---|---|
| `v2026-05-06-expressiveness-evaluation.md` | `draft` | `amendment` | Expressiveness measurement under `delivery_quality` | Options-stage spec. Decomposes the vibe-check problem the 05-05 rubric folds into a single audio-judge dimension. |
