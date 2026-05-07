# Mini-spec 3 — Calibration Harness + Voice Rating UI

**Status:** draft
**Date:** 2026-05-06
**Owner:** Josh Zastrow
**Decomposes from:** [`v2026-05-06-eval-system-roadmap.md`](v2026-05-06-eval-system-roadmap.md) §7 (Wave C)
**Related code:** `rehearse/eval/scorers/`, `rehearse/eval/runner.py`, `rehearse/viewer.py`, `evals/golden/`

## Goal

Make the audio + content + naturalness judge scores admissible to training
data. Today every judge emits a number between 0 and 1, but we have no
evidence that those numbers track human judgment, so they cannot safely be
used as RL reward signal or as a regression gate. This spec ships the
calibration substrate: humans rate a small golden set of trajectories, the
harness computes per-dimension Spearman ρ vs the judges, and a 0.6 threshold
gates each dimension's downstream use.

The roadmap is explicit that **no judge's scores are admitted to training
data until ρ ≥ 0.6 on its dimension**. Mini-specs 7 (preference pairs) and 8
(stability) depend on this. Mini-spec 6 (BoN selector) depends on this for
the selector's own calibration.

## Non-goals

- Calibrating the *selector* (Mini-spec 6). The selector reuses this same UI
  for its own ρ measurement, but selector-side wiring is out of scope here.
- Active learning loops, rater disagreement adjudication, or inter-rater
  reliability metrics. Single rater (jz) is sufficient at v0.
- A cloud-hosted multi-rater console. Storage is local JSON; viewer routes
  serve a single rater on `localhost`.
- Re-rating cadence or drift detection. We rate once, gate at 0.6, and move
  on. If a judge's prompt changes (`judge_prompt_version` bumps), its prior
  ratings are dropped and the dimension is re-rated.

## Background

The current judge stack:

| Scorer | Dimension | Modality |
|---|---|---|
| `ContentJudgeScorer` | `content_quality` | text |
| `AffectPerceptionJudgeScorer` | `affect_perception` | audio-in + text |
| `DeliveryJudgeScorer` | `delivery_quality` | audio-in + audio-out |
| `NaturalnessScorer` | `naturalness.{interruption_rate,silence_after_affect,speech_rate_band}` | timing |

Naturalness is deterministic arithmetic — calibration is informative but not
gating (the metric *is* the threshold). The three judge dimensions are the
load-bearing cases.

The roadmap calls for *voice rating*: humans rate by listening to the
trajectory and speaking structured scores aloud, which an LLM parser
converts to scalars. The goal is to keep the rater's attention on prosody,
not on a typing UI; this matters because the dimensions being rated include
audio.

## Design

### 1. `HumanRating` schema

A new pydantic model in `rehearse/eval/types.py` (or alongside `RubricScore`
in `rehearse/types.py` — TBD; matches whatever house style currently uses
for eval-side types):

```python
class HumanRating(Strict):
    rater_id: str                       # short alpha id, e.g. "jz"
    session_id: str                     # the trajectory rated
    dimensions: dict[str, float]        # {"content_quality": 0.7, ...}
    rubric_version: str                 # rubric the rater was shown
    judge_prompt_versions: dict[str, str]  # for invalidation on prompt bump
    rated_at: datetime                  # UTC
    raw_audio_path: str | None          # voice recording the parser consumed
    raw_transcript: str | None          # what the LLM parser saw
    flags: list[str]                    # e.g. "low_audio_confidence"
```

Stored at `evals/golden/v1/ratings/{session_id}__{rater_id}.json`. One file
per (session, rater) tuple; new ratings overwrite. Dropped if any
`judge_prompt_versions[dim]` no longer matches the current version when the
harness loads.

### 2. Voice rating UI

New routes on the existing FastAPI app (mounted in `rehearse/app.py`):

- `GET /viewer/{session_id}/rate` — renders the trajectory (transcript +
  audio player) plus a structured prompt: "Rate {dimensions} on 0–1 by
  speaking each score and a brief reason." Includes a record button that
  uploads a single WAV via `POST /viewer/{session_id}/rate`.
- `POST /viewer/{session_id}/rate` — accepts `multipart/form-data` with the
  audio blob and `rater_id`. Pipeline:
  1. Persist the audio under `evals/golden/v1/recordings/`.
  2. Transcribe via Hume's existing STT path (or a small wrapper around
     OpenAI Whisper; choose the cheapest available — TBD with one-line
     decision in the impl PR).
  3. Pass the transcript + the dimension list to a deterministic parser
     (`rehearse/eval/calibration/parser.py`) backed by a strict-output
     Anthropic call. Parser only converts utterances to scalars — it never
     evaluates the trajectory itself.
  4. If the parser cannot extract a clear scalar for any dimension, return
     422 with a re-record prompt. No partial saves.
  5. Persist a `HumanRating` JSON file.

The viewer for `production-replay` examples reuses this same UI by routing
through the source `session_id` (which the replay env preserves via
`RolloutResult.payload`).

### 3. Golden-set sampler

`scripts/sample_golden_set.py` — selects ~25 trajectories spread across the
affect distribution from the existing `production-sessions` dataset (or
sandbox runs once Mini-spec 2 second half lands). Heuristics:
- Stratify by opening affect (use the user's first turn, classified by the
  affect judge in stub mode just to bucket).
- Stratify by call duration band (short / typical / long).
- Drop any session with `flags` containing `audio_missing` or
  `consent_pending`.

Output: `evals/golden/v1/manifest.json` listing chosen `session_id`s + a
brief rationale per pick.

### 4. `CalibrationHarness`

`rehearse/eval/calibration/harness.py`:

```python
def compute_calibration(
    ratings_dir: Path,
    judge_scores: list[RubricScore],
    rubric_version: str,
) -> CalibrationReport:
    """Join human ratings against judge scores; return per-dimension Spearman ρ
    plus n, p-value, sample IDs, and pass/fail at the 0.6 floor."""
```

Returns a `CalibrationReport` keyed by dimension. Pure function — no I/O
beyond reading the ratings dir.

### 5. `data_card.json` writer

The runner gains a `_write_data_card` step at end-of-run. Schema:

```json
{
  "run_id": "20260506T...",
  "rubric_version": "v1",
  "calibration_status": {
    "content_quality":    {"rho": 0.71, "n": 25, "passed": true},
    "affect_perception":  {"rho": 0.55, "n": 25, "passed": false, "reason": "below_floor"},
    "delivery_quality":   {"rho": 0.68, "n": 25, "passed": true}
  },
  "excluded_flags": ["audio_missing", "consent_pending"],
  "judge_prompt_versions": { ... },
  "naturalness_thresholds_version": "v1"
}
```

Written to `evals/runs/<run_id>/data_card.json`. Read by Spec 7 to filter
dimensions before writing preference pairs.

### 6. `excluded_flags` filter

A function that takes a list of `RubricScore` rows and a list of flag names
to exclude. Used by Spec 7 (`turn_candidate_set_to_pairs`) and any future
training-data writer. Lives in `rehearse/eval/calibration/__init__.py`.

## File layout

```
rehearse/eval/calibration/
  __init__.py          # public exports + excluded_flags filter
  harness.py           # compute_calibration + CalibrationReport
  parser.py            # speech-to-scalar parser
rehearse/viewer.py     # new GET/POST /viewer/{session_id}/rate routes
scripts/
  sample_golden_set.py # CLI to generate evals/golden/v1/manifest.json
evals/golden/v1/
  manifest.json
  ratings/{session_id}__{rater}.json
  recordings/{session_id}__{rater}.wav
```

## Tests

Unit:
- Voice rating round-trip: known audio → expected scalar dict.
- Parser rejection: ambiguous speech → re-record prompt; no `HumanRating`
  written.
- Spearman ρ on synthetic fixture data — exact expected value.
- `data_card.calibration_status[dim].passed` flips correctly at the 0.6
  floor.
- Stale rating drop: a rating with an old `judge_prompt_versions[dim]` is
  excluded from the harness's input.

Integration:
- `pytest evals/` runs end-to-end against a fixture session bundle.
- Viewer route smoke: `GET /viewer/{session_id}/rate` returns 200 with
  expected payload structure.

## Sequencing

1. Schema (`HumanRating`, `CalibrationReport`).
2. Parser + tests.
3. Harness + tests.
4. Golden-set sampler.
5. Data-card writer.
6. Viewer routes + UI.
7. Calibrate jz on the v1 set; record ρ per dimension; gate downstream
   specs.

Each step is a separable PR.

## Open questions

- **STT provider**: reuse Hume's STT (already in the runtime stack) or a
  thin Whisper wrapper? Hume keeps the dependency surface flat; Whisper is
  cheaper per call. Decide in the schema PR.
- **Rubric copy**: the spoken rubric the rater is shown needs explicit
  anchors (what does 0 look like? what does 1?). Drafted from Appendix A of
  `v2026-05-05-multimodal-trajectory-rubric-rlaif.md`. Reviewed in the
  viewer PR.
- **Single rater (jz) calibration**: sufficient for v0 but a follow-on spec
  should add inter-rater reliability if a second rater joins.

## Manifest update

Add to `docs/specs/MANIFEST.md`:

| Spec | Status | Policy | Applies to | Notes |
|---|---|---|---|---|
| `v2026-05-06-mini-spec-3-calibration.md` | `acknowledged` | `implementation` | Mini-spec 3 | Decomposed from the 05-06 roadmap §7. Gates Mini-specs 6/7's training-data use. |

When work begins, move to `wip`. When ρ measurements land for v1, move to
`done`.
