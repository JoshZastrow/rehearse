# Ablation-Based Rejection of Training Examples — Design

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan
**Builds on:** the session curation gate (`rehearse/train/curation/`, PR #33)

## Problem

The curation gate filters sessions *before* they reach the `rehearse-training`
volume, but its judgments are predictions. A session can pass review and still
hurt training — the only ground truth is the training result itself. We need a
feedback path that rejects examples *already on the volume* based on measured
training outcomes, and feeds those verdicts back into the selection criteria.

## Decisions (made during brainstorming)

| Question | Decision |
|---|---|
| Attribution method | **Exact ablation (retraining) runs** — affordable at current corpus scale (~120-step runs on one A10G); gradient approximations (DataInf, TracIn) deferred until retraining is too expensive |
| Eviction mechanics | **Tombstone** — drop from the train manifest, keep artifacts on the volume, flip `review.json` to rejected with the measured delta |
| Verdict metric | **Held-out loss** on a pinned eval split; rubric eval deferred (checkpoint→eval-rollout wiring doesn't exist yet) |
| Trigger (ablation) | **Auto on regression** — after every training run, if held-out loss regresses past the noise band, investigation launches automatically, with spend guardrails |
| Trigger (curation) | **Pre-training sweep** — `curate_sessions` runs as the first step of every training dispatch, re-reviewing anything stale under the current criteria version |
| Architecture | **Client-side orchestrator** in the curation package (approach A); a Modal-native watcher is a possible later deployment of the same design |

## Architecture & data flow

```
curation-train command (new)
  │
  ├─ 1. PRE-TRAINING SWEEP: curate_sessions()
  │      reviews unreviewed/stale sessions, assigns held-out split,
  │      rebuilds + pushes data/curated_sessions.jsonl (train)
  │      and data/heldout_sessions.jsonl (eval, frozen membership)
  │
  ├─ 2. TRAIN: run_training() with eval_data = held-out manifest,
  │      fixed max_steps and seed; ledger entry written
  │
  ├─ 3. CHECK: read held-out loss from run dir; compare against the
  │      best prior accepted run in the ledger
  │      └─ regression ≤ noise band → record, done
  │
  └─ 4. INVESTIGATE (auto): AblationOrchestrator
         cohort = sessions added since last good run
         bisect with budget-matched Modal runs (same max_steps)
         │
         ├─ conviction → TOMBSTONE: rewrite train manifest without
         │   the session, flip review.json (local + volume) to
         │   rejected with the measured delta; artifacts kept
         │   └─ SIA: update_criteria() called with the verdict
         │
         └─ budget exhausted / inconclusive → no conviction,
             cohort flagged in ledger, cooldown applied
```

## Components

All new code lives in `rehearse/train/curation/` beside the existing gate.
No `train/**` edits expected (active WIP workspace).

### `ledger.py` — RunLedger

JSONL at `data/run_ledger.jsonl` on the volume (read/written client-side via
the existing volume client). One entry per run:

```json
{
  "run_id": "...",
  "kind": "baseline" | "training" | "ablation",
  "manifest_hash": "...",
  "excluded_session_ids": [],
  "seed": 0,
  "max_steps": 120,
  "heldout_loss": 3.08,
  "eval_set_version": 1,
  "status": "completed" | "failed",
  "investigation_id": null,
  "created_at": "..."
}
```

- **Noise band:** estimated from ≥2 `baseline` entries with the same
  `manifest_hash` and different seeds. If missing when needed, the
  orchestrator runs one extra baseline seed first (charged to the budget).
- Comparisons are only valid between entries with the same
  `eval_set_version`.

### Held-out split (in `curate.py`)

- Approved sessions get a sticky `split` field: `train` or `heldout`.
  Assignment: deterministic by session-id hash, ~20% of approved sessions,
  minimum 2. Membership never changes once assigned; changing the eval set
  invalidates ledger comparability, so any forced change bumps
  `eval_set_version` and resets the baseline requirement.
- `data/curated_sessions.jsonl` contains only `split=train` entries;
  `data/heldout_sessions.jsonl` holds the eval split. Held-out sessions are
  never trained on.

### `trainer.py` — Trainer protocol

```python
class Trainer(Protocol):
    def run(self, *, train_manifest: str, excluded_ids: list[str], seed: int) -> float:
        """Dispatch a budget-matched run; return held-out loss."""
```

Live implementation wraps `rehearse.train.modal.run_training` with
`data.eval_data` pointed at the held-out manifest and fixed `max_steps`
(budget-matched: every ablation run takes the same optimizer steps as the
baseline, so the measurement isolates data quality from corpus size).
moshi-finetune already accepts eval data (`train/finetune/data/args.py`); the
adapter reads eval loss from the run dir on the volume. **Open item:** confirm
the exact metrics file the train loop writes (expected under
`/data/runs/<run>/logs/`) during implementation.

### `ablation.py` — AblationOrchestrator

- **Regression check:** current run's held-out loss worse than the best prior
  accepted run by more than the noise band.
- **Suspect cohort:** sessions added to the manifest since that run.
- **Bisection:** ablate half the cohort; if loss recovers within the band the
  harm is in the removed half; recurse to a single session. O(log n) runs.
- **Conviction requires both directions:** excluding X recovers the loss
  within the band, and runs including X account for the regression.
- **Guardrails:** `max_runs` per investigation (default 6, configurable);
  per-cohort cooldown (no re-investigation of the same flagged cohort until
  the corpus changes); every run ledgered under one `investigation_id`.
  Budget exhausted before isolation → nothing tombstoned, cohort flagged.
- **Idempotent:** re-running an investigation resumes from ledgered results
  rather than re-spending GPU.

### Tombstoning (in `curate.py`)

On conviction: rewrite the volume train manifest without the session; update
`review.json` locally and on the volume — `decision: rejected`, rationale
carrying the measured delta, plus an `ablation` block:

```json
"ablation": {
  "investigation_id": "...",
  "delta": 0.14,
  "noise_band": 0.03,
  "run_ids": ["..."]
}
```

Artifacts stay on the volume (auditable; re-admission is a manifest rewrite).

### Triggers

- **`curation-train`** (new CLI command in the curation package): pre-training
  sweep → train → check → investigate. The sweep re-reviews any session whose
  `review.json` was stamped under a stale criteria version (the existing
  version short-circuit makes the cached path nearly free), so criteria bumps
  from the SIA loop automatically re-gate the whole corpus before training.
- **`curation ablate --run <id>`**: manual escape hatch to (re)run an
  investigation by hand.

### SIA hook

After any eviction, `update_criteria()` is called automatically with the
reviews plus the investigation summary — the question posed to the criteria
agent is: "review vN approved this session; the ablation convicted it; what
should the criteria have caught?"

## Error handling & safety

- A failed Modal run marks its ledger entry `failed` and aborts the
  investigation. Convictions are never made on partial evidence.
- The manifest rewrite happens after all verdicts, and before any
  `review.json` flips — a crash leaves either the old corpus or the complete
  new one, never a half-state.
- The pre-existing `data/sessions.jsonl` remains untouched (unchanged from
  the curation-gate design).

## Testing

Hermetic, same pattern as the reviewer tests (fake judge backend):

- **FakeTrainer** with a planted loss model: each session has a hidden harm
  coefficient; `heldout_loss = base + Σ harm(included sessions) + seed_jitter`.
- Tests: regression detection respects the noise band; bisection isolates a
  planted bad session within budget; budget exhaustion convicts nothing and
  flags the cohort; tombstone rewrites the manifest and review.json while
  preserving artifacts; split assignment is sticky and deterministic; ledger
  round-trips; investigations resume idempotently.
- One `live_modal`-marked smoke test running a miniature real investigation.

## Costs

Each investigation ≤ `max_runs` × (one A10G run at baseline `max_steps`).
At current run lengths (~120 steps, minutes of GPU), a full 6-run
investigation costs single-digit dollars.

## Non-goals / future work

- **Rubric-eval verdicts:** layering the 7-dimension rubric on top of held-out
  loss as a confirmation gate, once checkpoint→eval-rollout wiring exists.
- **Gradient-based attribution:** DataInf / TracIn / pyDVL when the corpus
  outgrows retraining (~50+ sessions); calibrate against the exact-ablation
  verdicts collected while small.
- **Review-on-prepare hook:** chaining a per-session review after
  `prepare_session_async` for early visibility (the pre-training sweep remains
  the authoritative gate).
- **Modal-native watcher:** running this orchestrator as a scheduled Modal
  function for fully hands-off operation.
