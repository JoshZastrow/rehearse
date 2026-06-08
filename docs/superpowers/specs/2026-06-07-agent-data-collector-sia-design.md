# Agent Data Collector with SIA Feedback Loop

**Date:** 2026-06-07  
**Status:** Draft  
**Context:** Automated session curation, credit assignment annotation, and self-improving data selection criteria for caller-model fine-tuning.

---

## 1. Goal

Close the gap between `prepare_session_async` (which produces `audio_stereo.wav`) and the Modal training volume (which requires a curated manifest). Instead of pushing every session, a Claude agent reviews each session, annotates a turning-point for RL credit assignment, and approves or rejects it. A SIA feedback loop updates the selection criteria after each training run.

**Non-goals (deferred):** Audio trimming at the turning-point boundary. The annotation schema is designed to support it, but trimming is gated behind a config flag and validated before enabling.

---

## 2. Architecture

```
finalize()
  → annotate_session_async      (Whisper → audio.json)
    → prepare_session_async     (stereo WAV → audio_stereo.wav)
      → review_session_async    [NEW] Claude agent writes review.md
          if APPROVE → push_session_async [NEW] → dataset.py → Modal Volume

Nightly cron [NEW]
  → scans for audio_stereo.wav without review.md
  → reruns review_session_async for each gap

DataFeedbackAgent [NEW]  (fires after each training run)
  → reads review.md batch + training outcomes
  → writes updated docs/review_criteria.md
  → commits to git (harness update)
```

The review agent is the only new consumer of data the pipeline already produces. No changes to `finalize()`, `annotate_session_async`, or `prepare_session_async`.

---

## 3. Session Review Agent

### 3.1 Trigger

`prepare_session_async` in `train/pipeline/prepare.py` fires a background task at the end of its success path, mirroring how `annotate_session_async` chains off `finalize()`:

```python
asyncio.create_task(review_session_async(session_id, session_dir))
```

Idempotency guard: skip if `review.md` already exists in the session dir.

### 3.2 Inputs the agent receives

| Artifact | Path | Notes |
|---|---|---|
| Transcript | `transcript.jsonl` | Final turns only (`is_interim=false`), with `ts_start`, `speaker`, `text`, `phase` |
| Synthesis | `feedback.md` | Post-call coach synthesis narrative |
| Session manifest | `session.json` | Phase timings, completion status, outcome label |
| Selection criteria | `docs/review_criteria.md` | Versioned harness document (see §5) |
| Few-shot examples | 3 most recent approved `review.md` files | Loaded at runtime from the sessions root |

### 3.3 Agent tools

The agent is invoked via the Anthropic Python SDK with tool use. Four tools:

**`get_transcript()`** → returns the full final-turn transcript as a list of `{ts_offset_sec, speaker, text, phase}` dicts. Offsets are relative to the session start (computed from `ts_start` of the first turn). No audio read — transcript only.

**`annotate_turning_point(turn_index, speaker, ts_offset_sec, utterance_text, outcome_signal, confidence)`** → records the credit assignment label. Does not write any file; the label is embedded in the `review.md` the agent writes. Parameters:

| Field | Type | Meaning |
|---|---|---|
| `turn_index` | int | 0-based index into the final-turn transcript |
| `speaker` | `"provider" \| "caller"` | Who said the credited utterance |
| `ts_offset_sec` | float | Seconds from session start |
| `utterance_text` | str | Exact text of the credited turn |
| `outcome_signal` | str | What changed after this turn — what the agent observed in subsequent turns that indicates the credit |
| `confidence` | float 0–1 | Agent's confidence this is the correct attribution |

This tool is **always called** for approved sessions. If no clear turning point exists (the positive signal spans the whole session), the agent calls it with `confidence=0.2` and notes the uncertainty.

**`approve(trim_to_sec=None)`** → writes `review.md` with decision APPROVE and all annotations. `trim_to_sec` is accepted but ignored unless `enable_trimming: true` in config.

**`reject(reason)`** → writes `review.md` with decision REJECT and reason. No push occurs.

### 3.4 `review.md` format

```markdown
# Session Review: {session_id}
**Decision**: APPROVE | REJECT
**Reviewed**: 2026-06-07T16:30:00Z
**Criteria version**: {sha of review_criteria.md at review time}

## Quality Check
- Duration: 87s ✓  (threshold: ≥30s)
- Transcript turns: 15 final turns ✓  (threshold: ≥6)
- Completion status: complete ✓
- Stereo WAV: present ✓

## Training Value
{2–4 sentence narrative: what this session demonstrates, why it is or is not
a useful training example for the caller model.}

## Turning Point (RL Credit Label)
**Turn index**: 12  
**Speaker**: provider  
**Timestamp**: 4:12 (252s)  
**Utterance**: "What would it mean for you if this problem just went away?"  
**Outcome signal**: Caller shifted from objection to reflection in turns 13–14.
  Turn 13: "Hm. I hadn't thought about it that way."
  Turn 14: {continued engagement, no counter-argument}  
**Confidence**: 0.85  
**Trim boundary**: 4:30 (270s — end of caller's first positive response)

## Rejection Reason
{present only if REJECT}

## Rationale
{1–2 sentences on the final decision, referencing the criteria.}
```

The `## Turning Point` block is the structured RL credit label. `turn_index`, `ts_offset_sec`, `utterance_text`, `outcome_signal`, and `confidence` are machine-readable via regex parse from this section — no separate JSON needed. A future reward-model training pipeline reads this block to build `(utterance, credit_score)` pairs.

### 3.5 Credit assignment semantics

The turning point label is designed for GRPO/PPO reward modeling:

- **What gets credit**: the provider utterance that caused a measurable shift in caller behavior, as evidenced by the next 2–3 turns
- **What does not get credit**: conversation-level reward (the session was "good overall") or tail turns (winding down, pleasantries)
- **Confidence calibration**: the agent is instructed to use 0.8–1.0 only when the shift is unambiguous (explicit caller acknowledgment), 0.5–0.7 when inferred from tone/content change, 0.2–0.4 when the credit is diffuse across multiple turns
- **Reward model training**: a batch of `(utterance_text, outcome_signal, confidence)` triples extracted from approved `review.md` files forms the reward model's positive-class training set. Rejected sessions or low-confidence labels are negative or unlabeled class.

### 3.6 Implementation location

```
train/pipeline/
  review.py          # review_session_async, push_session_async, ReviewConfig
  review_criteria.md # initial criteria (committed to git — see §5)
```

`review.py` follows the same module pattern as `annotate.py` and `prepare.py`: an async hook function (`review_session_async`) that can be imported by `prepare.py`, and a standalone CLI for batch review.

### 3.7 Trimming (gated, Phase 2)

`ReviewConfig.enable_trimming: bool = False`. When `True`, `approve(trim_to_sec=X)` causes `push_session_async` to:

1. Slice `audio_stereo.wav` at `trim_to_sec * sample_rate` (in-memory numpy)
2. Filter `audio_stereo.json` alignments to `end_sec <= trim_to_sec`
3. Write `audio_stereo_clipped.wav` + `audio_stereo_clipped.json` alongside originals
4. Push the clipped pair instead of the originals

Originals are never modified. The `trim_boundary` field in `review.md` is populated regardless of the flag — this builds a labeled validation set for the trimming heuristic before it is enabled.

**Enabling criteria**: spot-check 50 sessions where `confidence >= 0.7`, compare agent's `trim_boundary` against human-labeled boundary. Enable when agent accuracy ≥ 80% on that set.

---

## 4. Nightly Catch-up Cron

A `CronCreate` job runs nightly at 02:00 local time. Logic:

1. Walk the sessions root for directories containing `audio_stereo.wav`
2. Skip any with an existing `review.md`
3. Run `review_session_async` for each gap — same agent, same tools, same criteria
4. Log `{session_id, decision, ts}` to `evals/runs/data-review/nightly-{date}.jsonl`

This handles: crashes during `review_session_async`, sessions completed during server downtime, and sessions produced by `generate_synthetic_calls.py` (which writes `audio_stereo.wav` directly).

The cron log is an input to the `DataFeedbackAgent` (§5.2).

---

## 5. SIA Feedback Loop

### 5.1 `docs/review_criteria.md` — the harness document

This is the selection policy the review agent reads on every session. Versioned in git. Updated only by the `DataFeedbackAgent`. Structure:

```markdown
# Session Review Criteria
**Version**: v1  
**Updated**: 2026-06-07  
**Updated by**: DataFeedbackAgent after training run rehearse_smoke-20260607

## Inclusion Thresholds
- Minimum duration: 30s
- Minimum final turns: 6
- Completion status: complete or partial (not failed, not declined)

## Training Value Criteria
Include sessions where the caller shows at least one of:
- Explicit acknowledgment of the provider's reframe ("I hadn't thought of it that way")
- Shift from objection-mode to question-mode across 2+ consecutive turns
- Voluntary disclosure of personal stakes unprompted

Reject sessions where:
- Caller shows no engagement beyond monosyllabic responses
- Session is interrupted before the practice phase begins
- Provider fails to complete a full coaching arc (intake only)

## Turning Point Heuristic
The turning point is the earliest provider utterance where the subsequent
2–3 caller turns show measurable shift. Prefer:
1. The utterance immediately before the first explicit acknowledgment
2. The utterance that introduced the reframe question
Assign confidence ≥ 0.8 only when the shift is unambiguous.

## Reject Rate Target
Aim for 30–50% rejection. If rejecting <20%, criteria are too loose.
If rejecting >70%, criteria are too strict — check recent improvement.md.
```

### 5.2 DataFeedbackAgent trigger

Fires automatically at the end of a successful training run (in `rehearse/train/modal.py`, after `volume.commit()`). Receives:

- Path to the training run directory (`/data/runs/{run_name}/`)
- Training loss curve (from `torchrun` stdout, parsed from Modal logs)
- List of session IDs used in this training batch (from the manifest)

### 5.3 DataFeedbackAgent inputs

For each session in the training batch:

| Artifact | Source |
|---|---|
| `review.md` | Session dir — decision, turning point, rationale |
| Rubric scores | `evals/runs/` if available; otherwise skip |
| Nightly cron logs | `evals/runs/data-review/nightly-*.jsonl` — gap and rejection rates |
| Prior `improvement.md` | `evals/runs/data-review/improvement.md` — rolling improvement history |
| Current `review_criteria.md` | `docs/review_criteria.md` |

### 5.4 DataFeedbackAgent output

Two files:

**`evals/runs/data-review/improvement.md`** — per-run analysis appended in SIA `context.md` style:

```markdown
## Training Run: rehearse_smoke-20260607-152325
**Sessions reviewed**: 47  
**Approved**: 31 (66%)  
**Rejected**: 16 (34%)  
**Training loss**: 2.41 → 1.87 (50 steps)

### What worked
Sessions with explicit caller acknowledgment (confidence ≥ 0.8) showed 
tighter loss curves — the turning-point label aligns well with the reward 
signal.

### What didn't
4 sessions were approved with diffuse turning points (confidence 0.3–0.4). 
Loss on those examples was 40% higher than the high-confidence batch. 
Rejecting confidence < 0.5 in future runs may improve signal quality.

### Criteria update
Raising minimum confidence threshold for approval from 0.0 to 0.5.
Updating reject rate target from 30–50% to 35–50%.
```

**Updated `docs/review_criteria.md`** — the agent rewrites this file with the new policy and commits to git with message: `chore(data): update review criteria after training run {run_name}`.

### 5.5 Plateau detection

Mirrors SIA's K=3 rule. If rubric scores across the last 3 training runs show no improvement on the dimensions that were improving (`rwrd`, `cont`), the `DataFeedbackAgent` escalates from a harness update to flagging a weight-update cycle:

```markdown
## Plateau Detected
3 consecutive training runs without rubric improvement on rwrd, cont.
Harness updates have been exhausted on current data distribution.
Recommended action: trigger LoRA training with updated replay buffer 
and raised confidence threshold (≥0.7 only).
```

This flag is written to `evals/runs/data-review/plateau.md` and surfaces in the next nightly cron log. A human (or a future automated agent) acts on it. No automatic weight-update dispatch in Phase 1.

---

## 6. File Layout

```
sessions/{id}/
  audio_stereo.wav          existing
  audio_stereo.json         existing
  review.md                 NEW — per-session review decision + credit label
  audio_stereo_clipped.wav  NEW (Phase 2, enable_trimming only)
  audio_stereo_clipped.json NEW (Phase 2)

docs/
  review_criteria.md        NEW — versioned selection policy (harness)

evals/runs/data-review/
  nightly-2026-06-07.jsonl  NEW — catch-up cron log
  improvement.md            NEW — rolling SIA feedback history
  plateau.md                NEW — plateau flag (written on detection)

train/pipeline/
  review.py                 NEW — review_session_async, push_session_async, ReviewConfig
```

---

## 7. Configuration

`ReviewConfig` (chz dataclass, same pattern as `AnnotateConfig`):

| Field | Default | Description |
|---|---|---|
| `enable_trimming` | `False` | Gate for Phase 2 audio clipping |
| `min_duration_sec` | `30.0` | Minimum session duration to consider |
| `min_turns` | `6` | Minimum final transcript turns |
| `few_shot_count` | `3` | Number of recent approved `review.md` files to include as context |
| `model` | `claude-opus-4-7` | Model for the review agent |
| `criteria_path` | `docs/review_criteria.md` | Path to the harness document |
| `sessions_root` | `sessions/` | Root for session directories |
| `cron_hour` | `2` | Hour for nightly catch-up cron |

---

## 8. Phased Delivery

**Phase 1 (this spec):**
- `review_session_async` + 4 tools + `review.md` output
- `push_session_async` (pushes full session, no trimming)
- Nightly catch-up cron
- `docs/review_criteria.md` initial version
- `DataFeedbackAgent` (harness updates only, no plateau auto-dispatch)
- `improvement.md` rolling log

**Phase 2 (after validation):**
- Enable `enable_trimming: True` after 50-session spot-check
- Clipped artifact generation in `push_session_async`
- Reward model training pipeline reading `review.md` turning-point blocks

---

## 9. Testing

- **Unit**: `review.py` — mock the Anthropic client; assert `review.md` is written with correct structure for both APPROVE and REJECT paths; assert idempotency guard fires on second call
- **Integration**: Run `review_session_async` against `sessions/f3cbe179b31e4171bcf249b31d72a95f/` (real session) with a test criteria doc; assert `review.md` appears with all required sections
- **Cron**: Use `test_finalize_sweeper.py` as a pattern — inject a sessions root with a mix of reviewed and unreviewed dirs; assert only unreviewed ones are processed
- **DataFeedbackAgent**: Inject a mock training run with 3 prior improvement.md entries showing no rubric gain; assert `plateau.md` is written
