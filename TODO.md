# TODO

## Session data persistence (training-data flywheel)

- [x] **Persist web-call transcripts to the `rehearse-sessions` Modal Volume.**
  Done in `infra/web.py`: `serve_room_job` mounts `rehearse-sessions` at
  `/mnt/sessions`, sets `SESSION_ROOT` to it, and `sessions_vol.commit()`s on call
  end. session_id already matches the GPU interactive server's (agent passes it in
  the handshake), so transcript + audio co-locate under `/mnt/sessions/<id>/`.
- [ ] **Verify on the first live call** that both file sets survive in the same
  dir — agent writes `transcript.jsonl`/`session.json`; the interactive server
  writes `caller_stream.pcm`/`provider_stream.pcm`/`tokens.jsonl`/`mask.jsonl`.
  Two containers commit to the same dir; Modal merges different files, but confirm
  nothing clobbers (this is the one untested assumption).
- [ ] **Consent/retention gate before collecting.** Real conversations = user PII.
  Confirm the Clerk consent language + data-retention policy actually cover
  "we store your conversation to improve the model" before this runs in prod.

## In progress

- [ ] **Eval scoring baseline** — run `make eval-voice-rollout` across all 3 scenarios, capture baseline scores to track regressions

## Scoring bugs / calibration

- [ ] **`affect_perception` consistency** — scoring varies 0.0–0.65 across runs on the same scenario; Gemini response sometimes truncated even at 8192 tokens. Add retry on parse failure before zeroing.
- [ ] **`speech_rate_band` always 0** — check naturalness scorer logic; this dimension has never scored non-zero across any run observed
- [ ] **`intake_fidelity` stuck at 0.5** — only one `expected.intake_*` field is populated in seed scenarios; add `intake_relationship` + `intake_user_goal` to remaining scenario rows

## Phase transition / conversation structure

- [ ] **FEEDBACK phase runs too long** — FEEDBACK consumed 11/24 turns in the last run; add a FEEDBACK turn cap (3–4 turns) or compress the feedback budget further
- [ ] **Cue-based INTAKE→PRACTICE never fires** — `_INTAKE_READY_CUES` phrases ("let's practice", "i'm ready") don't appear in natural LLM output; either add them to the customer INTAKE prompt as an exit signal or remove the cue dependency and rely on budget alone
- [ ] **Phase budget constants need a home** — `_EVAL_BUDGETS` in `runtime_sandbox.py` are currently hardcoded; consider making them configurable per dataset row (`payload["phase_budgets"]`) for scenario-specific pacing

## Eval coverage

- [ ] **Run all 90 scenarios** — current runs use `--limit 1`; need a full pass to surface systematic failures
- [ ] **Multi-scenario `intake_fidelity` regression test** — monkeypatch `_infer_relationship` to always return `"counterparty"`, confirm score drops on at least one seed scenario (spec acceptance criterion 4)
- [ ] **`voice-judges-smoke` and `production-voice-replay` evals** — confirm whether these are still exercised; if not, delete their dedicated scorers (~860 LOC bloat)

## Bloat cleanup (confirmed safe, needs approval)

- [ ] Delete `lib/` (empty directory)
- [ ] Delete `docs/superpowers/specs/` (empty directory)
- [ ] Delete non-strict scorers if `voice-judges-smoke` / `production-voice-replay` are confirmed dead: `scorers/content_judge.py`, `scorers/affect_perception_judge.py`, `scorers/delivery_judge.py`, `eval/deepeval_adapter/` (~860 LOC)
- [ ] Archive or delete superseded April specs (`docs/specs/v2026-04-27-*`, `v2026-04-28-*`)

## Tone, comfort, and naturalness (hill-climbing toward human-feeling calls)

These metrics close the gap between what we measure today (content correctness, holistic delivery score) and
what drives customer delight: a call that feels warm, paced right, and naturally structured. Each item below
is tied to a customer outcome — quality, retention, or cost efficiency.

### Deterministic signals (no LLM cost — add to NaturalnessScorer)

- [ ] **Acoustic monotone index** — pitch (F0) variance per coach turn via `librosa`. Low variance = deadpan robot.
  Score `1 - exp(-σ_F0 / threshold)` and add as `naturalness.pitch_variance`.
  _Business outcome_: directly measures the "sounds like a robot" complaint. A score above threshold
  is a necessary condition for customer trust and repeat usage. Low pitch variance correlates with
  early call abandonment in voice UX research.

- [ ] **Energy envelope variance** — RMS amplitude variance per coach turn, same `librosa` pass as pitch.
  Flat energy across all turns = no emphasis, no warmth.
  _Business outcome_: expressiveness in delivery is the strongest human-perceivable signal of coach
  presence. Customers who feel "heard" are more likely to complete the session and return for a second call.

- [ ] **Speaking rate variability** — stddev of WPM across turns within a session, not just average WPM.
  A coach at exactly 145 WPM every turn regardless of emotional weight reads as scripted.
  _Business outcome_: rate flexibility is how a skilled coach signals they're responding to the user,
  not reciting. Measured variance → coachable signal → fewer "robot" drop-offs.

- [ ] **Turn-length reciprocity** — ratio of coach words to user words per exchange (target ≤ 0.6).
  A coach monologuing while the user is engaged misreads the room.
  _Business outcome_: directly tied to session completion rate. Users who can't get a word in disengage.
  Reciprocity below threshold is a leading indicator of a session not converting to a second booking.

- [ ] **User response latency** — time from coach turn end to user starting to speak (from `timing.jsonl`).
  Fast response = comfortable and engaged; long pause = confusion or hesitation.
  _Business outcome_: best available proxy for caller comfort without a post-call survey. A falling
  latency curve across the session = the user is warming up. Rising = something's wrong. Gives us a
  real-time comfort signal we can optimize against without adding survey friction.

- [ ] **User turn length trajectory** — are the user's turns getting longer across the session?
  A comfortable caller opens up over time. A flat or shrinking trajectory = the call is closing them down.
  _Business outcome_: longer user turns = richer practice reps = more value delivered per call.
  This is a quality-per-minute metric — optimizing it increases value without increasing call length.

- [ ] **Phase transition gap** — silence/bridge duration specifically at intake→practice and practice→feedback
  boundaries. Target: 0.5–2.0s. Below = abrupt; above = awkward dead air.
  _Business outcome_: the handoff moment is when customers consciously notice "this is a robot."
  Getting it right removes a jarring UX seam and reduces early hang-ups at phase boundaries.

### Targeted LLM signals (replace holistic delivery score with dense per-turn signal)

- [ ] **Per-turn expressiveness score** — narrow Gemini prompt per coach turn: "Does this turn sound
  monotone or expressive? Rate 0–1 and flag if it sounds robotic." Replaces the holistic session-level
  delivery score with a per-turn signal we can attribute to specific turns and hill-climb against.
  _Business outcome_: the holistic score averages away the worst moments. A per-turn score surfaces
  exactly which turn broke the conversation — giving us a precise training signal for BoN selection
  (Mini-spec 6) and a faster feedback loop for prompt tuning.

- [ ] **Warmth vs. competence score** — targeted Gemini prompt: "Does the coach sound warm and human,
  or efficient and clinical? Rate 0–1 where 1 = warm." One dimension, asked once per session.
  _Business outcome_: warmth is the primary driver of Net Promoter Score in coaching and therapy
  voice UX. A warm-feeling call converts to word-of-mouth. This metric gives us a single number
  to optimize against that maps directly to customer referral behavior.

- [ ] **Transition naturalness judge** — Gemini prompt over the 2–3 turns surrounding each phase
  handoff, asking specifically whether the transition felt natural or abrupt.
  _Business outcome_: phase transitions are the highest-risk moments for customer disengagement.
  A targeted judge here closes the measurement gap on the "handoff doesn't feel right" complaint
  and gives the BoN selector a signal to prefer smoother transitions at runtime.

### Calibration (the unlock for all of the above)

- [ ] **Mini-spec 3: human listening study** — Josh listens to 10–15 call pairs, rates which felt
  warmer/more natural. Compute Spearman ρ between each metric above and those ratings.
  Gate: no metric is used as a training signal until ρ ≥ 0.6 on its dimension.
  _Business outcome_: this is 2–3 hours of listening that determines whether every other metric on
  this list is pointed in the right direction. Without it, we are hill-climbing against proxy signals
  that may not track what customers actually feel. With it, every metric above becomes a trusted lever.
  Cost of being wrong without calibration: weeks of engineering toward a metric that doesn't matter.

## Deferred (post-v1 stability)

- [ ] Delete `rehearse/eval/environments/voice_agent_sandbox.py` (572 LOC) after ≥1 week of green `runtime-sandbox` runs
- [ ] Delete `rehearse/eval/environments/live_rollout_audio.py` (275 LOC) — subsumed by `runtime_sandbox._synthesize_audio`
- [ ] Update spec `docs/specs/v2026-05-07-runtime-eval-alignment.md` §11 to reflect audio-in-v1 decision (needs explicit permission)
