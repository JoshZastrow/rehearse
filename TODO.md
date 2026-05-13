# TODO

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

## Deferred (post-v1 stability)

- [ ] Delete `rehearse/eval/environments/voice_agent_sandbox.py` (572 LOC) after ≥1 week of green `runtime-sandbox` runs
- [ ] Delete `rehearse/eval/environments/live_rollout_audio.py` (275 LOC) — subsumed by `runtime_sandbox._synthesize_audio`
- [ ] Update spec `docs/specs/v2026-05-07-runtime-eval-alignment.md` §11 to reflect audio-in-v1 decision (needs explicit permission)
