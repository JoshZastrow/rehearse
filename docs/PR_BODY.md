## Summary

Two commits on this branch:

1. **`runtime-eval alignment`** — implements `docs/specs/v2026-05-07-runtime-eval-alignment.md` Phases 1–4. The runtime is now lifted out of `telephony.py` into `RuntimeHost`, which boots identically in serving and in eval. The eval harness drives the real runtime against a synthetic caller. New `IntakeFidelityScorer` adds a deterministic regression signal at 5% weight.
2. **`spec: runtime mirror eval + streaming scores`** — proposal for the next phase: live audio in eval rollouts (post-hoc TTS), streaming scores to a `scores.jsonl` queue tailed by a `rehearse-eval watch` terminal renderer, and a `mirror-full` release-gate tier that exercises the real Hume EVI loop end-to-end.

## What landed (commit 1)

- **`rehearse/runtime.py`** — new module. `RuntimeHost` owns `PhaseProcessor`, `IntakeProcessor`, `TranscriptWriter`, `TimingWriter`. Two `CoachVoiceAdapter` impls: `TextOnlyCoachAdapter` (eval, calls Anthropic directly) and `HumeCoachAdapter` (serving, routes through `HumeEVIClient`).
- **`rehearse/transport.py`** — moved from `rehearse/eval/transports.py`. Old import path stays working via shim; will be removed in a follow-up.
- **`IntakeComplete` frame** added to `rehearse/frames.py`. `PhaseProcessor` now waits for `intake.json` to exist on disk before emitting `INTAKE→PRACTICE`. Eliminates a persona-compiler race.
- **`rehearse/eval/customers/llm_customer.py`** — `LLMCustomerDriver` (synthetic caller). Phase-aware system prompts, sends first turn without waiting for a runtime greeting, switches prompts on `phase_transition` control events, hard cap of 12 turns.
- **`rehearse/eval/environments/runtime_sandbox.py`** — `RuntimeSandboxEnvironment`. Wires `RuntimeHost` + `LLMCustomerDriver` + `InMemoryDuplexTransport`. Registered as `runtime-sandbox`. `voice-agent-sandbox` is now a deprecation shim that redirects with a stderr warning.
- **`rehearse/eval/scorers/intake_fidelity.py`** — deterministic scorer. Compares `intake.json` against `expected.intake_relationship`/`intake_stakes`/`intake_user_goal` on the example. Returns flagged zero with `intake_missing` when the artifact is absent. Five seed scenarios in the dataset have expected fields populated.
- **`voice-rollout-judges`** updated: `runtime-sandbox` is now the preferred environment; weights re-normalized to make room for `intake_fidelity: 0.05`.
- **DeepEval default judge** switched from OpenAI GPT to Anthropic Sonnet (`AnthropicModel(model="claude-sonnet-4-5")`). The codebase otherwise standardizes on Anthropic; `OPENAI_API_KEY` is no longer required for the legacy `ContentJudgeScorer` to construct.
- **Telephony regression fix.** `tests/test_telephony.py::test_media_websocket_bridges_twilio_to_fake_hume` was failing on HEAD (pre-existing). Root cause: starlette's `TestClient` cancels the anyio scope wrapping the ASGI handler immediately after `close(1000)`, interrupting `asyncio.to_thread()` calls in the handler's `finally` block before `intake.json` can be persisted. Fixed by wrapping the cleanup in `anyio.CancelScope(shield=True)`.
- **Provenance block in `summary.md`** when environment is `runtime-sandbox`. Documents which adapter, which TTS provider, which keys were available.

## What this PR proposes (commit 2)

The new spec at `docs/specs/v2026-05-08-runtime-mirror-eval-streaming.md` covers the outstanding work after Phase 4 lands:

1. **Audio in eval rollouts.** Adapt the post-hoc TTS pattern from `live_rollout_audio.py` into `runtime_sandbox.py`. Per-turn user/coach WAVs + `timing.jsonl`. Audio judges (`affect_perception`, `delivery_quality`) and `NaturalnessScorer` stop degrading.
2. **Streaming scores → `scores.jsonl` → `rehearse-eval watch`.** Each `RubricScore` appends to a per-run JSONL queue as soon as the scorer returns it. A separate `rehearse-eval watch <run_dir>` command tails the file and re-renders an aggregate table with `rich`. Decoupled — runner writes, watcher reads.
3. **Schema-diff CI.** Boot `RuntimeHost` via the serving path and via `runtime-sandbox`, diff the JSON schemas of `Session`/`IntakeRecord`/`CounterpartyPersona`, fail on divergence. Catches contract drift between paths.
4. **Two tiers, with a release gate.**
   - `text-plus-tts` (default, every CI, ~$0.10/rollout). Fast iteration loop.
   - `mirror-full` (release-gate, manual + nightly cron, ~$0.50/rollout). Real `HumeCoachAdapter`, audio-producing synthetic caller, real `HumeAudioBridge`. **Required green within 24h before any production deploy.**
5. **Renaming pass (Phase E).** Drop telecom and test-automation jargon: `InMemoryDuplexTransport` → `InMemoryTwoWayChannel`, `LLMCustomerDriver` → `SyntheticCaller`, `TwilioBridgeTransport` → `TwilioPhoneBridge`. Old names re-exported with `DeprecationWarning` for one release cycle.

The spec includes a "surfaced assumptions" section per the principle that *the developer's assumptions are what ship to production*. Specifically calls out where the cheap default tier is not a true mirror (Hume integration, ASR fidelity, interruption dynamics, live prosody coupling) and gates it behind `mirror-full` for release confidence.

## Test plan

- [x] `uv run pytest -q` — 377 passed, 0 failed (the previously-failing telephony test is now green; the previously-failing content judge test is now green via the Anthropic default + stub key)
- [x] `uv run rehearse-eval list-environments` shows `runtime-sandbox`
- [x] `grep -R "system.*coach\|COACH_SYSTEM" rehearse/eval/environments/runtime_sandbox.py` returns no hits (no static coach prompt)
- [ ] Manual: `make eval-voice-rollout --limit 1` end-to-end (requires `ANTHROPIC_API_KEY`)
- [ ] Manual: live Twilio smoke call still works (telephony path untouched apart from the cleanup shield)

## Follow-ups (not in this PR)

- Implement the v2026-05-08 spec (Phase B–E). Default-tier audio + streaming + schema-diff + renames is ~6 dev days; mirror-full release gate is +2 days on top.
- Audit `HumeCoachAdapter` line-by-line against `telephony.py`'s coach loop. Spec calls this out as Phase C.2.
- Pick a stddev budget per dimension for hill-climbing — single-run deltas under the budget should be treated as LLM noise, not regressions.
- Update `docs/specs/v2026-05-07-runtime-eval-alignment.md` to drop "v1 is text-only" language (already partially done in this PR; spec author should confirm the audio-update sections).

https://claude.ai/code/session_01LhZcBJgk5QMaoEd4xWHCUW
