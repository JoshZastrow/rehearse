# rehearse — Specs Manifest

This manifest is the routing table for specs. Before committing to work, check this
file first, then read only the specs marked `implementation` or explicitly named by
the phase you are building.

## Status Vocabulary

| Status | Meaning | Commit guidance |
|---|---|---|
| `acknowledged` | Accepted as project direction, but no committed implementation work yet. | Safe to plan against. Mark `wip` when a PR/branch starts implementing it. |
| `wip` | Committed implementation work exists, but the spec is not fully delivered. | Read before changing affected code. Update phase notes as work lands. |
| `done` | Delivered, frozen, or resolved. | Use as reference. Do not reopen without a new amendment spec. |
| `superseded` | Replaced by a newer spec or decision. | Do not implement from it. Read only for historical context. |

## Read Policy

| Policy | Meaning |
|---|---|
| `foundation` | Stable baseline that frames the whole project. |
| `implementation` | Active implementation source of truth. Read before committing related work. |
| `amendment` | Modifies one or more implementation specs. Read alongside the affected spec. |
| `historical` | Preserved decision record. Do not use as a build handoff. |

## Current Manifest

| Spec | Status | Policy | Applies to | Notes |
|---|---|---|---|---|
| [`../../SPEC.md`](../../SPEC.md) | `done` | `foundation` | Whole product | Foundational design, treated as frozen unless a new amendment says otherwise. |
| [`v2026-04-27-eval-harness.md`](v2026-04-27-eval-harness.md) | `wip` | `implementation` | Eval harness | Phases 1-2 have shipped. Later eval phases remain open. |
| [`v2026-04-28-mme-emotion-and-audio-targets.md`](v2026-04-28-mme-emotion-and-audio-targets.md) | `acknowledged` | `implementation` | Eval phases A1-A6 | Next active eval direction. Supersedes EQ-Bench as the primary eval path. |
| [`v2026-04-29-mme-seeded-rl-sandbox-eval.md`](v2026-04-29-mme-seeded-rl-sandbox-eval.md) | `acknowledged` | `implementation` | RL-style eval phases RLE1-RLE4 | Uses MME clips as emotional seeds for sandbox conversation rollouts and RLAIF-style judging. |
| [`v2026-04-27-runtime.md`](v2026-04-27-runtime.md) | `acknowledged` | `implementation` | Runtime phases R1-R7 | Read with the Drop Pipecat amendment. Sections C3, C5, and C7 are no longer authoritative. |
| [`v2026-04-28-drop-pipecat.md`](v2026-04-28-drop-pipecat.md) | `acknowledged` | `amendment` | Runtime phases R2-R7, eval simulated transport | Authoritative replacement for Pipecat-shaped runtime pieces. |
| [`v2026-04-28-hume-evi-bridge.md`](v2026-04-28-hume-evi-bridge.md) | `superseded` | `historical` | Runtime R2 decision history | Kept only to explain the bridge decision. Do not implement from it. |
| [`v2026-04-28-runtime-workstream.md`](v2026-04-28-runtime-workstream.md) | `done` | `implementation` | Runtime workstream end-to-end integration | Verified end-to-end 2026-05-01. Reference for runtime wiring and bus/storage contracts. |
| [`v2026-05-01-consent-and-outcome-capture.md`](v2026-05-01-consent-and-outcome-capture.md) | `acknowledged` | `implementation` | Consent capture + inline end-of-call outcome | Amended 2026-05-05 to capture outcome inline rather than via deferred SMS. Gates production-replay eval. |
| [`v2026-05-05-multimodal-trajectory-rubric-rlaif.md`](v2026-05-05-multimodal-trajectory-rubric-rlaif.md) | `acknowledged` | `amendment` | Eval rubric + runtime BoN + RLAIF data shape | Amends 04-29 §6.3, §7. Introduces multimodal rubric, runtime Best-of-N, naturalness + stability metrics. Decomposed into the 05-06 roadmap. |
| [`v2026-05-06-eval-system-roadmap.md`](v2026-05-06-eval-system-roadmap.md) | `wip` | `implementation` | Eval system sequencing | Roadmap that decomposes the 05-05 spec into mini-specs. **Shipped**: Mini-spec 0 (DeepEval adapter, commit 55c916c), Mini-spec 1 (schema + content judge + aggregator, commit fc34cbe), Mini-spec 2 first half (audio judges + voice-judges-smoke), Mini-spec 4 (NaturalnessScorer + fixture-emitted timing.jsonl + live runtime `TimingWriter` on the bus), Mini-spec 5 (production-replay environment + production-voice-replay eval, commit 1f72b10), Mini-spec 8 (stability via repetitions + `StabilityScorer` meta-scorer + runner `repetitions` + `make nightly-stability`, commit 75f9faa). **Open**: Mini-spec 2 second half (sandbox TTS for real audio), Mini-spec 3 (calibration + voice rating UI), Mini-spec 6 (runtime BoN), Mini-spec 7 (preference pairs). |
| [`v2026-05-06-mini-spec-3-calibration.md`](v2026-05-06-mini-spec-3-calibration.md) | `acknowledged` | `implementation` | Mini-spec 3 — calibration + voice rating UI | Decomposed from the 05-06 roadmap §7. Gates Mini-specs 6/7's admission of judge scores to training data. Move to `wip` when implementation begins. |
| [`v2026-05-06-expressiveness-evaluation.md`](v2026-05-06-expressiveness-evaluation.md) | `acknowledged` | `amendment` | Expressiveness measurement under `delivery_quality` | Options-stage spec for the vibe-check problem the 05-05 rubric folds into a single audio-judge dimension. Pending decision on primary signal shape (pairwise vs anchored multi-dim). |
| [`v2026-05-06-hume-config-as-code.md`](v2026-05-06-hume-config-as-code.md) | `acknowledged` | `implementation` | Hume EVI persona/config management | Move Hume EVI configs from dashboard to versioned code. Touches `rehearse/services/hume_evi.py`, `rehearse/config.py`. |
| [`v2026-05-06-persona-routing.md`](v2026-05-06-persona-routing.md) | `wip` | `implementation` | SMS-body persona classifier → Hume config selection | Foundations shipped (persona_key on Session, SMS classifier, `relationship_coach` persona, orchestrator setter). Remaining: wire `_classify_and_set_persona` into the inbound-SMS path and have `HumeEVIClient._connect` read the persona-derived config id. |
| [`v2026-05-06-persona-config-swap.md`](v2026-05-06-persona-config-swap.md) | `draft` | `implementation` | Per-phase coach/character/feedback role swap via CLM | Routes Hume's per-turn LLM through our phase-aware `/chat/completions` so the assistant flips coach → in-character counterparty → feedback-coach as the manifest transitions. Adds `feedback_coach` role + `PersonaSwapCoordinator` bridge utterances. |
| [`v2026-05-06-r6-reliability-backlog.md`](v2026-05-06-r6-reliability-backlog.md) | `done` | `implementation` | R6 runtime reliability punch list | All four items shipped: 1 (stream WAV to disk, commit 5d89572), 2 (Hume reconnect with backoff schedule + budget), 3 (finalize sweeper, commit b6c5242), 4 (`FinalizeSweeper.recover_orphans()` wired into app lifespan startup). |
| [`v2026-05-06-time-aware-clm.md`](v2026-05-06-time-aware-clm.md) | `done` | `implementation` | Per-turn time card injection into CLM | Shipped via commits 8601f1c → a54d91b. TimeCard built and rendered as a second cached system block on every CLM turn. |
| [`v2026-05-11-voice-participant-protocol.md`](v2026-05-11-voice-participant-protocol.md) | `wip` | `implementation` | Voice participant abstraction layer | Implementation started on `codex/voice-participant-protocol`: defines `VoiceParticipant` + `VoiceSpeaker`, `SpeakRequest` Pydantic type, removes bare `hume.say` from business logic. |

## Workstream Map

| Workstream | Active specs to read | Ignore for implementation |
|---|---|---|
| Eval harness maintenance | `v2026-04-27-eval-harness.md` | `v2026-04-28-hume-evi-bridge.md` |
| Audio-native eval work | `v2026-04-27-eval-harness.md`, `v2026-04-28-mme-emotion-and-audio-targets.md` | `v2026-04-28-hume-evi-bridge.md` |
| RL-style sandbox eval work | `v2026-04-27-eval-harness.md`, `v2026-04-29-mme-seeded-rl-sandbox-eval.md` | `v2026-04-28-hume-evi-bridge.md` |
| Multimodal eval + RLAIF | `v2026-05-05-multimodal-trajectory-rubric-rlaif.md`, `v2026-05-06-eval-system-roadmap.md`, `v2026-05-06-expressiveness-evaluation.md` | `v2026-04-28-hume-evi-bridge.md` |
| Runtime R1 | `v2026-04-27-runtime.md` | `v2026-04-28-hume-evi-bridge.md` |
| Runtime R2-R7 | `v2026-04-27-runtime.md`, `v2026-04-28-drop-pipecat.md`, `v2026-04-28-runtime-workstream.md` | `v2026-04-28-hume-evi-bridge.md`; superseded runtime sections C3, C5, C7 |
| CLM / persona / Hume config | `v2026-05-06-hume-config-as-code.md`, `v2026-05-06-time-aware-clm.md` | `v2026-04-28-hume-evi-bridge.md` |
| Voice participant / coach swap | `v2026-05-11-voice-participant-protocol.md` | `v2026-04-28-hume-evi-bridge.md` |
| Consent + production data capture | `v2026-05-01-consent-and-outcome-capture.md`, `v2026-04-28-runtime-workstream.md` | `v2026-04-28-hume-evi-bridge.md` |
| ML data pipeline | `../../SPEC.md` | No dedicated spec yet. Write one before implementation. |

## Update Rules

1. Add every new spec to the manifest in the same PR that adds the spec.
2. Move a spec to `acknowledged` once it is accepted as build direction.
3. Move a spec to `wip` when committed implementation work begins.
4. Move a spec to `done` only when its acceptance criteria are delivered and verified.
5. Mark older specs `superseded` instead of deleting them when a decision record is useful.
6. When a spec amends another spec, name the superseded sections in both files and in this manifest.
