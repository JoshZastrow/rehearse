# Rehearse Evals — TODO

Last updated: 2026-06-26

---

## Full-Duplex-Bench harness (2026-06-26)

Build an eval harness that runs our model against **Full-Duplex-Bench (FDB)** —
the public, audio-only benchmark for full-duplex spoken dialogue. It scores the
exact interactive skill the product depends on: *when* to speak, not just *what*
to say. It's the runnable public proxy for Thinking Machines' internal TimeSpeak
/ CueSpeak benchmarks (those are not released). TML themselves report on FDB
V1, V1.5, and V3.

Why this one and nothing else: our stack is audio-first, so the TML video
benchmarks (ProactiveVideoQA, RepCount-A, Charades) are off the critical path.
FDB is audio-native, open (code + data), and its v1.5 metrics — stop/response
**latency**, interruption and backchannel handling — line up directly with the
current `fix/interactive-true-latency-profiling` work.

Source: [github.com/DanielLin94144/Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(repo folders `v1_v1.5/`, `v2/`, `v3/`; v1 = arXiv:2503.04721, v1.5 =
arXiv:2507.23159; v3 code + data + paper released 2026-05). Vendor FDB's own
scorers — don't reimplement — so our numbers are comparable to published ones.

- [ ] **Stand up `evals/fdb/` harness.** One runner that loads an FDB version's
  data, drives our model under FDB's full-duplex protocol (streaming audio in,
  model decides when to respond / interrupt / backchannel), and writes FDB's
  metrics into our existing `runs/{run_id}/` artifact schema. Wrap FDB's
  server-client inference scripts (`v1_v1.5/model_inference/`) rather than
  porting the loop.

- [ ] **V1.5 first — overlap handling.** Four scenarios: user interruption,
  user backchannel, talking to others, background speech. Metrics: categorical
  dialogue behavior, stop + response latency, prosodic adaptation. This is the
  highest-signal slice for a coaching call; it also reuses the latency
  instrumentation already on this branch.

- [ ] **V1 — turn-taking baseline.** Static offline turn-taking eval. Cheaper
  than v1.5; run it first to validate the harness plumbing end to end against a
  known baseline before the overlap scenarios.

- [ ] **V3 — latest.** Add once v1/v1.5 are green. Confirm what v3 measures from
  the `v3/` folder + paper before wiring (it's a newer axis, not a re-run of
  v1.5).

- [ ] **Report vs TML.** Record our per-version scores next to TML-Interaction-
  Small's reported FDB numbers so we can see the gap on each axis.

---

## STT fine-tuning track (2026-06-12)

### Shipped

- [x] **`dev/STT` educational lab** — three-lesson curriculum for supervised
  fine-tuning of a speech-to-text model, modeled on the tinker-cookbook SDFT
  recipe. Lesson 0: the STT decoder is a language model (timestamps are
  tokens). Lesson 1: canonical Whisper SFT + WER eval. Lesson 2 (capstone):
  fine-tune to emit a diarized `[speaker, time, word]` stream in one decoding
  pass — two-speaker mixes built so speaker is ground truth by construction,
  word times distilled from the pretrained model's alignment.
- [x] **Verified end to end on laptop MPS** with reference results recorded in
  the lesson README: format validity 0% → 91.9%, WER 0.155, speaker accuracy
  95.9%, word-time error 0.19 s (whisper-tiny, 80 clips, 300 steps).
- [x] **Architecture findings written up** (`dev/STT/README.md`, "Where this
  points in rehearse"): today the interactive backend transcribes the caller
  with a post-hoc per-turn faster-whisper pass — turn-level, no word times —
  and caller `ProsodyEvent`s are hardcoded zeros (`backend.py:275`); the
  `ProsodyService` seam exists but only the Null impl is wired.

### Next top priority

- [ ] **Apply the lesson-2 recipe to real rehearse data.** Build a
  `[speaker, word, time]` training set from stereo session recordings
  (`audio_stereo.wav`): channel = ground-truth speaker, per-channel alignment
  = word times — the same construct-the-labels trick as `make_dataset.py`,
  on real conversational audio. Fine-tune whisper-small and evaluate on the
  same four axes. Gate: beats the current post-hoc turn-level ASR before any
  backend integration.

Runner-up (small, unblocks the Feedback-Agent loop): wire a real
`ProsodyService` into `InteractiveBackend` instead of inline zeros, so
`prosody.jsonl` carries signal before curation or the SIA loop depends on it.

## Result

Real coaching calls can now be scored automatically on three dimensions —
what was said, how it sounded, and how naturally the conversation flowed.
Quality of the coaching experience is finally measurable instead of
subjective.

## Purpose

Without measurement we can't tell if changes to the coach are improvements
or regressions. Once we trust the scores, every call becomes a data point
we can use to teach the coach to get better — the foundation of the AI
training loop the pitch depends on.

---

## Runtime-Eval Alignment (2026-05-07 DX review)

- [ ] **Pre-Phase 1 audit**: Before extracting `RuntimeHost`, document all global state, event loop assumptions, and Twilio lifecycle hooks in `telephony.py:mount_twilio_routes` that the extraction must preserve. (~1 hour; prevents a prod incident.)

---

## Tomorrow's massive action plan

- [ ] Refill the AI scoring credits (yesterday's run hit a quota wall before producing results).
- [ ] Score 3 real coaching calls and read the report.
- [ ] Lock the design for the human-rating system that proves our automatic scores are trustworthy.
- [ ] Kick off building that rating system — first piece is the data structure for storing human scores.

## Shipped today (2026-05-06)

- [x] Real recorded calls can be replayed through the scoring system.
- [x] Coach speech duration is now captured correctly (was being recorded as zero-length).
- [x] One-command shortcuts to run evaluations from the terminal.
- [x] Test set expanded from 1 sample to 5 covering a range of emotional tones.
- [x] Design doc written for the next major piece (the calibration / human-rating system).
- [x] Switched to a cheaper AI model for scoring — same quality, ~10× lower cost.

---

## What's coming next (multi-week roadmap)

### Sandbox audio (1–2 weeks)

- **Result:** simulated practice conversations include real audio, not just text.
- **Purpose:** lets us test thousands of coaching variations cheaply, instead of waiting for real users to call.
- **Actions:**
  - Hook the text-to-speech bridge into the simulated practice loop.
  - Generate audio for both sides of the conversation.
  - Confirm the per-call cost is low enough to run at scale.

### Best-of-N coaching (4–5 weeks, HIGHER RISK)

- **Result:** every coach response in a live call is silently picked from 2 options; both options + the choice get saved.
- **Purpose:** turns each real call into self-generated training data, no extra human labeling needed.
- **Actions:**
  - Write the implementation plan first (this one's risky enough to deserve its own design doc).
  - Verify the calibration system above gives us trust before any rollout.
  - Stress-test latency against the live phone provider — if it's too slow, we route around it.
  - Build the candidate-generation + selector + recording pipeline; keep the old single-response path as a safety net.

### Training data pipeline (2 weeks)

- **Result:** the saved candidate pairs from above become formatted training data for the AI.
- **Purpose:** closes the loop — calls become measurements, measurements become training data, training data improves the coach.
- **Actions:**
  - Read candidate pairs from saved sessions.
  - Score the unchosen options after the fact so we have head-to-head comparisons.
  - Only emit pairs where the calibration system has cleared the relevant dimension.

---

## Status

Foundational pieces — naturalness measurement, varied test samples, replay
system, and stability checks — are all shipped.

## Runtime-Eval Alignment — Additional TODOs (2026-05-07 Eng Review)

- [ ] **CoachVoiceAdapter live_api tests**: Add `@pytest.mark.live_api` tests for `TextOnlyCoachAdapter` (verifies CLM produces coherent coaching text) and adapter-switch invariant (switching to `HumeCoachAdapter` doesn't change RuntimeHost call sequence). ~30 min. Block: none.

- [ ] **IntakeComplete signal verification**: After Phase 1+2 ships, manually verify that `IntakeProcessor` emits `IntakeComplete` on the FrameBus before `PhaseProcessor` transitions to PRACTICE. Check `test_intake_complete_happens_before_practice()` passes. Block: Phase 1+2.

