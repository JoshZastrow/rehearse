# Rehearse Evals — TODO

Last updated: 2026-06-26

---

## Interactivity benchmark harness (2026-06-26)

Build an eval harness that runs our model against the interactivity benchmarks
Thinking Machines reported for TML-Interaction-Small. No competing model scores
meaningfully on these, so they're a clean target for the interactive/proactive
direction. TimeSpeak and CueSpeak are TML-internal (no public release found —
we'd reproduce them from the published task definitions); the other three adapt
existing public datasets.

- [ ] **Stand up `evals/interactivity/` harness.** One runner that loads a
  benchmark spec, streams clips/audio to our model under a proactive protocol
  (model decides *when* to speak, not just *what*), and emits per-benchmark
  metrics in our existing `runs/{run_id}/` artifact schema. Reference baseline
  to beat: TML-Interaction-Small.

- [ ] **TimeSpeak** (timed speech initiation) — metric: macro-accuracy
  (TML-Small = 64.7%). TML-internal; reconstruct from the task definition.

- [ ] **CueSpeak** (verbal cue-triggered speech) — metric: macro-accuracy
  (TML-Small = 81.7%). TML-internal; reconstruct from the task definition.

- [ ] **ProactiveVideoQA** (visual cue-triggered speech) — metric: PAUC
  (TML-Small = 31.5; no-response baseline = 25.0). Public:
  arXiv:2507.09313, repo github.com/yellow-binary-tree/ProactiveVideoQA
  (PAUC scorer + data live here; see also MMDuet2, arXiv:2512.06810).

- [ ] **RepCount-A** (visual action counting) — metric: off-by-one accuracy
  (TML-Small = 33.4%). Public: RepCount dataset from TransRAC (arXiv:2204.01018);
  RepCount-A split used by FCA-RAC (arXiv:2406.12178).

- [ ] **Charades temporal localization** (visual action timing) — metric: mIoU
  (TML-Small = 30.4). Public: Charades-STA from TALL (arXiv:1705.02101) over
  the Charades dataset.

Notes: ProactiveVideoQA/RepCount/Charades are video benchmarks — our model is
audio-first, so the harness needs a vision path (or we scope to the two speech
benchmarks first and gate vision behind the multimodal track). PAUC and mIoU
scorers should be vendored from the source repos, not reimplemented, so our
numbers are comparable to the reported baselines.

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

