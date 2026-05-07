# Rehearse Evals — TODO

Last updated: 2026-05-06

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
