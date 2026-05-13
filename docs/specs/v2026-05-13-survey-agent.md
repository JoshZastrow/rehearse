# rehearse — Post-Call Survey Agent

**Status**: draft (implementation-facing)
**Owner**: jz
**Date**: 2026-05-13
**Depends on**: `SPEC.md`, `rehearse/types.py`,
`docs/specs/v2026-05-01-consent-and-outcome-capture.md`,
`docs/specs/v2026-05-12-agent-design-patterns.md`

---

## 0. One-line summary

A new `SURVEY` phase appended to the call pipeline that runs an isolated
agent — with its own memory, question state, and persisted artifact — to
collect structured user feedback by conversationally asking 2–3 questions
generated from the session's eval dimensions and feedback content.

---

## 1. Why

The existing `OutcomeProbe` captures one binary label ("did this help? Y/N")
at a fixed point in the feedback budget. That label is useful for preference
pair mining but it misses two things:

1. **Dimensional coverage.** The eval harness scores eight `RubricDimension`
   values per session (character believability, fault precision, feedback
   groundedness, etc.). The OutcomeProbe collapses all of them into one bit.
   A survey that asks about the dimensions the eval tracks gives us a human
   signal alongside the automated signal — the starting point for rubric
   calibration and human-in-the-loop preference labeling.

2. **Conversational richness.** A Y/N question at the tail end of the
   feedback phase captures a reflex, not a considered response. A distinct
   phase where the user knows they are being asked for feedback — and where
   follow-up is possible — captures more useful verbatim signal.

The survey is implemented as an independent component rather than a bolt-on to
`OutcomeProbe` because it has meaningfully different concerns: it manages a
multi-turn question queue, calls an LLM to generate targeted questions from
session artifacts, classifies responses across multiple axes, and writes its
own artifact file. Coupling that to `OutcomeProbe` would make both harder to
maintain.

---

## 2. Scope

In scope:
- A new `SURVEY` phase (added to the `Phase` enum) that begins when the
  feedback budget elapses.
- Splitting the FEEDBACK budget from 60s to 30s to keep total call length
  unchanged.
- A new `SurveyAgent` component in `rehearse/survey.py` that owns the full
  survey lifecycle: question generation, delivery, response capture, and
  persistence.
- A new `survey.json` artifact containing `SurveyRecord` (session-level
  metadata) and `SurveyResponse` rows (one per question answered).
- Backward compatibility: the existing `OutcomeLabel` and
  `outcome_probe_status` fields on `Session` are populated from the final
  survey question (the "did this help?" question) so downstream training
  pipelines are unaffected.
- The existing `OutcomeProbe` is retired; `SurveyAgent` subsumes its role.

Out of scope:
- A web-based survey surface or post-call SMS survey link (voice only, v1).
- Cross-session "longitudinal" surveys that compare multiple calls.
- Human rubric labeling UI (separate workstream).
- Changes to the eval harness, synthesis prompts, or training pipelines beyond
  backward-compat wiring of `OutcomeLabel`.
- Caller memory integration (the survey agent's memory is per-call, not
  cross-session; caller memory is spec'd separately in
  `v2026-05-12-consent-caller-memory.md`).

---

## 3. Functional requirements

### 3.1 Phase split

- **F-S1.** The `FEEDBACK` phase budget MUST be reduced from 60s to 30s.
  The `SURVEY` phase budget MUST be 30s. Total wall-clock call length is
  unchanged (INTAKE ~60s + PRACTICE ~120s + FEEDBACK 30s + SURVEY 30s).
- **F-S2.** `PhaseProcessor` MUST transition `FEEDBACK → SURVEY` on the
  existing budget mechanism; no new cue-based exit is required. The transition
  emits a `PhaseSignal(from_phase=FEEDBACK, to_phase=SURVEY, reason="budget")`.
- **F-S3.** The coach MUST speak a brief handoff line before the SURVEY phase
  begins (e.g., "Before we hang up — I'd love your quick take on a couple of
  things."). This is a fixed string, not LLM-generated, delivered via
  `hume.say()` at the moment `PhaseSignal(to_phase=SURVEY)` fires.

### 3.2 Question generation

- **F-Q1.** At `PhaseSignal(to_phase=SURVEY)`, `SurveyAgent` MUST trigger a
  single synchronous LLM call to generate the survey question queue. The call
  MUST complete before the first question is spoken.
- **F-Q2.** The question-generation prompt MUST include:
  - The session's `IntakeRecord` (situation, goal, counterparty relationship).
  - The coach's utterances from the `FEEDBACK` phase (read from `TranscriptFrame`
    rows with `phase=FEEDBACK` and `speaker=COACH` already buffered by the agent).
  - The full list of `RubricDimension` values with one-line descriptions, as
    context for which dimensions the system evaluates.
  - A hard cap: generate exactly `SURVEY_QUESTION_COUNT` questions (default 2)
    plus the fixed outcome question (always last). Total is at most 3.
- **F-Q3.** Generated questions MUST be:
  - Specific to what the coach actually covered in feedback (not generic).
  - Phrased for spoken delivery (short, no jargon).
  - Classified at generation time as `scale`, `binary`, or `open` response type.
  - Annotated with the `RubricDimension` they are intended to probe.
- **F-Q4.** If the LLM call fails or times out (`SURVEY_GENERATION_TIMEOUT_SECONDS`,
  default 5s), `SurveyAgent` MUST fall back to a static question bank keyed by
  `ScenarioCategory` (defined in `rehearse/survey.py`). The fallback MUST be
  indistinguishable to the user from a generated question.
- **F-Q5.** The fixed final question — "Overall, did this rehearsal feel
  useful?" — MUST always be the last question and MUST always be classified as
  `binary`. It maps directly to `OutcomeLabel.did_it_help` for backward compat.

### 3.3 Conversational delivery and capture

- **F-D1.** `SurveyAgent` MUST speak each question in sequence via `hume.say()`,
  waiting for a classified user response before advancing.
- **F-D2.** The agent MUST classify each user response within
  `SURVEY_RESPONSE_TIMEOUT_SECONDS` (default 12s per question):
  - `scale` questions: parse a digit 1–5 or a word equivalent ("great" → 5,
    "terrible" → 1). Unresolvable → `unclear`.
  - `binary` questions: reuse `classify_outcome` from `rehearse/outcome.py`.
  - `open` questions: always resolve as `verbatim`; no classification needed.
- **F-D3.** On `unclear` for `scale` or `binary` questions, the agent MUST
  re-prompt once with a short clarification (e.g., "On a scale of one to five —
  one being least useful, five being most"). On second `unclear` or on timeout,
  the agent MUST record `SurveyResponse.captured = False` and advance to the
  next question.
- **F-D4.** On user hangup during the survey, the agent MUST record all in-flight
  questions as `captured = False` during the existing finalize path. Survey status
  transitions from `"in_progress"` to `"partial"`.
- **F-D5.** After the final question is answered (or skipped), the agent MUST
  speak a closing line ("Thanks — enjoy the conversation.") and allow the call to
  end naturally. The agent MUST NOT hold the call open past the `SURVEY` phase
  budget.
- **F-D6.** All survey interactions MUST be written to `transcript.jsonl` with
  `phase=SURVEY` so they appear in existing artifact viewers without changes.

### 3.4 Persistence

- **F-P1.** `SurveyAgent` MUST write `survey.json` to the session directory
  containing one `SurveyRecord` with the generated questions and their
  corresponding `SurveyResponse` rows.
- **F-P2.** `Session.survey_status` MUST be updated atomically via
  `store.update_session` at each lifecycle transition: `None → "generating" →
  "in_progress" → "complete" | "partial" | "skipped"`.
- **F-P3.** On survey completion, `SurveyAgent` MUST write `OutcomeLabel` from
  the final binary question's response (if captured) and set
  `Session.outcome_probe_status = "captured"` — or `"skipped"` if the question
  was not captured. This preserves backward compat with the training pipeline.
- **F-P4.** `survey.json` MUST be written before `survey_status` is set to
  `"complete"` so a crash between the two always leaves the artifact with
  `survey_status = "generating"` rather than a missing file.

---

## 4. Non-functional requirements

### 4.1 Latency

- **N-L1.** The LLM question-generation call MUST complete in under 5s p95.
  Questions longer than `SURVEY_MAX_QUESTION_CHARS` (default 120) are
  truncated before delivery.
- **N-L2.** `classify_outcome` (binary) and the scale parser MUST run in under
  50ms. No LLM in the per-response classification path.
- **N-L3.** The handoff line (§F-S3) MUST begin within 500ms of the
  `FEEDBACK → SURVEY` phase signal, so the transition feels seamless to the
  caller.

### 4.2 Reliability

- **N-R1.** The static question bank fallback (§F-Q4) MUST always produce a
  non-empty question list. A missing LLM key or network partition MUST NOT
  produce a silent survey — at minimum the fixed outcome question is always asked.
- **N-R2.** `survey.json` writes MUST be idempotent: if the agent crashes and
  restarts mid-survey (not expected in prod, but exercised in tests), a partial
  `survey.json` is treated as `survey_status = "partial"` by the finalize path.
- **N-R3.** The survey lifecycle MUST NOT block synthesis. `SessionSynthesizer`
  reads `survey.json` if present, but `synthesis.py` MUST handle its absence
  gracefully (survey artifact is optional in `SessionArtifacts`).

### 4.3 Privacy

- **N-P1.** Open-ended survey responses are stored verbatim. They are NOT
  rewritten or summarized before persistence. The training pipeline is
  responsible for any normalization.
- **N-P2.** The question-generation LLM prompt MUST NOT include raw audio
  paths or prosody scores — only transcript text and metadata.
- **N-P3.** If `Session.consent == DECLINED`, the survey phase MUST NOT run.
  Declined sessions finalize partial before FEEDBACK and never reach SURVEY.

---

## 5. Data sources

| Source | Use | Where it enters `SurveyAgent` |
|---|---|---|
| `IntakeRecord` (from `session.intake`) | Situation, goal, counterparty — grounds question generation | Read from `store.get_session(session_id)` at survey start |
| `TranscriptFrame` rows buffered during FEEDBACK phase | What the coach said in feedback — makes questions specific | Buffered in `SurveyAgent._feedback_coach_turns` as frames arrive |
| `RubricDimension` enum + descriptions | Which eval dimensions the system tracks — included in question-generation prompt | Static; imported from `rehearse/types.py` |
| `ScenarioCategory` on `IntakeRecord` | Selects the static fallback question bank | Read from `Session.intake.scenario_category` (new optional field) |
| Live transcript frames (`TranscriptDelta`) | User responses to survey questions | Subscribed via `FrameBus` |

No new external data dependencies beyond the existing LLM client already used
in the CLM responder.

---

## 6. Model selection

### 6.1 Question generation

**Choice:** single Claude Haiku call (non-streaming, ~1–2s) at survey phase start.

Rationale:
- The question set is generated once per session, not per turn. A 1–2s LLM
  latency is acceptable here — the agent speaks the handoff line (§F-S3) while
  the call is in flight, masking the delay.
- Haiku is sufficient for the task: the output is 2 short sentences per question,
  not long reasoning chains.
- Streaming adds complexity without benefit; the full question list must be
  ready before any question is spoken.

Prompt structure (in `rehearse/survey.py`):
```
SYSTEM: You are generating short spoken survey questions for a voice coach app.
        Each question probes one of the eval dimensions listed below. Use only
        what the coach covered in this specific feedback session.

USER:   <situation>…</situation>
        <goal>…</goal>
        <coach_feedback>…</coach_feedback>
        <eval_dimensions>…</eval_dimensions>

        Generate {n} questions. Each question:
        - Is ≤ 15 words
        - Is phrased for spoken delivery (no lists, no jargon)
        - Has response_type: scale | binary | open
        - Has rubric_dimension: one of the listed dimensions

        Return JSON array: [{question, response_type, rubric_dimension}, ...]
```

The JSON response is parsed with `pydantic.TypeAdapter[list[SurveyQuestion]]`.
On parse failure → fallback (§F-Q4).

### 6.2 Response classification

**Choice:** deterministic classifiers only. No LLM.

- **Scale**: regex `\b[1-5]\b` first; then a word→integer map (`"one"→1`,
  `"terrible"→1`, `"great"→5`, etc.). Words between digits default to
  `unclear`.
- **Binary**: delegate to `classify_outcome` from `rehearse/outcome.py`.
  Reuse both the lexicons and the existing test coverage.
- **Open**: always `verbatim`; no classification attempted.

Rationale: the response classifiers run on the live audio loop. Latency budget
is ~50ms p95. Any LLM call here would introduce a perceptible gap between the
user's answer and the next question.

### 6.3 Why not a long-running agent loop

The earlier design considered a fully agentic survey loop where the LLM decides
which question to ask next based on prior answers. That approach was rejected
because:
- A 30-second budget with 3 questions leaves ~10s per question including
  delivery and response. An agentic loop adds at least one LLM round-trip
  per question (~1–2s) against a p95 budget that's already tight.
- The session artifacts at survey time do not contain eval scores — the eval
  harness runs post-call. A true adaptive agent would need those scores to be
  meaningful. That's a future spec.
- The fixed-queue approach is easier to test, replay in evals, and reason about.
  An agentic loop adds statefulness that the question-generation call already
  provides implicitly.

---

## 7. Processing flow

### 7.1 Phase transition

```
PhaseProcessor enters FEEDBACK (30s budget)
  └─ SurveyAgent subscribes to FrameBus
  └─ SurveyAgent buffers coach TranscriptFrames with phase=FEEDBACK
        (for later question-generation context)

PhaseProcessor: FEEDBACK budget elapses
  └─ emits PhaseSignal(from=FEEDBACK, to=SURVEY, reason="budget")
  └─ SurveyAgent receives signal
  └─ coach speaks handoff line via hume.say() [synchronous]
  └─ SurveyAgent._generate_questions() fires async LLM call
  └─ session.survey_status ← "generating" persisted
  └─ LLM returns question list (or timeout → fallback)
  └─ session.survey_status ← "in_progress" persisted
  └─ question loop begins
```

### 7.2 SurveyAgent lifecycle

```
For each question in questions[]:
  └─ coach speaks question.text via hume.say()
  └─ agent sets state = "listening", records asked_at
  └─ watches TranscriptDelta bus for next final USER frame
       ├─ classify(frame.text, question.response_type)
       │    ├─ resolved
       │    │    └─ build SurveyResponse(captured=True, value=..., verbatim=...)
       │    │    └─ store response (in-memory for now)
       │    │    └─ advance to next question
       │    └─ unclear
       │         └─ if reprompts < 1: coach speaks clarification, retry
       │         └─ else: build SurveyResponse(captured=False, verbatim=frame.text)
       │              └─ advance to next question
       └─ timeout (SURVEY_RESPONSE_TIMEOUT_SECONDS per question)
            └─ build SurveyResponse(captured=False, verbatim=None)
            └─ advance to next question

After last question:
  └─ coach speaks closing line via hume.say()
  └─ write survey.json (SurveyRecord + all SurveyResponse rows)
  └─ session.artifact_paths["survey"] ← path persisted
  └─ if final question captured: write OutcomeLabel + outcome_probe_status="captured"
  └─ else: outcome_probe_status="skipped"
  └─ session.survey_status ← "complete"
  └─ allow call to end naturally (SURVEY budget expires or coach/user hangs up)
```

Edge cases:
- **User hangs up mid-survey.** `EndOfCall` frame received. All un-captured
  questions get `SurveyResponse(captured=False)`. `survey.json` is written
  with whatever was captured. `survey_status ← "partial"`. Finalize path
  checks for still-`"generating"` or `"in_progress"` status and promotes to
  `"partial"`.
- **LLM generation call times out.** Fallback question bank used; `SurveyRecord.
  generation_method` field set to `"fallback"` for observability.
- **SURVEY budget elapses while the agent is mid-question.** Phase budget
  does not force-interrupt question delivery — the agent is allowed to finish
  the current question and one response cycle. If it is still mid-survey when
  `EndOfCall` fires, same hangup path applies.
- **Consent declined.** `ConsentGate` finalize-partial path runs before
  FEEDBACK. Survey phase never receives a `PhaseSignal`. `SurveyAgent` idles
  and never fires. `survey_status` remains `None`.

### 7.3 Component sketch

`rehearse/survey.py` — single module, one main class, helpers:

| Symbol | Purpose |
|---|---|
| `SURVEY_HANDOFF_LINE` | Fixed coach line spoken at SURVEY phase start |
| `SURVEY_CLOSING_LINE` | Fixed coach line spoken after last question |
| `SURVEY_STATIC_QUESTIONS` | `dict[ScenarioCategory, list[SurveyQuestion]]` — fallback bank |
| `SURVEY_OUTCOME_QUESTION` | Fixed final binary question (always appended) |
| `SurveyAgentConfig` | Knobs: timeouts, question count, LLM model |
| `SurveyAgent` | Lifecycle: `run(frames)` + `_generate_questions()` + `_ask_question()` + `_persist()` |
| `classify_scale(text)` | Digit + word-map classifier; returns `int \| None` |
| `build_survey_response(...)` | Builds a `SurveyResponse`; mirrors `build_label` in `outcome.py` |

Wiring in `telephony.py::media_stream`:
```python
survey_task = asyncio.create_task(
    SurveyAgent(session_id, store, hume_client, llm_client).run(bus.subscribe())
)
```
Awaited in the same `finally` block as the other writers. The existing
`outcome_task` is removed once `SurveyAgent` subsumes it.

---

## 8. Schema changes

### 8.1 `rehearse/types.py`

**`Phase` enum** — add one value:
```python
SURVEY = "survey"
```

**New types**:
```python
class SurveyQuestion(Strict):
    text: str
    response_type: Literal["scale", "binary", "open"]
    rubric_dimension: RubricDimension | str
    asked_at: datetime | None = None

class SurveyResponse(Strict):
    question_text: str
    response_type: Literal["scale", "binary", "open"]
    rubric_dimension: RubricDimension | str
    captured: bool
    value: int | bool | str | None = None   # int for scale, bool for binary, str for open/verbatim
    verbatim: str | None = None
    captured_at: datetime | None = None

class SurveyRecord(Strict):
    session_id: str
    generation_method: Literal["llm", "fallback"]
    questions: list[SurveyQuestion]
    responses: list[SurveyResponse]
    started_at: datetime
    completed_at: datetime | None = None
```

**`Session`** — add two fields:
```python
survey_status: Literal["generating", "in_progress", "complete", "partial", "skipped"] | None = None
```
`survey` is added to `artifact_paths` (no new field — the existing dict).

**`PhaseTiming`** — no change; `SURVEY` phase timings are written by the
existing `TimingWriter` once `Phase.SURVEY` is in the enum.

No removals. No breaking changes to existing artifacts. Pydantic defaults
(`None`) handle existing sessions on disk.

### 8.2 `PhaseBudgets` (in `rehearse/phases.py` or `rehearse/runtime.py`)

```python
FEEDBACK = 30   # was 60
SURVEY   = 30   # new
```

---

## 9. Configuration

New environment variables:

| Var | Default | Purpose |
|---|---|---|
| `SURVEY_QUESTION_COUNT` | `2` | LLM-generated questions per call (not counting the fixed outcome question) |
| `SURVEY_GENERATION_TIMEOUT_SECONDS` | `5` | How long to wait for the question-generation LLM call before falling back |
| `SURVEY_RESPONSE_TIMEOUT_SECONDS` | `12` | How long to wait per question before marking `captured=False` |
| `SURVEY_MAX_QUESTION_CHARS` | `120` | Maximum characters per question; longer questions are truncated |
| `SURVEY_LLM_MODEL` | `"claude-haiku-4-5-20251001"` | Model for question generation |

Loaded by `RuntimeConfig.from_env`. Existing config tests get parametrized cases.

---

## 10. Observability and logging

All emitted via `structlog`. No new infra.

| Event | Fields | When |
|---|---|---|
| `survey.phase.started` | `session_id` | SURVEY PhaseSignal received |
| `survey.generation.started` | `session_id, method` | LLM call fires |
| `survey.generation.complete` | `session_id, method, question_count, latency_ms` | Questions ready |
| `survey.generation.timeout` | `session_id, timeout_s` | LLM call timed out; fallback used |
| `survey.question.asked` | `session_id, question_idx, rubric_dimension, response_type` | Coach speaks a question |
| `survey.response.classified` | `session_id, question_idx, response_type, captured` | Response parsed (no verbatim — privacy) |
| `survey.response.unclear` | `session_id, question_idx, attempt` | Re-prompt fired |
| `survey.response.timeout` | `session_id, question_idx` | No response within timeout |
| `survey.complete` | `session_id, questions_asked, responses_captured, latency_ms` | All questions done |
| `survey.partial` | `session_id, reason ∈ {"hangup","budget"}` | Survey ended before complete |

Counter to track over time: `capture_rate = responses_captured / questions_asked`.
Alert if it drops below 0.5 — indicates questions are too long, unclear, or the
timing budget is too tight.

LLM generation latency (`survey.generation.complete.latency_ms`) appended to
`telemetry.jsonl` so it appears in eval-time replays.

---

## 11. Test plan

**Unit:**
- `classify_scale` over a 25-item fixture: digits 1–5, word variants ("one",
  "terrible", "great", "pretty good → 4"), numbers out of range (0, 6 → `None`),
  multi-word ambiguous responses ("maybe three?" → 3), blanks → `None`.
- `build_survey_response` for all three response types: scale captured, binary
  captured, open verbatim, and `captured=False` paths.
- Question-generation prompt builder: given a fixed intake record + feedback
  transcript, assert the rendered prompt contains the situation text, the coach
  utterances, and all RubricDimension keys. No LLM call in this test — just
  the string assembly.
- LLM JSON parsing: well-formed output → `list[SurveyQuestion]`; malformed JSON
  → fallback bank; valid JSON wrong shape → fallback bank.
- `classify_scale` and `classify_outcome` (binary) both called through
  `SurveyAgent._classify_response` dispatch to confirm routing by `response_type`.

**Integration:**
- **Full survey captured path.** Build a fake `FrameBus`, push
  `PhaseSignal(to=SURVEY)`, inject a mocked LLM client returning two questions,
  push user transcript frames with classifiable responses, assert `survey.json`
  written with `captured=True` for all responses, `survey_status == "complete"`,
  `OutcomeLabel` written.
- **LLM timeout → fallback path.** Mocked LLM client that sleeps past
  `SURVEY_GENERATION_TIMEOUT_SECONDS`. Assert `generation_method == "fallback"`,
  fallback questions delivered, responses captured normally.
- **Hangup mid-survey.** Push `EndOfCall` after first question, before second.
  Assert first response captured or `captured=False`, second response
  `captured=False`, `survey_status == "partial"`, `survey.json` present.
- **Unclear response → re-prompt → skip.** Push `unclear` response twice for a
  scale question. Assert `SurveyResponse.captured=False`, agent advances to next
  question without blocking.
- **FEEDBACK budget reduction.** Fake phase budget clock, confirm `FEEDBACK`
  exits at 30s (not 60s), `SURVEY` `PhaseSignal` fires, agent lifecycle begins.
- **Backward-compat OutcomeLabel.** Full path where final binary question is
  captured. Assert `session.outcome_label.did_it_help == <response>` and
  `outcome_probe_status == "captured"`.

**Regression:**
- Existing `tests/test_outcome.py` and `tests/test_consent.py` continue to pass.
  `OutcomeProbe` is removed from the wiring but its classifier functions remain
  in `rehearse/outcome.py` and are now called by `SurveyAgent`.
- `tests/test_synthesis.py` passes with `survey.json` absent (optional artifact).

---

## 12. Rollout

Single PR. The `OutcomeProbe` task in `telephony.py` is replaced by `SurveyAgent`
in the same changeset. No feature flag: there is no state where the old 60s
feedback + 15s outcome probe is preferable once this ships.

Backward compatibility on disk: existing session artifacts without `survey.json`
are unaffected. `Session.survey_status = None` (default) is treated as
pre-survey by all new readers.

The FEEDBACK budget cut from 60s to 30s is the only visible change to existing
callers. This is a product decision, not an implementation risk.

---

## 13. Success criteria

The workstream is complete when all of the following are true:

1. A live call proceeds through INTAKE → PRACTICE → FEEDBACK (30s) → SURVEY
   (30s). The coach speaks 2 generated questions plus the outcome question. All
   three responses are captured in `survey.json`.
2. The generated questions reference something specific from the coach's 30s
   feedback — not boilerplate. (Manual review of 3 sessions.)
3. On a live call where the user hangs up mid-survey, `survey_status == "partial"`
   and `survey.json` contains partial response rows.
4. On a live call where the LLM generation call fails (simulated by pointing
   `SURVEY_LLM_MODEL` at a bad endpoint), the fallback bank produces questions
   and the survey completes without error.
5. The final binary question's response populates `OutcomeLabel` on `session.json`
   and the existing training pipeline processes it unchanged.
6. Unit + integration tests from §11 all pass in under 10s on local hardware.
7. `uv run pytest -q` passes with no regressions in existing suites.

---

## 14. Decision rule

If you're adding code under this spec, ask:

**Does this code exist to (a) generate targeted questions from session context,
(b) deliver and classify those questions in the SURVEY phase, or (c) persist
the responses and backward-compat OutcomeLabel?**

- If yes, it belongs in `rehearse/survey.py` or `rehearse/types.py`.
- If no — even if it touches `phases.py`, `telephony.py`, or `outcome.py` —
  it belongs in a separate PR.

The one permitted exception: changes to `rehearse/outcome.py` to expose
`classify_outcome` as a shared utility callable by `SurveyAgent`. That
refactor is additive and belongs in this PR.

---

## 15. Open questions

These are not blocking for v1 but should be resolved before the next spec
that builds on survey data:

1. **Scenario category on intake.** §5 notes `ScenarioCategory` as a data
   source for the fallback bank. `IntakeRecord` does not currently carry
   this field. The fallback bank could use a single universal question list
   instead, which is simpler for v1. Decide before implementation begins.

2. **Survey in eval harness.** The eval harness (`SyntheticCaller`) does not
   currently respond to survey questions. It should be extended to answer
   surveys plausibly for automated runs. That work is a follow-on to this
   spec once the `SurveyAgent` interface is stable.

3. **OutcomeProbe removal timing.** `OutcomeProbe` code can be deleted once
   `SurveyAgent` is live and verified on 5+ real calls. The recommendation is
   to keep the module (without the wiring) for one sprint as a fallback, then
   delete it.

4. **Survey data in synthesis.** `SessionSynthesizer` currently reads intake,
   transcript, and prosody. Once survey data accumulates, it could enrich the
   `feedback.md` artifact ("the caller said the pacing feedback wasn't clear").
   That's a synthesis prompt change, separate spec.
