# rehearse — Consent + Outcome Capture

**Status**: draft (implementation-facing) — amended 2026-05-05 to capture
outcome inline at end-of-call rather than via deferred SMS.
**Owner**: jz
**Depends on**: `SPEC.md`, `rehearse/types.py`, `docs/specs/v2026-04-28-runtime-workstream.md`
**Separate from**: `docs/specs/v2026-04-27-eval-harness.md`

---

## 0. One-line summary

Two coupled features that turn the runtime from a demo into a data-collection
product: explicit verbal consent captured during the intake phase of the live
call, and a sparse outcome label captured by SMS after the user has had the
real conversation they were rehearsing for.

## 1. Why these two together

`Session.consent` and `Session.outcome_label` already exist in `types.py` but
nothing in the runtime writes to them today. They are coupled because:

- Consent gates whether a session may exist as a recording at all.
- Outcome turns a consented session into a usable training record.

Without consent capture, recording a non-founder caller is illegal in
two-party-consent jurisdictions (CA included). Without outcome capture, the
training corpus has no preference signal and T1–T2 in the ML roadmap cannot
start.

The two features share the same Twilio + storage primitives, the same session
identity, and the same legal framing (explicit caller agreement to be in the
data loop), so they are speced together.

## 2. Scope

In scope:
- Verbal consent prompt at the start of the intake phase; deterministic
  parse of the user's spoken response; gate on practice phase entry.
- Audio + transcript handling when consent is declined.
- Scheduled outbound outcome SMS after a consented session.
- Inbound SMS handler that records `OutcomeLabel` keyed by phone number.
- Schema, manifest, and observability changes to support the above.

Out of scope:
- A web UI for consent or outcome (SMS only, matching existing surfaces).
- Withdrawal of consent after the call (separate spec; harder problem).
- Multi-region two-party-consent rule enforcement (we treat all sessions as
  two-party-consent for safety).
- Any change to synthesis prompts, eval scorers, or training pipelines.

## 3. Functional requirements

### 3.1 Consent (during call)

- **F-C1.** The first coach utterance after the WS handshake MUST be a
  consent prompt naming what is recorded, what it is used for, and how to
  decline. Concrete text lives in `rehearse/personas.py`.
- **F-C2.** The runtime MUST classify the next final user transcript frame
  as `granted`, `declined`, or `unclear` using a deterministic
  affirmative/negative parser (see §6).
- **F-C3.** On `granted`, the runtime MUST set
  `Session.consent = ConsentState.GRANTED` and allow the phase machine to
  transition `intake → practice → feedback`.
- **F-C4.** On `declined` or unrecoverable `unclear` (no valid response in
  20 seconds or after one re-prompt), the runtime MUST:
  1. Set `Session.consent = ConsentState.DECLINED`.
  2. Have the coach speak a short acknowledgement and end the call.
  3. Call `orchestrator.finalize(session_id, "partial")`.
  4. Skip synthesis (no `story.md`, no `feedback.md`).
  5. Skip the viewer-link SMS.
  6. Purge `audio.wav` and `prosody.jsonl`. Retain `session.json` and
     `transcript.jsonl` truncated to the consent exchange only — these are
     the legal record that consent was offered and declined.
- **F-C5.** Practice-phase transitions MUST be blocked while
  `Session.consent != GRANTED`. If the phase budget elapses without consent,
  the call ends per F-C4.
- **F-C6.** Consent state changes MUST be persisted to `session.json` before
  the corresponding behavior change is observable on the call.

### 3.2 Outcome (inline, end of call)

Captured on the same call before hangup, in the style of a post-support
survey ("stay on the line for a brief survey"). No deferred SMS, no
scheduler.

- **F-O1.** When `Phase.FEEDBACK` is active and ~`OUTCOME_PROMPT_LEAD_SECONDS`
  remain in the feedback budget (default 15s), the runtime MUST cause the
  coach to speak `OUTCOME_PROMPT` (canonical text in
  `rehearse/outcome.py`).
- **F-O2.** The runtime MUST classify the next final user transcript frame
  with `classify_outcome` (deterministic; see §6.3) into one of
  `positive`, `negative`, or `unclear`.
- **F-O3.** On `positive` or `negative`, the runtime MUST write
  `OutcomeLabel(captured_at=now, did_it_help=<bool>, notes=<verbatim user
  text or None if it's exactly Y/N>)` to `session.json`, set
  `Session.outcome_probe_status = "captured"`, and let the call end
  naturally.
- **F-O4.** On `unclear`, the runtime MAY re-prompt up to
  `OUTCOME_REPROMPT_LIMIT` times (default 1). On exhausted retries OR on
  no final transcript frame within `OUTCOME_RESPONSE_TIMEOUT_SECONDS`
  (default 15s), the runtime MUST set
  `Session.outcome_probe_status = "skipped"` and let the call end.
- **F-O5.** The probe MUST run only when `Session.consent == GRANTED`.
  Declined sessions are finalized partial without ever entering feedback.
- **F-O6.** The probe MUST be idempotent within a call: if the user keeps
  talking after a captured label, additional frames MUST NOT overwrite it.
- **F-O7.** Outcome capture MUST NOT block hangup. If the user hangs up
  during the prompt or before answering, the runtime sets
  `outcome_probe_status = "skipped"` during the existing finalize path.

## 4. Non-functional requirements

### 4.1 Legal

- **N-L1.** No audio frame, prosody event, or transcript line MAY be
  written to durable storage when `consent == DECLINED`. Streaming writers
  must respect the consent gate.
- **N-L2.** The transcript fragment retained for declined sessions MUST be
  bounded to the consent exchange (the prompt + the user's first response
  + optional re-prompt + response). No practice-phase content may exist
  for declined sessions.
- **N-L3.** Outcome SMS opt-out: any inbound message containing `STOP`,
  `UNSUBSCRIBE`, or `CANCEL` (case-insensitive) MUST be honored — record
  the opt-out and never send another outbound to that number. Twilio
  enforces this at the carrier level too; we mirror it for safety.
- **N-L4.** Phone numbers MUST stay hashed everywhere except the active
  in-memory `SessionHandle`. The outcome route hashes inbound `From`
  before matching.

### 4.2 Reliability

- **N-R1.** Consent classification MUST run in under 200ms p95. The user
  hears no delay between their answer and the next coach turn.
- **N-R2.** The outcome scheduler MUST tolerate process restarts. State
  lives on disk in `session.json`, not in memory.
- **N-R3.** A failed outbound outcome SMS MUST be retried up to 3 times
  with exponential backoff before the session is marked
  `outcome_dispatch_failed`. No silent drops.
- **N-R4.** The inbound outcome route MUST be idempotent: a duplicate
  Twilio webhook delivery for the same SMS MUST not double-write the
  label.

### 4.3 Privacy

- **N-P1.** The default consent prompt names exactly three things: that the
  call is recorded, that the recording is used to give the caller feedback,
  and that the caller can decline. No marketing, no upsell.
- **N-P2.** Outcome notes are stored verbatim — no LLM rewriting — to keep
  the human signal intact for training.
- **N-P3.** A future "purge by phone number" admin command (out of scope
  here) must be able to delete all sessions for a hashed number. The
  schema already supports this.

## 5. Data sources

| Source | Use | Where it enters the system |
|---|---|---|
| Live transcript frames (`TranscriptDelta`) | Consent classification input | `rehearse/intake.py` consumes from `FrameBus` |
| `Session.consent`, `Session.completion_status`, `Session.created_at` | Outcome scheduler eligibility | Read from `sessions/<id>/session.json` |
| Twilio inbound SMS to the rehearse number | Outcome label payload | New route `POST /twilio/sms/outcome` (or routed inside existing `/twilio/sms` based on prior session existence) |
| Twilio outbound SMS API | Deliver the outcome prompt | `TelephonyClient.send_sms` — already implemented |
| Hashed phone number on `Session.phone_number_hash` | Reverse lookup of session by SMS sender | Hashed via `_hash_number` (already in `session.py`) |

No new external data dependencies. Everything reuses the existing Twilio +
`LocalFilesystemStore` surface.

## 6. Model selection

### 6.1 Consent classification

**Choice:** deterministic regex / keyword classifier. No LLM.

Reasoning:
- The decision is binary and the input vocabulary is small.
- Latency budget is ~200ms p95; an LLM round-trip risks blowing it.
- Legal exposure prefers a classifier whose behavior we can inspect and
  unit-test exhaustively over one whose behavior is emergent.
- A misclassification toward `unclear` is safe (we re-prompt); a confident
  false `granted` from a hallucinating LLM is not.

Implementation sketch (in `rehearse/personas.py`):

```
AFFIRMATIVE = {"yes", "yeah", "yep", "sure", "okay", "ok", "go ahead",
               "that's fine", "sounds good", "alright", "absolutely"}
NEGATIVE    = {"no", "nope", "not okay", "don't", "do not", "decline",
               "i'd rather not", "stop"}

def classify_consent(text: str) -> Literal["granted", "declined", "unclear"]:
    norm = text.strip().lower().rstrip(".!?")
    if any(norm.startswith(a) or norm == a for a in AFFIRMATIVE): return "granted"
    if any(norm.startswith(n) or norm == n for n in NEGATIVE): return "declined"
    return "unclear"
```

The classifier is unit-tested with a fixture of ~30 phrases covering
common variants, accents, and edge cases ("yes but...", "no problem").

### 6.2 Outcome classification (inline)

**Choice:** deterministic classifier in `rehearse/outcome.py::classify_outcome`,
mirroring `classify_consent`. No LLM for the binary decision.

Reasoning:
- The classifier runs on the live audio loop; latency budget is the same
  ~200ms p95 as consent. An LLM round-trip risks a noticeable gap.
- The output is a binary plus optional verbatim notes — there's nothing
  to summarize or rewrite.
- Misclassification toward `unclear` is safe (re-prompt). False
  confident captures are not — they pollute training data.

Lexicons live alongside `classify_consent` in
`rehearse/personas.py` (or `rehearse/outcome.py`, see implementer's
choice). Suggested seed sets:

```
POSITIVE = {"yes", "yeah", "yep", "useful", "helpful", "great",
            "definitely", "absolutely", "it did", "i think so"}
NEGATIVE = {"no", "nope", "not really", "not useful", "not helpful",
            "wasn't helpful", "didn't help"}
```

Notes capture: when the user says more than the matched keyword (e.g.
"yeah, the part about pacing was useful"), the entire utterance is
stored verbatim as `OutcomeLabel.notes`. When the user says exactly `Y`,
`N`, `yes`, `no`, etc., `notes = None`.

### 6.3 Why no LLM at all in v1

The earlier draft of this spec included a Claude Haiku fallback for
free-form SMS replies ("kind of, she got upset but agreed"). That
ambiguity was a feature of the *deferred-outcome* design where users
type at leisure. Inline capture happens during a live voice call: the
user is already speaking, and the coach can re-prompt for clarity in
under a second. The LLM fallback adds latency and behavior risk for no
gain. If a richer signal becomes important later, add it as a follow-up
spec — do not add it to v1.

## 7. Processing flow

### 7.1 Consent flow (during call)

```
WS /media/{session_id} accepted
  └─ orchestrator.start(...) sets Session.consent = PENDING
  └─ HumeEVIClient connected, FrameBus running
  └─ Coach speaks consent prompt (first utterance)
  └─ IntakeProcessor.consent_gate watches transcript frames
       ├─ first final USER frame
       │     ├─ classify_consent(text) → "granted"
       │     │     └─ store.update_session(consent=GRANTED)
       │     │     └─ emit ConsentResolved frame on bus
       │     │     └─ PhaseProcessor unblocks; intake proper begins
       │     ├─ "declined"
       │     │     └─ store.update_session(consent=DECLINED)
       │     │     └─ Coach speaks acknowledgement
       │     │     └─ orchestrator.finalize(session_id, "partial")
       │     │     └─ purge audio.wav + prosody.jsonl
       │     │     └─ truncate transcript.jsonl to consent exchange
       │     │     └─ skip synthesis + viewer SMS
       │     └─ "unclear"
       │           └─ Coach re-prompts once
       │           └─ next final frame retried; on second "unclear" → declined branch
       └─ 20s timeout with no final USER frame → declined branch
```

The consent gate is a small new collaborator on `IntakeProcessor`. The
phase machine grows one new condition: `_maybe_advance_for_budget` early-
returns when `current_phase == INTAKE and consent != GRANTED`.

### 7.2 Outcome flow (inline, end of call)

Single component: `rehearse/outcome.py::OutcomeProbe`. Lives on the same
`FrameBus` as the existing writers and `IntakeProcessor`. Owned by the
`/media/{session_id}` handler — instantiated alongside the other
phase-aware processors.

```
PhaseProcessor enters Phase.FEEDBACK → emits PhaseSignal
  └─ OutcomeProbe sees the signal
  └─ schedule the prompt utterance ~OUTCOME_PROMPT_LEAD_SECONDS before
     feedback budget elapses (today: 15s before 60s budget = at +45s)
  └─ coach speaks OUTCOME_PROMPT (text in rehearse/outcome.py)
  └─ probe sets outcome_probe_status = "asked", records asked_at
  └─ watch transcript bus for next final USER frame
       ├─ classify_outcome(text) → "positive"
       │     └─ build_label(...) → OutcomeLabel(did_it_help=True, notes=...)
       │     └─ store.update_session(label + status="captured")
       ├─ "negative"
       │     └─ build_label(...) → OutcomeLabel(did_it_help=False, notes=...)
       │     └─ store.update_session(label + status="captured")
       └─ "unclear"
             └─ if reprompts < OUTCOME_REPROMPT_LIMIT: coach re-prompts
             └─ else: store.update_session(status="skipped")
       (timeout: same skipped path after OUTCOME_RESPONSE_TIMEOUT_SECONDS)
  └─ coach speaks closing line; call ends naturally
  └─ existing finalize path runs synthesis (story.md + feedback.md) and
     SMSes the viewer link
```

Edge cases handled:
- User hangs up before answering → existing `/twilio/status` finalize
  observes `outcome_probe_status` is still `"asked"` and rewrites it to
  `"skipped"`.
- User keeps talking after `"captured"` → probe is idempotent; further
  frames are ignored once a label is written.
- Feedback phase ends before the prompt fires (budget overrun upstream)
  → probe sets `outcome_probe_status = "skipped"` and the call ends
  normally.

### 7.3 Component sketch

The shipping shape is a single-file module with one class and two
helpers, all stubbed at `rehearse/outcome.py`:

| Symbol | Purpose |
|---|---|
| `OUTCOME_PROMPT` | Canonical prompt text the coach speaks |
| `OutcomeProbeConfig` | Knobs: response timeout, reprompt limit |
| `OutcomeProbe` | Lifecycle: `run(frames)` consumes the bus, calls `_persist_label` or `_mark_skipped` |
| `classify_outcome(text)` | Deterministic three-way classifier |
| `build_label(classification, body, captured_at)` | Builds an `OutcomeLabel` or returns `None` for unclear |

Wiring: in `telephony.py::media_stream`, register one more task on the
existing bus subscription pattern, alongside `TranscriptWriter`,
`ProsodyWriter`, etc.:

```python
outcome_task = asyncio.create_task(
    OutcomeProbe(session_id, orchestrator.store).run(bus.subscribe())
)
```

Awaited in the same `finally` block as the other writers so it cannot
leak.

## 8. Schema changes

`rehearse/types.py` additions on `Session` (already stubbed in code):

```python
finalized_at: datetime | None = None
outcome_probe_status: Literal["pending", "asked", "captured", "skipped"] | None = None
```

`finalized_at` is set by `SessionOrchestrator.finalize` and is durable
state (used so a restart between feedback and finalize does not lose
the probe's status interpretation).

`outcome_probe_status` traces the inline probe's lifecycle on disk:
- `None` — call has not reached feedback phase yet.
- `"pending"` — feedback active, prompt not yet spoken.
- `"asked"` — coach has spoken `OUTCOME_PROMPT`; awaiting user reply.
- `"captured"` — `OutcomeLabel` is populated.
- `"skipped"` — probe ran but no usable label was captured (timeout,
  unclear, or hangup mid-prompt).

No removals. No breaking changes to existing artifacts. Existing test
fixtures load unchanged thanks to pydantic defaults — verified by the
test suite (`tests/test_synthesis.py`, `tests/test_session_storage.py`,
`tests/test_types.py` all pass after stubbing).

`OutcomeLabel` and `ConsentState` are unchanged — they were already
sufficient.

The earlier draft included an opt-out registry and three SMS-dispatch
status fields on `Session`. With the inline pivot none of that is
needed; removed.

## 9. Configuration

New environment variables, all with safe defaults:

| Var | Default | Purpose |
|---|---|---|
| `CONSENT_PROMPT_TIMEOUT_SECONDS` | `20` | How long to wait for the user's first consent response |
| `CONSENT_REPROMPT_LIMIT` | `1` | Number of re-prompts before treating "unclear" as declined |
| `OUTCOME_PROMPT_LEAD_SECONDS` | `15` | Seconds before feedback budget elapses to fire the outcome prompt |
| `OUTCOME_RESPONSE_TIMEOUT_SECONDS` | `15` | How long to wait for the user's first outcome response |
| `OUTCOME_REPROMPT_LIMIT` | `1` | Number of re-prompts before marking probe `"skipped"` |

Loaded by `RuntimeConfig.from_env`. Existing config tests get one
parametrized case per new var.

## 10. Observability and logging

All emitted via `structlog` to match the existing runtime conventions.
No new infra.

### 10.1 Consent

| Event | Fields | When |
|---|---|---|
| `consent.prompt.spoken` | `session_id` | Coach delivers consent prompt |
| `consent.classify` | `session_id, classification, text_len` | After each parse attempt; `text_len` not text — privacy |
| `consent.granted` | `session_id, latency_ms` | On grant; latency = prompt → grant |
| `consent.declined` | `session_id, reason ∈ {"explicit","timeout","unclear"}` | On decline |
| `consent.purge` | `session_id, files_removed` | After audio/prosody purge in declined branch |

Counter to track over time: `granted / (granted + declined + timeout)`.
Alert if it drops below 0.7 — likely indicates the consent prompt is
confusing or the classifier is too strict.

### 10.2 Outcome

| Event | Fields | When |
|---|---|---|
| `outcome.prompt.spoken` | `session_id, lead_seconds` | Coach delivers the outcome prompt |
| `outcome.classify` | `session_id, classification, text_len` | After each parse attempt; text_len, not text |
| `outcome.captured` | `session_id, did_it_help, has_notes, latency_ms` | Label persisted; latency = prompt → capture |
| `outcome.reprompt` | `session_id, attempt` | A re-prompt is being spoken |
| `outcome.skipped` | `session_id, reason ∈ {"timeout","unclear","hangup"}` | Probe marked skipped |

Counter: outcome capture rate = `captured / (captured + skipped)`.
Target ≥ 0.7. Below 0.5 means the prompt copy or timing is wrong.

### 10.3 Telemetry

Consent latency (`consent.granted.latency_ms`) and outcome capture
latency (`outcome.captured.latency_ms`) are appended to
`telemetry.jsonl` so they show up in eval-time replays.

## 11. Test plan

Unit:
- `classify_consent` over a 30-phrase fixture (all branches).
- `classify_outcome` over a 20-phrase fixture (positive / negative /
  unclear, with and without trailing notes).
- `build_label`: Y/N → `notes=None`; "yes, the pacing thing was useful"
  → `notes` carries the verbatim text; `unclear` → `None`.

Integration:
- End-to-end consent declined path: build a fake `FrameBus`, push a
  consent prompt + a "no" frame, assert `consent == DECLINED`,
  `audio.wav` absent, transcript truncated, no `story.md`.
- End-to-end outcome captured path: drive a fake bus through intake →
  practice → feedback, fire `OutcomeProbe`, push a "yeah it was useful"
  frame, assert `outcome_label.did_it_help is True`,
  `outcome_probe_status == "captured"`, notes carry the verbatim text.
- Outcome timeout path: same setup, no user response within
  `OUTCOME_RESPONSE_TIMEOUT_SECONDS`, assert
  `outcome_probe_status == "skipped"` and no `OutcomeLabel`.
- Hangup-during-probe path: WS closes after `outcome.prompt.spoken` but
  before any user frame arrives. Assert finalize promotes status to
  `"skipped"` rather than leaving it `"asked"`.

Replayability:
- A frozen captured session round-trips through `SessionSynthesizer`
  unchanged — the probe is purely additive on top of synthesis input.

## 12. Rollout

Single PR. No feature flag — these features are required before the next
non-founder uses the system, and there's no value in shipping the runtime
without them in that mode.

Backward compatibility: existing sessions on disk lack the new
`outcome_dispatched_at` / `finalized_at` fields. The pydantic defaults
(`None`) cover this; `OutcomeScheduler` skips sessions where
`finalized_at is None` rather than synthesizing one.

## 13. Success criteria

The workstream is complete when all of the following are true:

1. A real call where the user says "yes" to the consent prompt completes
   normally with `consent == GRANTED` on `session.json`.
2. A real call where the user says "no" ends within ~5 seconds, leaves no
   `audio.wav` or `prosody.jsonl`, and writes a truncated
   `transcript.jsonl` containing only the consent exchange.
3. On a real granted call that reaches the feedback phase, the coach
   speaks `OUTCOME_PROMPT` ~15s before hangup, the user's spoken reply
   is captured as an `OutcomeLabel`, and `outcome_probe_status ==
   "captured"` is persisted before finalize.
4. A real call where the user is silent through the outcome probe ends
   normally with `outcome_probe_status == "skipped"` and no
   `OutcomeLabel`.
5. The unit + integration tests in §11 all pass and run in under 5
   seconds on local hardware.
6. Existing end-to-end runtime tests (`tests/test_synthesis.py` etc.)
   continue to pass with the new fields and component in place.

## 14. Decision rule

If you're adding code under this spec, ask:

**Does this code exist to either (a) gate recording on explicit caller
agreement or (b) capture and persist the post-call outcome label?**

- If yes, it belongs here.
- If no — even if it touches `intake.py` or `telephony.py` — it belongs
  in a separate PR, not in this workstream.

## 15. Pickup notes (handoff)

This spec was amended on 2026-05-05 to swap the deferred-outcome SMS
flow for an inline post-call survey ("stay on the line"). The next
engineer inherits the following starting state:

**Already landed in code:**
- Two new fields on `Session` in `rehearse/types.py`:
  `finalized_at: datetime | None`,
  `outcome_probe_status: Literal["pending","asked","captured","skipped"] | None`.
  Both default to `None` and existing tests pass (`pytest -q` confirms 9/9
  for synthesis + storage + types).
- `rehearse/outcome.py` exists with the public surface stubbed:
  `OUTCOME_PROMPT`, `OutcomeProbeConfig`, `OutcomeProbe.run`,
  `OutcomeProbe._persist_label`, `OutcomeProbe._mark_skipped`,
  `classify_outcome`, `build_label`. Every method raises
  `NotImplementedError` with a pointer to the spec section that
  describes it.

**What still needs to land (suggested order):**
1. `classify_outcome` + `build_label` + a unit-test fixture in
   `tests/test_outcome.py`. Mirror the pattern in
   `tests/test_synthesis.py`.
2. `OutcomeProbe.run` lifecycle. Consume from a `FrameBus` subscription
   the same way `IntakeProcessor.run` does; gate on
   `PhaseSignal(to_phase=Phase.FEEDBACK)`.
3. Coach speaker channel for the prompt utterance. Today the coach's
   speech path is via the CLM webhook stream. Decide: do we inject the
   prompt via the CLM stream (simpler, but couples to LLM availability)
   or via a dedicated Hume "say this" call (more robust, requires a
   small `HumeEVIClient` extension)? Spec is agnostic; recommend the
   second option for reliability.
4. Wiring in `rehearse/telephony.py::media_stream`. Add one more
   `asyncio.create_task` next to the writers, await it in the same
   `finally` block.
5. Consent gate (§3.1). Extend `IntakeProcessor` with a
   `consent_gate` collaborator, block phase transitions until granted,
   purge audio/prosody on decline.
6. `RuntimeConfig` knobs from §9. Wire through the existing
   `from_env` constructor.
7. Observability events from §10.
8. Integration tests from §11. The synthesis test file is a good
   pattern reference for fixture-based async testing.

**Existing assets to lean on:**
- `rehearse/intake.py::IntakeProcessor` — same architectural pattern
  (`run(frames: AsyncIterator[Frame])`, `update_session` mutations).
- `rehearse/synthesis.py::SessionSynthesizer` — already produces the
  call summary (`story.md`) and feedback (`feedback.md`), so this
  workstream does not need to build any summarization. The probe runs
  *before* synthesis on the same call and its label is part of the
  frozen artifact set synthesis reads.
- `rehearse/storage.py::LocalFilesystemStore.update_session` — atomic
  manifest mutation, the only way to update `Session` fields.

**What was removed in the amendment** (do not resurrect without a new
spec): the `OutcomeScheduler` background task, the
`/twilio/sms/outcome` route, the opt-out registry, the LLM fallback
parser, the `outcome_dispatched_at` / `outcome_dispatch_status` /
`outcome_message_sid` fields. The git history of this file shows the
prior shape if you need it.

**What is intentionally deferred:**
- Real-world outcome capture ("did the actual conversation go well a
  day later"). This is a stronger training signal than call-quality.
  Add as a follow-up spec when the data loop demands it. The
  inline-captured `OutcomeLabel.did_it_help` is sufficient for v1
  preference-pair mining.

## Completion note (2026-05-05)

Landed in this PR (R8 workstream). Test verification only — no real-call
verification yet; the success criteria in §13 still need the four-call
walk-through described in the brief.

Implemented:
- Deterministic `classify_outcome` + `build_label` lexicons in
  `rehearse/outcome.py`; deterministic `classify_consent` lexicons in
  `rehearse/personas.py`. Both mirror the same pattern: lowercase, strip
  trailing punctuation, start-of-string match against affirmative /
  negative phrase sets, default to `unclear`.
- `OutcomeProbe` (`rehearse/outcome.py`) with the full lifecycle from
  spec §7.2: watches `PhaseSignal(to_phase=FEEDBACK)`, schedules the
  prompt `feedback_budget − prompt_lead` seconds later, classifies the
  next final user transcript frame, persists `OutcomeLabel` with verbatim
  notes (or `None` for bare yes/no), re-prompts up to
  `OUTCOME_REPROMPT_LIMIT`, marks `"skipped"` on timeout/unclear/hangup,
  is idempotent post-capture.
- `ConsentGate` (`rehearse/consent.py`) covering §3.1 / §7.1: speaks the
  prompt as the first coach utterance, classifies replies, re-prompts on
  unclear, declines on `no` / timeout / repeated unclear, calls an
  injected `on_decline` callback that the WS handler uses to break the
  audio loop and trigger `orchestrator.finalize(_, "partial")`.
- Phase machine gate: `PhaseProcessor` accepts a `consent_getter`; the
  budget and cue advance paths early-return while
  `current_phase == INTAKE and consent != GRANTED`.
- Decline-aware `SessionOrchestrator.finalize`: when `consent ==
  DECLINED`, skips synthesis and viewer SMS; purges `audio.wav` and
  `prosody.jsonl`; truncates `transcript.jsonl` to at most 2 user finals
  (the bound implied by §N-L2: prompt + response + optional
  re-prompt + response); drops the `audio` and `prosody` keys from the
  manifest's `artifact_paths`. Also promotes any still-`asked` /
  `pending` `outcome_probe_status` to `"skipped"` during normal finalize.
- Coach speaker channel: `HumeEVIClient.say(text)` wraps
  `send_assistant_input(AssistantInput(text=...))` (verified in the
  installed `hume` SDK at `socket_client.py:157`); the consent gate and
  outcome probe both receive `hume.say` as their `speak` callable.
- `ConsentResolved` frame added to `rehearse/frames.py`. `EndOfCall`
  reasons extended with `"consent_decline"`.
- `RuntimeConfig` knobs from §9: `CONSENT_PROMPT_TIMEOUT_SECONDS`,
  `CONSENT_REPROMPT_LIMIT`, `OUTCOME_PROMPT_LEAD_SECONDS`,
  `OUTCOME_RESPONSE_TIMEOUT_SECONDS`, `OUTCOME_REPROMPT_LIMIT`. All
  default to spec-prescribed values; new parametrized tests cover each.
- Observability per §10: structlog events (`consent.prompt.spoken`,
  `consent.classify`, `consent.granted`, `consent.declined`,
  `consent.purge`, `outcome.prompt.spoken`, `outcome.classify`,
  `outcome.captured`, `outcome.reprompt`, `outcome.skipped`). Latency
  metrics (`consent.granted.latency_ms`, `outcome.captured.latency_ms`)
  appended to `telemetry.jsonl` per §10.3.
- Telephony wiring (`rehearse/telephony.py::media_stream`):
  `consent_task` and `outcome_task` join the existing
  `asyncio.create_task` block beside the writers; both awaited in the
  same `finally`. The audio inbound loop checks a `declined` flag and
  breaks; finalize-partial runs once after task drain.

Tests added / extended (run with `uv run pytest -q`, 167 passing):
- `tests/test_outcome.py` (33 cases): 24 classifier rows, 4 label rows,
  5 lifecycle paths (capture, timeout, reprompt-then-skip, idempotent
  after capture, hangup-after-prompt).
- `tests/test_consent.py` (28 cases): 23 classifier rows, 3 gate
  lifecycle paths (grant, decline, reprompt-then-decline), 2
  finalize-on-decline paths (artifact purge + transcript truncation,
  outcome-status promotion to skipped).
- `tests/test_config.py`: parametrized coverage of all five new knobs +
  default-values check.
- `tests/test_telephony_r1.py`: existing fake updated to add `say` and
  to send a `"yes"` consent reply before practice content, so the new
  consent gate doesn't decline the synthetic call.

Verification (this PR):
- `uv run pytest -q` → 167 passed, 2 deselected, ~3s.
- `uv run ruff check rehearse tests` → 2 errors, both preexisting and
  not in code I touched (`voice_agent_sandbox.py:318`, a long line in
  `tests/eval/test_coach_dialogue_smoke.py:35`).
- `uv run mypy rehearse` → 38 errors, all preexisting (missing
  third-party stubs for `structlog`/`fastapi`/`twilio`/`hume`, the
  pre-existing `phases.py::_transition` reason-type mismatch, and the
  pre-existing `telephony.py::twilio_status` CompletionStatus narrowing).
  None introduced by this PR.

Decision log:
- Coach speaker channel: chose Hume `send_assistant_input` over
  CLM-stream injection for the consent and outcome utterances. The
  spec recommended this (§15 item 3) for reliability — the deterministic
  prompts no longer depend on the LLM path, and the production wiring
  is one tiny method (`HumeEVIClient.say`) injected into both
  components.
- Decline-path transcript bound: implemented as a hard cap of 2 user
  finals during `finalize`, not a timestamp-based truncation. This is
  defense-in-depth on top of the production flow (which already closes
  the bus on decline before any practice content can be written). The
  cap matches the §N-L2 maximum of "first response + optional re-prompt
  + response."
- Outcome probe lead time: re-using `feedback_budget_seconds` from
  `PhaseBudgets` rather than introducing a separate "fire after N
  seconds" knob. The probe sleeps `feedback_budget − prompt_lead`
  seconds after the FEEDBACK signal, then asks. Default
  `60 − 15 = 45s` matches §F-O1.

Real-call verification (still pending, per the brief's success criteria
§13): four phone calls — granted, declined, captured, silent-outcome —
to confirm the manifest fields land as specified. All deterministic logic
is exercised by the test suite; only the Hume `say` round-trip and the
WS-loop decline shutdown remain to be validated on a live call.
