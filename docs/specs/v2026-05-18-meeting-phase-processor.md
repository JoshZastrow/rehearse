# MeetingPhaseProcessor — LLM-Driven Phase Transitions

**Date**: 2026-05-18
**Status**: `acknowledged`
**Policy**: `implementation`
**Applies to**: Phase transition decision-making in the live call runtime
**Amends**: `v2026-04-27-runtime.md` §4 (phase machine), `v2026-05-14-conversation-backend.md`

---

## 1. Problem

The current `PhaseProcessor` is a fixed heuristic machine: time budgets fire
transitions after N seconds, and a frozen set of cue phrases triggers early
transitions. These rules cannot improve — they only accumulate exceptions. The
LLM generating coach responses has full conversational context but zero
influence over when phases change.

## 2. Goal

Replace the heuristic phase machine with a small, fast LLM that observes the
live conversation and calls a `transition_phase` tool when it judges a phase
is complete. The decision is autonomous, observable (via the `reason` field),
and improvable via outcome-scored data collection.

## 3. Design

### 3.1 Architecture

`MeetingPhaseProcessor` is a drop-in replacement for `PhaseProcessor`. It runs
as a standalone `asyncio.Task` that subscribes to the shared `FrameBus` — the
same pattern as today. From the outside, nothing changes: it receives frames
and emits `PhaseSignal` frames. The `PhaseSignal` contract is unchanged.

```
TranscriptDelta(USER, is_final=True)
        │
        ▼
MeetingPhaseProcessor (asyncio.Task)
  ├─ build conversation window (last N turns)
  ├─ check min dwell guard
  ├─ call Classifier LLM with transition_phase tool
  │       model: claude-haiku-4-5 (or equivalent cheap/fast model)
  │       tool_choice: auto
  └─ if tool called → emit PhaseSignal → persist manifest
```

### 3.2 Tool Protocol

The classifier receives one tool:

```python
TRANSITION_PHASE_TOOL = {
    "name": "transition_phase",
    "description": (
        "Move the conversation to the next phase when the current phase goals "
        "are complete. Do not call this tool if there is any uncertainty — "
        "only call it when the evidence in the transcript is clear."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "enum": ["practice", "feedback", "survey"],
                "description": "The target phase."
            },
            "reason": {
                "type": "string",
                "description": (
                    "One sentence explaining what in the transcript signals "
                    "the current phase is complete. This is stored as a "
                    "learning signal."
                )
            }
        },
        "required": ["to", "reason"]
    }
}
```

`tool_choice: {"type": "auto"}` — the model decides whether to call the tool
on each turn. If it does not call the tool, no transition fires and the
conversation continues.

### 3.3 Input: Conversation Window

On each qualifying frame (`TranscriptDelta`, `speaker=USER`, `is_final=True`),
the processor builds a rolling window of the last `context_turns` transcript
turns (default 10, configurable). The window is passed as `messages` to the
classifier.

The system prompt provides:
- The current phase name and its goal
- What "complete" looks like for each phase (replacing the frozen cue sets)
- An instruction to be conservative — only call the tool when clearly warranted

### 3.4 Phase Goals (System Prompt Content)

| Phase | Goal | Completion signal |
|---|---|---|
| `INTAKE` | Collect scenario, counterparty, and stakes | User has named the situation and is ready to rehearse |
| `PRACTICE` | Run the roleplay | User has completed at least one exchange and signals they are done |
| `FEEDBACK` | Debrief and coach | User has received feedback and signals they are done |

These replace `_INTAKE_READY_CUES` and `_FEEDBACK_READY_CUES`.

### 3.5 Min Dwell Guard

The processor retains `PhaseBudgets.min_dwell_for(phase)` as a floor before
any classifier call fires. If the current phase has not reached its minimum
dwell time, the transcript frame is buffered but no LLM call is made. This
prevents premature transitions on the first user utterance and keeps
classifier cost bounded.

Budget-driven transitions (time budget exhausted) are removed. The LLM is
the sole decision-maker after the min dwell floor.

### 3.6 Transport: Output

When the tool is called, `MeetingPhaseProcessor` emits a `PhaseSignal` frame
on the bus (identical to the current `PhaseProcessor._transition()` path) and
persists the manifest change via `LocalFilesystemStore.update_session`. The
`reason` string is written to `telemetry.jsonl` alongside the transition event
for later scoring against session outcomes.

```jsonl
{"event": "phase_transition", "from": "intake", "to": "practice", "reason": "...", "ts": "..."}
```

### 3.7 LLM Client

The classifier uses a separate `AsyncAnthropic` client instance with
`model="claude-haiku-4-5-20251001"`. No gateway (LiteLLM) is required for the
initial implementation. The model string is configurable via
`RuntimeConfig.phase_classifier_model` so providers can be swapped without
code changes.

The classifier client is instantiated once and shared across turns within a
session. It does not share state with the conversational coach client.

### 3.8 Latency

The classifier call is asynchronous and does not block the conversational
coach. It runs in the same asyncio event loop as the other bus subscriber
tasks. Classifier latency is typically 200–400ms for Haiku on a short window;
this is invisible to the caller because the coach has already responded by the
time the transition fires.

## 4. Config

Two fields added to `RuntimeConfig`:

| Field | Env var | Default | Notes |
|---|---|---|---|
| `phase_classifier_model` | `PHASE_CLASSIFIER_MODEL` | `claude-haiku-4-5-20251001` | Classifier model string |
| `phase_classifier_context_turns` | `PHASE_CLASSIFIER_CONTEXT_TURNS` | `10` | Rolling window size |

`enable_meeting_phase_processor: bool` is added as a flag to toggle between
`PhaseProcessor` (heuristic, default `False`) and `MeetingPhaseProcessor`
(LLM-driven, default `False` until validated). Both processors implement the
same interface and are interchangeable at the call site in `telephony.py` and
`RuntimeHost`.

## 5. Files

| File | Change |
|---|---|
| `rehearse/phases_llm.py` | New file — `MeetingPhaseProcessor` class |
| `rehearse/config.py` | `phase_classifier_model`, `phase_classifier_context_turns`, `enable_meeting_phase_processor` fields |
| `rehearse/telephony.py` | Toggle between `PhaseProcessor` and `MeetingPhaseProcessor` based on config |
| `rehearse/runtime.py` | Same toggle for the eval path |
| `docs/specs/MANIFEST.md` | Add this spec |

`rehearse/phases.py` is unchanged. `PhaseProcessor` remains available.

## 6. Learning Loop (Future)

The `reason` field written to `telemetry.jsonl` is the foundation for a
supervised improvement loop:

1. Score `(conversation_window, reason, transition_timestamp)` against session
   outcome (survey score, user-reported goal achievement)
2. Filter for high-quality transitions (correct timing, coherent reason)
3. Fine-tune the classifier or use the data for prompt optimization

This is out of scope for this spec. The telemetry schema is designed to
support it.

## 7. Acceptance Criteria

- [ ] `MeetingPhaseProcessor` subscribes to the bus and emits `PhaseSignal`
      with the same fields as `PhaseProcessor`
- [ ] Min dwell guard fires correctly — no LLM call before floor elapsed
- [ ] `reason` string written to `telemetry.jsonl` on every transition
- [ ] `enable_meeting_phase_processor=False` (default) uses old `PhaseProcessor`
      with no behavioural change
- [ ] Eval harness passes with `MeetingPhaseProcessor` enabled
- [ ] Classifier cost per session logged in telemetry

## 8. Out of Scope

- Budget-driven transitions (removed — LLM decides)
- LiteLLM gateway (additive later if multi-provider needed)
- Fine-tuning or RL training loop (gated on telemetry collection first)
- Changes to `PhaseSignal`, `FrameBus`, or `LocalFilesystemStore`
