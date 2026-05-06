# Time-aware CLM (per-turn time card injection)

**Status:** draft
**Date:** 2026-05-06
**Owner:** Josh Zastrow
**Related code:** `rehearse/agents/clm.py`, `rehearse/phases.py`, `rehearse/personas.py`

## Goal

Keep every rehearsal call under Hume's 5-minute model-provider cap by giving the
CLM live awareness of which segment it is in, how much time is left, and how
many words it should speak on the current turn. Today the model has the right
system prompt for the active phase but no signal about elapsed/remaining time,
so it drifts past phase boundaries and over-runs the call budget.

## Non-goals

- Splitting the CLM into multiple endpoints or models. One Anthropic call per
  turn, same webhook, same role-routing as today.
- Hard scripted bridge lines on phase transitions ("option B" from
  brainstorming). We may add this later if soft steering proves insufficient.
- Changing the existing `PhaseProcessor` budgets, cue heuristics, or transition
  logic.
- Changing Hume EVI configuration or audio path.

## Design

### TimeCard

A small dataclass built fresh on every CLM request:

```python
@dataclass(frozen=True)
class TimeCard:
    phase: Phase
    seconds_elapsed_in_phase: int
    seconds_remaining_in_phase: int
    seconds_remaining_in_call: int
    word_budget_this_turn: int
```

Built from the session manifest's `phase_timings` (the open row gives
`started_at`; budget gives the cap) plus a clock. `seconds_remaining_in_call`
is `300 - (now - call_started_at)` clamped to zero, where `call_started_at` is
the `started_at` of the first phase timing.

### Word-budget formula

```
model_share         = {INTAKE: 0.50, PRACTICE: 0.30, FEEDBACK: 0.70}[phase]
avg_turn_seconds    = 15
expected_turns_left = max(1, seconds_remaining_in_phase / avg_turn_seconds)
words_per_second    = 160 / 60
words_this_turn     = (seconds_remaining_in_phase * model_share * words_per_second)
                      / expected_turns_left
word_budget         = clamp(words_this_turn, 15, 80)
```

### Rendered string

Appended to the system payload as a separate, uncached block:

```
Live timing
- Phase: practice (3:00 budget)
- Elapsed in phase: 0:42
- Remaining in phase: 2:18
- Remaining in call: 3:18 (hard cap)
- Target length for THIS reply: ~36 words
Speak only as long as the target. When phase time is nearly up, land the
current beat in one closing sentence.
```

### Integration

`clm.py::_handle_clm_request` already loads the session and resolves the role.
Add one step in `AnthropicCLMResponder.stream_reply`:

1. Build `TimeCard` from `session` + clock.
2. Render it to the string above.
3. Send `system` as a list of two blocks instead of one string:
   - Block 1: existing `_system_prompt_for_role(role, session)` output, with
     `cache_control={"type": "ephemeral"}`.
   - Block 2: the time card text, no cache control.

The `ScriptedCLMResponder` ignores the time card (fallback path, not used in
production calls).

### Why two system blocks

Anthropic's `system` parameter accepts a list of blocks with per-block cache
control. The static persona/coach prompt is identical every turn for a given
session and is the largest single chunk we send. Caching it gives ~5-token
cache reads after the first turn instead of re-billing the full prompt on
every CLM call. The time card is short and changes each turn, so it sits
outside the cache.

### Where the time card is NOT

- Not added to the `messages` array. Hume owns message history; we only
  control `system`.
- Not appended to the static prompt as a string concatenation. That would
  bust the prompt cache on every turn.

## Open questions

- Should `avg_turn_seconds` be tuned per phase? (Practice turns may be
  longer than feedback turns.) Defer until we have call data.
- Should the time card include the persona name during practice and the
  user's stated goal during intake? Probably yes for the static prompt, but
  those are session-stable so they belong in block 1, not the time card.

## Test plan

- Unit: `TimeCard.build(session, now)` for each phase given a stub manifest;
  verify clamping, hard-cap floor, formula at boundaries (start of phase,
  end of phase, post-cap).
- Unit: rendered string contains phase name, MM:SS-formatted remaining,
  and word target.
- Integration: `AnthropicCLMResponder.stream_reply` patched to capture the
  outgoing `system` payload — assert it is a list of two blocks, the first
  has `cache_control` set, the second contains "Live timing".
- Manual: one live call through Hume, watch session viewer to confirm phase
  transitions and that the model wraps within phase budgets.

## Out of scope (follow-ups)

- Scripted handoff lines on `PhaseSignal` (option B in brainstorming).
- Per-phase model selection (cheap intake, strong practice, reflective
  feedback).
- Dynamic adjustment of `model_share` based on observed user verbosity.
