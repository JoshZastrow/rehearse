# rehearse — Spec: Persona / Config Swap Path

**Status**: draft
**Owner**: jz
**Date**: 2026-05-06
**Depends on**: `v2026-04-27-runtime.md`, `v2026-04-28-drop-pipecat.md`, `v2026-05-06-persona-routing.md`
**Affects**: `rehearse/services/hume_configs.py`, `rehearse/services/hume_evi.py`, `rehearse/agents/clm.py`, `rehearse/personas.py`, `rehearse/phases.py` (consumer of `PhaseSignal`)

---

## 0. One-line summary

When the runtime advances `Phase.INTAKE → PRACTICE → FEEDBACK`, the live
voice must change role and prompt accordingly: coach → in-character
counterparty → coach. Today the manifest transitions but the live voice
does not, because Hume runs Anthropic directly with one static prompt and
never reaches our phase-aware CLM webhook.

## 1. Outcome

### 1.1 What's broken (evidence)

Session `ca4299ff49b54902a501725d1cc83b02` (2026-05-06):

- `session.json.phase_timings`: intake → practice → feedback all transitioned in
  the manifest within the first 53 seconds.
- `transcript.jsonl`: the assistant kept asking intake-style elicitation
  questions ("Who are you planning to talk to?", "What outcome are you hoping
  for?") for the full 4+ minute call.
- The user explicitly called this out at u22: *"Hey, we're already four
  minutes in. This is supposed to switch to feedback."*

### 1.2 Root cause

`PERSONAS["default"].language_model` is `provider="ANTHROPIC"`. With that
setting, Hume routes each turn directly to Anthropic using only the static
`prompt_text` baked into the Hume config. Our `/chat/completions` (CLM
webhook) is never called, so the `_resolve_role` logic in
`rehearse/agents/clm.py` (which already swaps coach↔character based on
the open phase) never fires.

The static prompt in the Hume config explains all three phases conceptually
("1. INTAKE … 2. PRACTICE … 3. FEEDBACK"), but a single LLM context can't
reliably decide on its own when to flip from elicitation to roleplay to
debrief on a 5-minute call.

### 1.3 Success criteria

After this lands:

1. On the next live call where intake → practice transitions in the
   manifest, the assistant audibly stops asking intake questions and
   begins playing the compiled counterparty in-character.
2. On practice → feedback, the assistant audibly drops character and gives
   one specific reflection + one concrete next-line suggestion.
3. The transition is visible in the transcript: a short bridge utterance
   ("Okay, let's start. I'm the CEO now.") followed by in-character
   speech.
4. Replayability holds: synthesis on the resulting artifacts produces
   correct citations because the transcript reflects the role the
   assistant was playing per turn.

## 2. Audience

The engineer wiring per-phase persona switching for the live runtime.
Assumes familiarity with Hume EVI configs (`rehearse-hume` registry),
the FastAPI CLM endpoint, and `rehearse.phases.PhaseProcessor`.

## 3. Inputs and outputs

### 3.1 Inputs

| Input | Source | Used by |
|---|---|---|
| `PhaseSignal(from_phase, to_phase, reason)` | `phases.PhaseProcessor` (`bus.publish`) | `PersonaSwapCoordinator` (new) |
| `Session` manifest (`phase_timings`, `persona`, `intake`, `persona_key`) | `LocalFilesystemStore` | `clm._resolve_role`, `clm._system_prompt_for_role` |
| `CLMChatRequest.messages` | Hume calling our `/chat/completions` | `AnthropicCLMResponder.stream_reply` |
| `RuntimeConfig.hume_clm_url` (new) + `hume_clm_secret` | env | Hume persona config `language_model` |

### 3.2 Outputs

| Output | Consumer | Format |
|---|---|---|
| Per-turn system prompt for the live LLM | Hume (via our CLM stream) | `coach` / `character` / `feedback_coach` system text |
| Optional bridge utterance on phase transition | Hume `assistant_input` | one short sentence (≤15 words) |
| Updated `transcript.jsonl` rows | writers | `phase` field reflects the live phase at utterance time |
| `telemetry.jsonl` `role` field per assistant turn | `TelemetryLogger` (extended) | `coach \| character \| feedback_coach` |

### 3.3 Non-outputs (explicitly out of scope)

- **Voice swap mid-call.** Voice is pinned by Hume `config_id`. Hot-swap
  of `config_id` is `NotImplementedError` and Hume's support is
  unconfirmed (runtime spec Q1). One voice for the whole call. Voice
  variation per role is a v2 feature.
- **Per-phase model swap.** All three phases run on the same Anthropic
  model in v1.

## 4. Requirements

### 4.1 Functional

- **F1.** When the live phase is `INTAKE`, the assistant uses the coach
  prompt: warmly elicit situation, counterparty, goal, stakes — one
  question per turn, no roleplay.
- **F2.** When the live phase is `PRACTICE`, the assistant uses the
  character prompt compiled from `Session.persona`. It speaks as the
  counterparty (relationship, stakes, hot buttons, likely reactions).
  No coaching mid-scene.
- **F3.** When the live phase is `FEEDBACK`, the assistant uses a
  feedback-coach prompt: drops character, reflects one specific thing
  the user did well, suggests one concrete next line, cites the user's
  actual words.
- **F4.** On every `PhaseSignal(to_phase=…)`, the runtime emits one
  short bridge utterance to Hume so the user hears the role change.
  The bridge is deterministic copy, not LLM-generated, to keep latency
  predictable.
- **F5.** The CLM webhook resolves role from the *current* phase on the
  manifest at request time, not from a cached value. (The phase may
  change between two consecutive Hume turns.)
- **F6.** If the manifest cannot be loaded (e.g. early intake), the
  CLM webhook falls back to `coach`.
- **F7.** The Hume persona configs in `rehearse/services/hume_configs.py`
  are updated to route `language_model` to our CLM endpoint, not directly
  to Anthropic. Bearer auth via `HUME_CLM_SECRET` is honored.
- **F8.** Bridge utterances are emitted via `HumeEVIClient.say(...)` (an
  `assistant_input` to Hume), so they land in the transcript as
  `assistant_message` with `phase = to_phase`.

### 4.2 Non-functional

- **NF1. Latency.** Adding our CLM hop must not blow the 800ms p50
  utterance-end → first audio budget. Bridge utterance is fixed copy
  (zero LLM latency). CLM webhook uses prompt caching on the static
  system prompt and the persona block.
- **NF2. Robustness.** A failure in the CLM webhook (timeout, 5xx) must
  not silently fall back to a wrong-role prompt. On error, return a
  fallback line ("hold on a moment") and log; do not let Hume's
  pre-CLM static prompt take over silently.
- **NF3. Observability.** Every CLM turn logs
  `{session_id, role, phase, latency_ms}`.
- **NF4. Replayability.** Synthesis on frozen artifacts must work
  without contacting Hume or our live CLM. The role stamped on each
  transcript row + the manifest's phase_timings is the source of
  truth for post-call reasoning.
- **NF5. Backwards compatibility.** Sessions whose Hume persona config
  still has `provider="ANTHROPIC"` (existing remote configs) continue
  to function; the roll-out is gated by re-syncing personas via
  `rehearse-hume`.

## 5. Implementation

### 5.1 Component shape

Add one component, modify three:

```
PhaseProcessor ─publishes─▶ PhaseSignal ─▶ PersonaSwapCoordinator (NEW)
                                              │
                                              └─speak via HumeEVIClient.say(bridge_text)

Hume turn ────POST /chat/completions────▶ clm._resolve_role(session_id)
                                              │
                                              ├─ phase == INTAKE   → coach prompt
                                              ├─ phase == PRACTICE → character prompt
                                              └─ phase == FEEDBACK → feedback-coach prompt
```

### 5.2 New: `rehearse/agents/persona_swap.py`

```python
class PersonaSwapCoordinator:
    """Speak a short bridge utterance whenever the live phase changes."""

    def __init__(
        self,
        session_id: str,
        speak: Callable[[str], Awaitable[None]],
        *,
        bridges: Mapping[Phase, str] | None = None,
    ) -> None: ...

    async def run(self, frames: AsyncIterator[Frame]) -> None: ...
```

Default bridges:

| `to_phase` | Bridge |
|---|---|
| `PRACTICE` | "Okay, let's run it. I'll be {relationship} now." |
| `FEEDBACK` | "Let's pause the scene. Quick reflection." |

Bridges interpolate `Session.persona.relationship` when available; fall
back to "the other person" if not. The coordinator pulls the manifest
on each `PhaseSignal` to resolve `{relationship}`.

The coordinator is wired in `rehearse/telephony.py` as another bus
subscriber inside the existing per-call task group.

### 5.3 Modify `rehearse/personas.py`

Add a third prompt:

```python
FEEDBACK_PROMPT = """You are Rehearse, the coach again, debriefing the user
right after a short rehearsal. Drop any character.

Your one job for the next ~60 seconds:
- Reflect ONE specific thing the user did well, citing their actual words.
- Offer ONE concrete next line they could try if they have to do this for
  real.
- Be warm and direct. No bullet lists. Two short paragraphs max.
"""

def feedback_coach_system_prompt() -> str: ...
```

### 5.4 Modify `rehearse/agents/clm.py`

Extend `_resolve_role` to return `feedback_coach` when phase is
`Phase.FEEDBACK`. Extend `_system_prompt_for_role` to dispatch
`feedback_coach` → `feedback_coach_system_prompt()`. Three valid roles:
`coach`, `character`, `feedback_coach`.

### 5.5 Modify `rehearse/services/hume_configs.py`

For each persona, set:

```python
language_model=HumeLanguageModel(
    provider="CUSTOM_LANGUAGE_MODEL",
    model=None,  # Hume routes to the configured webhook URL
    temperature=None,
),
custom_language_model_url=f"{public_base_url}/chat/completions",  # if Hume
                                                                  # config supports
                                                                  # this field directly
```

If Hume requires CLM URL on the workspace settings rather than per-config,
record that in the rollout note (§7) and update `rehearse-hume` accordingly.

After editing, run `rehearse-hume sync` to push new versions. The runtime
already reads `HUME_CLM_SECRET` for auth.

### 5.6 Modify `rehearse/telephony.py`

Add the coordinator to the per-call task group:

```python
swap = PersonaSwapCoordinator(session_id, speak=hume.say)
swap_task = asyncio.create_task(swap.run(bus.subscribe()))
```

Tear down with the rest of the tasks.

### 5.7 Telemetry

`TelemetryLogger` records `role` (resolved at CLM request time). The CLM
endpoint logs `clm.turn` with `{role, phase, latency_ms, prompt_chars}`.

## 6. Tests

| Test | File | What it asserts |
|---|---|---|
| `_resolve_role` returns `feedback_coach` when phase=FEEDBACK | `tests/test_clm.py` | role mapping covers all three phases |
| `_system_prompt_for_role("feedback_coach")` returns feedback prompt | `tests/test_clm.py` | prompt dispatch is wired |
| `PersonaSwapCoordinator` emits one bridge utterance per `PhaseSignal` | `tests/test_persona_swap.py` (new) | exact text varies with relationship |
| Coordinator does NOT speak on `PhaseSignal(to_phase=INTAKE)` | `tests/test_persona_swap.py` | only practice / feedback get bridges |
| Bridge interpolates `Session.persona.relationship` correctly | `tests/test_persona_swap.py` | "I'll be the CEO now." for relationship="CEO" |
| End-to-end: intake → practice transition flips system prompt mid-stream | `tests/test_clm.py` (extended) | second `/chat/completions` call after a phase write returns character prompt; first returns coach |
| Hume config registry sets `language_model.provider` to the custom-LM provider for both personas | `tests/test_hume_configs.py` | regression on roll-back |
| Bridge utterance lands in transcript as assistant_message tagged with the new phase | `tests/test_telephony_r1.py` (extended, mock Hume) | replayability invariant |

## 7. Rollout

1. **Land code-only changes first** (no remote sync): coordinator,
   feedback prompt, three-way `_resolve_role`, telephony wiring, tests.
2. **Update Hume registry**: switch `language_model` to the custom-LM
   provider in `PERSONAS`. Run `rehearse-hume plan` (dry-run), review
   diffs.
3. **Sync to staging Hume workspace**: `rehearse-hume sync`. New config
   versions are appended; old versions remain available for rollback.
4. **Place a real call** against staging. Verify by replaying
   `transcript.jsonl`: each phase's assistant turns should reflect the
   role-appropriate prompt.
5. **Promote to prod** once one clean staging call is observed.

Rollback: `rehearse-hume sync` again with `language_model.provider`
reverted to `"ANTHROPIC"`; the runtime code path falls back gracefully
because Hume calls Anthropic directly and skips our CLM.

## 8. Open questions

| # | Question | Needed by |
|---|---|---|
| Q1 | Does Hume's per-config `custom_language_model_url` field exist on `PostedLanguageModel`, or must the URL be set workspace-wide? | §5.5 implementation |
| Q2 | Latency budget for our CLM hop: measure round-trip on first staging call. If >500ms, consider Anthropic prompt caching tuning. | §4 NF1 verification |
| Q3 | Should the bridge utterance fire even on `reason="budget"` transitions (where the user did not signal readiness)? Likely yes, but worth a real-call check. | rollout step 4 |
| Q4 | If a phase transitions while the user is mid-utterance, do we wait for end-of-turn before speaking the bridge? Default: yes (use `assistant_message` after next user-turn-end). | rollout step 4 |

## 9. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-06 | Per-turn CLM-driven prompt swap, single voice for the call | Simplest path to correct conversational behavior. Voice swap requires unconfirmed Hume hot-swap support; defer. |
| 2026-05-06 | Bridge utterance is fixed copy, not LLM-generated | Latency and predictability. The bridge is a discourse marker, not content. |
| 2026-05-06 | Three roles, not two | `feedback_coach` is functionally distinct from intake `coach` (debrief vs. elicitation). Same model, different system prompt. |
