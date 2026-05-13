# Persona Voice Routing: Gender Selection + Memory-Backed Agent Dispatch

**Status:** wip
**Date:** 2026-05-13
**Owner:** Josh Zastrow
**Depends on:**
- `v2026-05-06-persona-routing.md` — existing Hume config routing infrastructure
- `v2026-05-12-agent-design-patterns.md` — `AgentRouter`, `CallerMemory`, `IntakeAwareRouter`

---

## 1. Vision

Rehearse should feel like calling a real person who was trained to play a specific role.

A caller practicing a conversation with their angry boss shouldn't get a generic "difficult character" — they should get a persona trained on their boss's actual complaints, speech patterns, and dismissal tactics. The persona is a combination of a Hume voice, a CLM (fine-tuned model or rich system prompt), and a dataset that shaped how the character behaves.

This spec builds the **Persona Registry** — the infrastructure for storing, routing to, and eventually training these characters. Phase 1 ships seed personas and routing. The fine-tuning pipeline is a future dependency.

---

## 2. Outcomes

| # | Outcome | Verifiable by |
|---|---|---|
| O1 | A first-time caller is asked "would you prefer a male or female practice partner?" during intake | LLM judge on transcript |
| O2 | The practice phase uses the persona matching the caller's answer | Audio judge: `VLLMAudioProvider` (Gemma) |
| O3 | A returning caller is NOT asked the gender question again | Absence check in transcript |
| O4 | The gender preference persists in Honcho across process restarts | Multi-session live test |
| O5 | Voice change produces a perceptibly different voice to the audio judge | Gemma audio classification |
| O6 | `PersonaRoutingAgent` selects the correct persona from the registry via tool call | Unit test: mock tool response |

---

## 3. Key Decisions

**Persona Registry, not a voice list.**
Routing picks from a registry of structured persona records — each with a voice, a CLM reference, a description, and tags. The routing agent calls `list_personas()` as a tool and reads the registry. Adding a new persona requires no code change, only a new registry entry.

**Gender preference scope: global.**
One preference per caller, not per topic. Stored in Honcho peer metadata. Overwritable by the caller at any time.

**Voice swap mechanism: `session_settings` mid-call.**
Hume EVI supports updating `voice_id` and `system_prompt` via a `session_settings` WebSocket message during an active chat without disconnecting or losing context. No separate Hume configs needed per persona.

**Routing agent: lightweight + tool call.**
A small agent (Claude Haiku or Gemma via vLLM) runs once at `IntakeComplete`. It reads the intake transcript and calls `list_personas()`. It returns the best-matching persona record. Completes within the bridge utterance window (~2s).

**Eval judge: `VLLMAudioProvider` (Gemma via vLLM).**
Existing provider (`rehearse/eval/providers/vllm.py`, model `gemma-4-e4b`). Handles both text routing checks and audio-based voice verification. Zero marginal cost per run.

**Fine-tuning pipeline: future dependency.**
Phase 1 ships prompt-based personas. Phase 2 adds a pipeline: ingest caller-provided data → assemble dataset → fine-tune model → register persona with `clm_endpoint`. The registry schema supports this today.

---

## 4. Persona Registry

### `PersonaRecord`

```python
# rehearse/personas/registry.py

@dataclass
class PersonaRecord:
    id: str                          # stable slug, e.g. "male_boss_direct"
    name: str                        # display name, e.g. "Direct Male Manager"
    description: str                 # one sentence for routing agent context
    gender: Literal["male", "female"]
    voice_name: str                  # Hume voice name, e.g. "Wise Man"
    voice_id: str | None = None      # resolved lazily from Hume voices API
    system_prompt_template: str = "" # character prompt; {relationship} interpolated
    clm_endpoint: str | None = None  # fine-tuned model URL (Phase 2)
    tags: list[str] = field(default_factory=list)  # ["work", "confrontational"]
```

### Seed registry

```python
PERSONA_REGISTRY: list[PersonaRecord] = [
    PersonaRecord(
        id="female_coach_default",
        name="Female Practice Partner",
        description="Warm but direct female counterparty for general rehearsal.",
        gender="female",
        voice_name="Inspiring Woman",
        tags=["default", "female"],
    ),
    PersonaRecord(
        id="male_coach_default",
        name="Male Practice Partner",
        description="Calm but direct male counterparty for general rehearsal.",
        gender="male",
        voice_name="Wise Man",
        tags=["default", "male"],
    ),
]
```

New personas are added as `PersonaRecord` entries — no code change required. When a fine-tuned model is ready, set `clm_endpoint` and the routing agent will prefer it automatically.

### `PersonaRegistry` class

```python
class PersonaRegistry:
    def __init__(self, records: list[PersonaRecord]) -> None: ...

    def list(self, *, gender: str | None = None, tags: list[str] | None = None) -> list[PersonaRecord]:
        """Return personas filtered by gender and/or tags."""

    def get(self, persona_id: str) -> PersonaRecord | None: ...

    def to_tool_response(self) -> list[dict]:
        """Serialize for the routing agent's tool call response."""
```

---

## 5. PersonaRoutingAgent

A single-turn agent that runs at `IntakeComplete`. It has one tool.

```python
# rehearse/agents/persona_routing_agent.py

class PersonaRoutingAgent:
    """Lightweight agent that picks a persona from the registry given an intake transcript."""

    def __init__(
        self,
        registry: PersonaRegistry,
        *,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None: ...

    async def select(
        self,
        transcript: str,
        gender_hint: Literal["male", "female"] | None,
    ) -> PersonaRecord:
        """
        Call the LLM with the transcript and a list_personas tool.
        The LLM calls list_personas(), reads the results, and returns
        the best-matching persona_id. Falls back to the default persona
        for the given gender on any error.
        """
```

### Tool definition

```python
{
    "name": "list_personas",
    "description": "Return available practice partner personas. Filter by gender or tags.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gender": {"type": "string", "enum": ["male", "female"]},
            "tags": {"type": "array", "items": {"type": "string"}}
        }
    }
}
```

The tool handler returns `registry.to_tool_response()` — a list of persona dicts with `id`, `name`, `description`, `tags`. The LLM picks one and returns `{"persona_id": "male_boss_direct"}`.

**Latency target**: ≤ 400ms. `max_tokens=30`, `temperature=0`. Runs during the bridge utterance.

---

## 6. Functional Requirements

**FR1 — Gender question gating.**
At `IntakeComplete`, check `memory.get_gender_preference(caller_hash)`. If found, skip question and route using stored preference. If not found, intake coach asks during the intake conversation (after situation is captured):

> *"One more thing — would you prefer to practice with a male or female voice?"*

**FR2 — Response classification.**
Rule-based: "male"/"man"/"he" → `"male"`; "female"/"woman"/"she" → `"female"`. Ambiguous defaults to `"female"`.

**FR3 — Preference stored.**
`memory.record_gender_preference(caller_hash, gender)` called after the answer is captured. Honcho metadata key `"gender"`.

**FR4 — Persona selected by routing agent.**
`PersonaRoutingAgent.select(transcript, gender_hint)` runs at `IntakeComplete`. Returns a `PersonaRecord`. The gender hint constrains the tool call so the agent only sees matching-gender personas.

**FR5 — Voice swap via `session_settings`.**
`PersonaSwapCoordinator.on_intake_to_practice()`:
1. Speaks: *"Just a moment while I connect you with your practice partner."*
2. Resolves `voice_id` from `persona.voice_name` via Hume voices API (cached).
3. Sends `session_settings` with `voice_id` + `system_prompt`.

**FR6 — `send_session_settings` on `HumeEVIClient`.**

```python
async def send_session_settings(
    self,
    *,
    voice_id: str | None = None,
    system_prompt: str | None = None,
) -> None:
    payload = {"type": "session_settings"}
    if voice_id:
        payload["voice_id"] = voice_id
    if system_prompt:
        payload["system_prompt"] = system_prompt
    await self._socket.send(json.dumps(payload))
```

**FR7 — Voice ID resolution.**
`resolve_voice_id(voice_name: str) -> str` calls `hume.empathic_voice.voices.list()`, finds the matching entry, caches the result in memory for the process lifetime. Cache invalidated on restart.

**FR8 — Memory clear for test setup.**
`CallerMemory.clear_caller(caller_hash)` removes all stored data. Used only in eval fixtures.

---

## 7. Non-Functional Requirements

**NFR1** — Persona routing adds ≤ 400ms before bridge line ends.
**NFR2** — Any failure falls back to default female persona silently.
**NFR3** — All existing tests pass with no Honcho, no vLLM, no Anthropic key.
**NFR4** — Only `"male"` / `"female"` stored in Honcho, no raw utterances.

---

## 8. Out of Scope

| Item | Reason |
|---|---|
| Fine-tuning pipeline | Future spec; schema ready today |
| Per-topic gender preferences | Deferred; global preference ships first |
| Custom Hume voice training | Requires Hume custom voice feature |
| Mid-call persona change after practice starts | Swap fires once at transition |
| Music generation during transfer | Deferred to follow-up spec |
| More than two gender options | Extend registry when needed |

---

## 9. Eval Judge: `VLLMAudioProvider` (Gemma)

### Judgment 1 — Routing check (text)

Uses `VLLMAudioProvider` in text mode. Input: transcript of intake + first two practice turns.

```python
prompt = f"""
You are evaluating a voice coaching call transcript.
Expected: {expected_behavior}
Transcript: {transcript}

Answer in JSON:
{{"gender_question_asked": true|false, "routing_correct": true|false, "reasoning": "..."}}
"""
```

### Judgment 2 — Voice verification (audio)

Input: WAV clip of the character's first 10 seconds in the practice phase.

```python
prompt = "Is the speaker in this audio clip male or female? Answer with one word: male or female."
```

Expected output matches the routing decision. This verifies that `session_settings` actually changed the voice.

---

## 10. Test Scenarios

Three abbreviated calls (consent + intake + 2 practice turns + cancel). Memory cleared before run.

```
call_1  new caller
        situation: "I need to ask my manager for a raise"
        caller says: "male"
        → routing agent selects male_coach_default
        → session_settings: voice_id=wise_man, male system prompt
        expected voice judgment: "male"

call_2  same caller
        → no gender question (from memory)
        → same persona
        expected voice judgment: "male"

call_3  same caller, preference change
        caller says: "actually female"
        → routing agent selects female_coach_default
        expected voice judgment: "female"
        expected memory after: "female"
```

---

## 11. How to Run

```bash
uv run pytest tests/eval/test_persona_voice_routing_eval.py \
  -v -m "live_api and live_honcho" --timeout=120
```

Clear a test caller:
```bash
uv run python -c "
import asyncio
from rehearse.memory import HonchoCallerMemory
m = HonchoCallerMemory(base_url='http://localhost:8001')
asyncio.run(m.clear_caller('your-test-hash'))
"
```

---

## 12. Artifacts Produced

| Artifact | Format | Contents |
|---|---|---|
| `transcript.jsonl` | JSONL | Full transcript |
| `routing_eval_result.json` | JSON | Per-call: `{gender_question_asked, persona_selected, routing_correct, reasoning}` |
| `voice_eval_result.json` | JSON | Per-call: `{expected_gender, gemma_judgment, pass}` |
| `character_audio_clip.wav` | WAV | First 10s of character speech (Gemma input) |
| `memory_state.json` | JSON | Honcho `gender` key after all calls |

---

## 13. File Inventory

| File | Change |
|---|---|
| `rehearse/personas/registry.py` | **New** — `PersonaRecord`, `PersonaRegistry`, `PERSONA_REGISTRY` |
| `rehearse/personas/__init__.py` | **New** — package init |
| `rehearse/agents/persona_routing_agent.py` | **New** — `PersonaRoutingAgent` with `list_personas` tool |
| `rehearse/memory.py` | Add `get_gender_preference`, `record_gender_preference`, `clear_caller` |
| `rehearse/agents/roles/character.py` | Add `MaleCharacterAgent`, `FemaleCharacterAgent` |
| `rehearse/agents/registry.py` | Register both character agents |
| `rehearse/agents/router.py` | `IntakeAwareRouter` reads gender preference + invokes routing agent |
| `rehearse/agents/persona_swap.py` | Extend: call routing agent, send `session_settings` at PRACTICE transition |
| `rehearse/services/hume_evi.py` | Add `send_session_settings(voice_id, system_prompt)` |
| `rehearse/services/hume_configs.py` | Add `resolve_voice_id(voice_name)` with in-process cache |
| `rehearse/intake.py` | Ask gender question when no preference found; store in `IntakeRecord` |
| `rehearse/types.py` | Add `gender_preference: Literal["male","female"] \| None` to `IntakeRecord` |
| `rehearse/config.py` | Add `rehearse_male_voice_name: str = "Wise Man"` |
| `tests/test_persona_registry.py` | **New** — registry lookup, tag filtering, tool serialization |
| `tests/test_persona_routing_agent.py` | **New** — tool call mocked, persona selection |
| `tests/test_gender_memory.py` | **New** — get/record/clear gender preference |
| `tests/test_session_settings.py` | **New** — `send_session_settings` WebSocket payload |
| `tests/eval/test_persona_voice_routing_eval.py` | **New** — 3-scenario eval |

---

## 14. Open Questions

| # | Question |
|---|---|
| Q1 | Is "Wise Man" the correct Hume voice name for the male voice? Verify in Hume dashboard before wiring. |
| Q2 | Does `session_settings` apply before or after the current TTS utterance finishes? Test empirically — affects bridge line timing. |
| Q3 | Should `PersonaRoutingAgent` use the same Anthropic client as the CLM, or a separate instance with a fixed Haiku model? |
