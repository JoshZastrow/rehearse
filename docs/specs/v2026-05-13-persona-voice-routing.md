# Persona Voice Routing: Gender Selection + Memory-Backed Agent Dispatch

**Status:** acknowledged
**Date:** 2026-05-13
**Owner:** Josh Zastrow
**Depends on:**
- `v2026-05-06-persona-routing.md` — existing Hume config routing infrastructure
- `v2026-05-12-agent-design-patterns.md` — `AgentRouter`, `CallerMemory`, `IntakeAwareRouter`
- `v2026-05-12-consent-caller-memory.md` — `CallerMemory` protocol

---

## 1. Outcomes

| # | Outcome | Verifiable by |
|---|---|---|
| O1 | A first-time caller is asked "would you prefer a male or female practice partner?" during intake | Transcript presence check |
| O2 | The practice phase uses the character agent matching the caller's answer | Agent name in router dispatch log |
| O3 | A returning caller working on the same topic is NOT asked the gender question again | Absence check in transcript |
| O4 | A returning caller with a new topic IS asked the question | Presence check in transcript |
| O5 | The topic→gender preference is persisted in Honcho and survives across process restarts | Multi-session test with live backend |

---

## 2. Inputs and Outputs

### Inputs to routing

| Input | Source | When available |
|---|---|---|
| `caller_hash` | `Session.phone_number_hash` | Start of call |
| `situation` | `IntakeRecord.situation` (transcript-derived) | End of intake phase |
| `topic_category` | LLM classifier over `situation` | Derived from intake record |
| `gender_preference` | Spoken by caller in response to question, or recalled from memory | During intake or from Honcho |

### Outputs

| Output | Destination | Shape |
|---|---|---|
| `agent_preference` | `CallerMemory` (Honcho peer metadata) | `{topic_category: str, gender: "male" \| "female"}` |
| Character agent selection | `AgentRegistry` lookup via `IntakeAwareRouter` | `MaleCharacterAgent` or `FemaleCharacterAgent` |
| Gender question spoken | Hume EVI TTS (via `send_assistant_input`) | Plain sentence |

---

## 3. Functional Requirements

**FR1 — Topic classifier**
A function `classify_topic(situation: str) -> str` maps a free-text situation
to one of a fixed set of topic categories. Initial set: `"work"`, `"relationship"`,
`"other"`. Implemented as a Claude Haiku call with `max_tokens=10, temperature=0`.
Falls back to `"other"` on any error or null result.

**FR2 — Preference lookup at intake start**
At the start of the intake phase, the intake coach queries memory for the
caller's stored preference for the topic derived from any prior intakes.
If a preference exists for the inferred topic, no gender question is asked.

**FR3 — Gender question during intake**
If no prior preference exists for the current topic, the intake coach speaks:
*"One more thing — would you prefer to practice with a male or female voice?"*
The response is captured by a lightweight classifier (`"male"` / `"female"` /
`"no preference"`). `"no preference"` defaults to `"female"` (current default
Hume voice).

**FR4 — Preference stored on IntakeComplete**
When `IntakeComplete` fires, `IntakeMemoryRecorder` stores both the situation
and the `(topic_category, gender_preference)` pair in the caller's Honcho peer
metadata.

**FR5 — IntakeAwareRouter uses preference**
`IntakeAwareRouter.route()` receives the `IntakeRecord` artifact. It classifies
the topic, queries memory for the preference, and returns `MaleCharacterAgent`
or `FemaleCharacterAgent`. If memory has no preference, it falls back to
`FemaleCharacterAgent`.

**FR6 — Character agent differentiation (Phase 1 — CLM prompts)**
`MaleCharacterAgent` and `FemaleCharacterAgent` differ only in their system
prompt: the character is described as male or female with a matching name. The
Hume EVI TTS voice is unchanged in Phase 1 (same Hume config for all calls).
Actual voice swapping is Phase 2.

**FR7 — Memory clear for test setup**
`CallerMemory` gains `clear_caller(caller_hash: str) -> None` on all
implementations. Used to reset a test caller before an eval run.

---

## 4. Non-Functional Requirements

**NFR1 — Latency**: Topic classification adds at most 400 ms before the first
intake word. Run in the background while the intake coach speaks the opening
line. Do not block on it.

**NFR2 — Graceful degradation**: Any failure in classification, memory read,
or preference parsing falls through to `FemaleCharacterAgent` without
surfacing an error to the caller.

**NFR3 — Test suite unchanged**: All existing tests pass with no Anthropic key
and no Honcho. New tests that require live services are marked
`@pytest.mark.live_api`.

**NFR4 — No PII in Honcho**: Only `topic_category` and `gender` are stored,
never the raw `situation` string. Situation storage (for Honcho Deriver) is
handled separately in `store_session()`.

---

## 5. Out of Scope

| Item | Reason |
|---|---|
| Actual male/female Hume TTS voices (Phase 2) | Requires mid-call config swap (`swap_config` stub); deferred |
| More than 2 genders or "no preference" routing to a distinct agent | Scope; ship binary first |
| Cross-topic preference generalization ("if you chose male for work, assume male for other") | Requires more signal; defer |
| Preference changing mid-call | Not needed for MVP |
| Preference editing by caller | No UI yet |
| SMS-body gender pre-routing (before call connects) | Builds on Phase 2; deferred |

---

## 6. Approach

### Phase 1 (this spec) — CLM-level gender routing

The Hume EVI session uses a single config (no voice change). Gender
differentiation happens at the CLM layer: `IntakeAwareRouter` selects
`MaleCharacterAgent` or `FemaleCharacterAgent`, each with a different system
prompt persona. The practice partner says "I'm Alex, your manager" vs "I'm
Sarah, your manager."

This is testable via transcript analysis without audio inspection.

### Phase 2 (future spec) — Hume config voice swap

When `swap_config` is implemented, pre-connect routing applies the known
gender preference at call start for returning callers. For new callers, a
reconnect with a different Hume config fires at the intake→practice
transition. Phase 2 also adds a real male Hume voice config.

### Topic classification

```python
async def classify_topic(
    situation: str,
    *,
    client: AsyncAnthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> Literal["work", "relationship", "other"]:
    """Classify a situation string into a topic category."""
```

One-shot prompt listing the three categories with examples. Returns `"other"`
on any failure. `max_tokens=10`, `temperature=0`.

### Intake gender question

The intake coach asks the gender question after confirming the situation, before
practice begins. Gated on `memory.get_agent_preference(caller_hash, topic_category) is None`.

Spoken by `send_assistant_input` (bypasses CLM, deterministic):

> "One more thing — would you prefer to practice with a male or female voice?"

Response classified by a rule-based check for "male"/"man"/"he" vs
"female"/"woman"/"she". Ambiguous → `"female"`.

---

## 7. Interface

### `CallerMemory` additions

```python
# rehearse/memory.py

async def get_agent_preference(
    self,
    caller_hash: str,
    topic_category: str,
) -> Literal["male", "female"] | None:
    """Return stored gender preference for a topic, or None if unknown."""
    ...

async def record_agent_preference(
    self,
    caller_hash: str,
    topic_category: str,
    gender: Literal["male", "female"],
) -> None:
    """Persist topic→gender preference for future routing."""
    ...

async def clear_caller(self, caller_hash: str) -> None:
    """Remove all stored data for this caller. Used in test setup."""
    ...
```

Honcho storage: `peer.aio.set_metadata({..., "agent_prefs": {"work": "male", "relationship": "female"}})`.

### New agent classes

```python
# rehearse/agents/roles/character.py

class MaleCharacterAgent:
    name = "male_character"
    _GENDER_PROMPT = "You are playing a male character. Use a male name."

class FemaleCharacterAgent:
    name = "female_character"
    _GENDER_PROMPT = "You are playing a female character. Use a female name."
```

Both extend the existing `CharacterAgent` base; only the system prompt differs.

### `IntakeAwareRouter` (already in spec, now fleshed out)

```python
async def route(self, session: Session, artifact: Any = None) -> RehearseAgent:
    phase = _current_phase(session)
    if phase != Phase.PRACTICE:
        return await self._phase_router.route(session)

    intake = await self._load_intake(session)
    topic = await classify_topic(intake.situation, client=self._llm)
    pref = await self._memory.get_agent_preference(
        session.phone_number_hash or "", topic
    )
    if pref == "male":
        return self._registry.get("male_character")
    return self._registry.get("female_character")  # default
```

---

## 8. Test Fixture

### Eval scenario: three calls, one caller

A synthetic `LLMCustomer` plays the caller. Calls are abbreviated: consent +
intake + 2 practice turns + cancel (no feedback phase). Memory is cleared
before the run via `clear_caller(test_caller_hash)`.

```
call_1:
  topic: "ask my manager for a raise" (work)
  caller is new → intake should ask gender question
  caller answers: "male"
  expected: MaleCharacterAgent selected in practice phase

call_2:
  topic: "ask my manager for a promotion" (work, same category)
  expected: no gender question
  expected: MaleCharacterAgent selected (from memory)

call_3:
  topic: "talk to my partner about moving in together" (relationship)
  caller is new to this topic → intake should ask gender question
  caller answers: "female"
  expected: FemaleCharacterAgent selected in practice phase
```

### Fixture setup

```python
@pytest.fixture
def routing_eval_memory(honcho_server: str) -> HonchoCallerMemory:
    """Fresh memory with a known test caller hash, cleared before the run."""
    memory = HonchoCallerMemory(base_url=honcho_server, workspace_id="rehearse-test")
    return memory

@pytest.fixture
def test_caller() -> str:
    return f"eval-caller-{uuid.uuid4().hex[:8]}"
```

The `honcho_server` fixture (from `conftest.py`) starts a local Honcho
instance. Tests are skipped when `lib/honcho/` is absent.

---

## 9. Eval Judge

**Model**: `claude-haiku-4-5-20251001` (cheapest Anthropic text model)

**Input**: call transcript text

**Output** (structured JSON):

```json
{
  "gender_question_asked": true,
  "agent_selected": "male_character",
  "routing_correct": true,
  "reasoning": "Intake coach asked gender question at turn 3. ..."
}
```

**Judge prompt sketch**:

```
You are evaluating a voice coaching call transcript.

Call context: {call_context}
Expected behavior: {expected}

Transcript:
{transcript}

Answer in JSON with keys:
  gender_question_asked: bool
  agent_selected: "male_character" | "female_character" | "unknown"
  routing_correct: bool
  reasoning: str (one sentence)
```

The judge is invoked once per call. Three calls = three judgments. Pass
condition: all three `routing_correct == true`.

---

## 10. How to Run

```bash
# Requires: make serve with Honcho running, ANTHROPIC_API_KEY set
uv run pytest tests/eval/test_persona_voice_routing_eval.py \
  -v \
  -m "live_api and live_honcho" \
  --timeout=120
```

Each call is capped at 90 seconds by cancelling after the second practice
turn. The full three-scenario suite runs in under 5 minutes.

To run a single scenario:

```bash
uv run pytest tests/eval/test_persona_voice_routing_eval.py::test_new_caller_is_asked_gender -v
```

---

## 11. Artifacts Produced

Each eval run writes to `sessions/<eval_session_id>/`:

| Artifact | Format | Contents |
|---|---|---|
| `transcript.jsonl` | JSONL | Full turn-by-turn transcript |
| `routing_eval_result.json` | JSON | `{call_id, gender_question_asked, agent_selected, routing_correct, reasoning}` per call |
| `memory_state.json` | JSON | Snapshot of `agent_prefs` from Honcho after all three calls |

Summary printed to stdout:

```
PASS  call_1  gender_asked=True  agent=male_character   correct=True
PASS  call_2  gender_asked=False agent=male_character   correct=True
PASS  call_3  gender_asked=True  agent=female_character correct=True

3/3 routing scenarios passed.
```

---

## 12. File Inventory

| File | Change |
|---|---|
| `rehearse/memory.py` | Add `get_agent_preference`, `record_agent_preference`, `clear_caller` to protocol + all 4 implementations |
| `rehearse/agents/topic_classifier.py` | **New** — `classify_topic()` |
| `rehearse/agents/roles/character.py` | Add `MaleCharacterAgent`, `FemaleCharacterAgent` |
| `rehearse/agents/registry.py` | Register both in `build_registry()` |
| `rehearse/agents/router.py` | Flesh out `IntakeAwareRouter` with topic + memory lookup |
| `rehearse/intake.py` | Emit gender question when no preference found; capture answer |
| `rehearse/types.py` | Add `gender_preference: "male" \| "female" \| None` to `IntakeRecord` |
| `tests/eval/test_persona_voice_routing_eval.py` | **New** — 3-scenario eval |
| `tests/test_topic_classifier.py` | **New** — unit tests for classifier |
| `tests/test_intake_gender_question.py` | **New** — unit tests for question gating logic |

---

## 13. Migration

1. Run `uv run rehearse-hume sync` after registering the new male/female character
   Hume configs (Phase 2 only; Phase 1 requires no new configs).
2. Existing callers have no `agent_prefs` in Honcho metadata → falls through
   to `FemaleCharacterAgent` as default. No disruption to live callers.
3. `clear_caller` is only called in test fixtures; it is a no-op in `NullCallerMemory`.

---

## 14. Open Questions

| # | Question | Impact |
|---|---|---|
| Q1 | Should the gender question be asked once globally (any topic) or per topic category? | If global: simpler memory, but caller may want different voice for work vs relationship. Per-topic is spec'd above. |
| Q2 | What's the right fallback voice gender when memory is empty? Female (current Hume voice) or prompt user always? | Spec above defaults to female. |
| Q3 | For Phase 2, does config swap happen at call start (pre-routed for returning callers) or mid-call (reconnect on intake complete)? | Architectural — defer to Phase 2 spec. |
