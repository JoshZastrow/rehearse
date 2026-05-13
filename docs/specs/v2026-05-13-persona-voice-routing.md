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
| O1 | A first-time caller is asked "would you prefer a male or female practice partner?" during intake | LLM judge on transcript |
| O2 | The practice phase uses the voice matching the caller's answer | Audio judge: `VLLMAudioProvider` (Gemma) |
| O3 | A returning caller is NOT asked the gender question again | Absence check in transcript |
| O4 | The gender preference persists in Honcho across process restarts | Multi-session live test |
| O5 | Voice change produces a perceptibly different voice to the audio judge | Gemma audio classification score |

---

## 2. Key Decisions

**Gender preference scope: global.**
Phase 1 stores one preference per caller — not per topic. Once a caller has expressed a preference, we use it for all future calls. Topic-scoped preferences are deferred.

**Voice swap mechanism: `session_settings` mid-call.**
Hume EVI supports a `session_settings` WebSocket message that can update `voice_id` and `system_prompt` during an active chat without disconnecting or losing conversation context. This eliminates the need for separate male/female Hume EVI configs and mid-call reconnects. The `PersonaSwapCoordinator` sends the settings update alongside the existing bridge utterance at the intake→practice transition.

**Eval judge: `VLLMAudioProvider` (Gemma via vLLM).**
The existing vLLM provider (`rehearse/eval/providers/vllm.py`, default `gemma-4-e4b`) handles both text-based routing checks and audio-based voice verification in one model. Marginal cost per eval run is near-zero once the inference server is running.

**Hold music: future enhancement.**
A music generation model (MusicGen/AudioCraft) that generates a short atmospheric clip from the intake situation is desirable UX but deferred. The `PersonaSwapCoordinator` bridge line ("just a moment while I connect you...") covers the transition for now. When the music feature lands, it kicks off generation at `IntakeComplete` and streams the clip to the caller during the `session_settings` update window.

---

## 3. Inputs and Outputs

### Inputs to routing

| Input | Source | When available |
|---|---|---|
| `caller_hash` | `Session.phone_number_hash` | Call start |
| `gender_preference` | Spoken by caller during intake OR recalled from Honcho | During intake |

### Outputs

| Output | Destination | Shape |
|---|---|---|
| `gender_preference` | Honcho peer metadata | `{"gender": "male" \| "female"}` |
| `session_settings` message | Hume EVI WebSocket | `{voice_id, system_prompt}` |
| Bridge utterance | Hume TTS via `send_assistant_input` | "Just a moment while I connect you..." |

---

## 4. Functional Requirements

**FR1 — Gender question gating**
At the start of the intake phase, `ConsentGate._speak_intake_context()` (or a new `IntakeCoach` hook) queries `memory.get_gender_preference(caller_hash)`. If a preference exists, skip the question and route directly. If not, the intake coach speaks the question after capturing the situation:

> *"One more thing — would you prefer to practice with a male or female voice?"*

**FR2 — Response classification**
The caller's answer is classified by a rule-based matcher (`"male"` / `"man"` / `"he"` → `"male"`; `"female"` / `"woman"` / `"she"` → `"female"`). Ambiguous or no response defaults to `"female"`. The intake coach confirms: *"Got it — let me set that up for you."*

**FR3 — Preference stored**
On answer (or at `IntakeComplete` if captured in the IntakeRecord), `memory.record_gender_preference(caller_hash, gender)` is called. Stored in Honcho peer metadata under key `"gender"`. Overwritable: a caller can change their preference by asking explicitly.

**FR4 — `session_settings` voice swap at practice start**
`PersonaSwapCoordinator.on_intake_to_practice()` is extended to:
1. Speak the bridge utterance via `send_assistant_input`.
2. Send a `session_settings` WebSocket message while TTS plays:
   ```json
   {
     "type": "session_settings",
     "voice_id": "<male_or_female_hume_voice_id>",
     "system_prompt": "<gender-appropriate character prompt>"
   }
   ```
3. The practice phase begins under the new voice. Conversation context is preserved.

**FR5 — Voice ID registry**
Two voice names are added to the Hume config:

```python
VOICE_REGISTRY = {
    "female": "Inspiring Woman",   # existing default
    "male": "Wise Man",            # new — must exist in the Hume workspace
}
```

The voice name is resolved to a `voice_id` at swap time by looking up the name in the Hume voices API (same pattern as `select_config_id`). `REHEARSE_MALE_VOICE_NAME` env var overrides the default for local testing.

**FR6 — Character system prompt differentiation**
`MaleCharacterAgent` and `FemaleCharacterAgent` produce different system prompts:

```python
# Female (default): "You are playing the other person in this conversation. ..."
# Male override: "You are playing a male character. Use a male name and male pronouns. ..."
```

Both are sent as the `system_prompt` override in the `session_settings` message.

**FR7 — Memory clear for test setup**
`CallerMemory` gains `clear_caller(caller_hash: str) -> None`. Used in eval fixture setup to reset a test caller before each run.

---

## 5. Non-Functional Requirements

**NFR1 — Latency**: Gender question and memory lookup must not extend the intake phase by more than 500 ms total. Memory lookup runs in background while intake coach speaks.

**NFR2 — Graceful degradation**: Any failure in memory read, voice swap, or preference classification falls through to the female voice (existing default behavior). No error surfaced to the caller.

**NFR3 — Test suite unchanged**: All existing tests pass with no Honcho and no vLLM server. New live tests are marked `@pytest.mark.live_api`.

**NFR4 — No PII stored**: Only `"male"` / `"female"` is stored in Honcho, never the raw utterance.

---

## 6. Out of Scope

| Item | Reason |
|---|---|
| Per-topic gender preferences | Deferred — global preference ships first |
| Mid-call preference change after practice starts | Voice swap only fires once at intake→practice transition |
| Music generation during transfer | Deferred to follow-up spec; noted in §2 |
| Non-binary / custom voice options | Scope; extend registry when needed |
| SMS-body gender pre-routing | Builds on this; deferred |
| Separate Hume EVI configs per gender | Not needed — `session_settings` handles in-session voice swap |

---

## 7. Interface

### `CallerMemory` additions

```python
# rehearse/memory.py

async def get_gender_preference(
    self, caller_hash: str
) -> Literal["male", "female"] | None:
    """Return stored voice gender preference, or None if not yet set."""
    ...

async def record_gender_preference(
    self,
    caller_hash: str,
    gender: Literal["male", "female"],
) -> None:
    """Persist the caller's voice gender preference."""
    ...

async def clear_caller(self, caller_hash: str) -> None:
    """Remove all stored data for this caller. Used in eval setup only."""
    ...
```

Honcho storage: `peer.aio.set_metadata({..., "gender": "male"})`.

### `PersonaSwapCoordinator` extension

```python
async def on_intake_to_practice(
    self,
    session: Session,
    gender: Literal["male", "female"],
) -> None:
    # 1. Speak bridge line
    await self._speaker.say(SpeakRequest(
        text="Just a moment while I connect you with your practice partner."
    ))
    # 2. Resolve voice_id from gender
    voice_id = await resolve_voice_id(gender)  # Hume voices API lookup
    character_prompt = _character_prompt(gender, session.persona)
    # 3. Send session_settings while bridge TTS plays
    await self._hume_client.send_session_settings(
        voice_id=voice_id,
        system_prompt=character_prompt,
    )
```

### `HumeEVIClient.send_session_settings`

New method on `HumeEVIClient`:

```python
async def send_session_settings(
    self,
    *,
    voice_id: str | None = None,
    system_prompt: str | None = None,
) -> None:
    """Send a session_settings message to update voice or prompt mid-call."""
    payload: dict[str, Any] = {"type": "session_settings"}
    if voice_id:
        payload["voice_id"] = voice_id
    if system_prompt:
        payload["system_prompt"] = system_prompt
    await self._socket.send(json.dumps(payload))
```

---

## 8. Eval Judge: `VLLMAudioProvider` (Gemma)

The existing `VLLMAudioProvider` (`rehearse/eval/providers/vllm.py`, default model `gemma-4-e4b`) is used for all eval judgments. It accepts audio directly — no transcription step needed for voice verification.

### Judgment 1 — Routing check (text)

Input: transcript text from intake + first two practice turns.

Prompt:
```
You are evaluating a voice coaching call transcript.

Expected behavior: {expected_behavior}

Transcript:
{transcript}

Answer in JSON:
{
  "gender_question_asked": true | false,
  "routing_correct": true | false,
  "reasoning": "<one sentence>"
}
```

### Judgment 2 — Voice verification (audio)

Input: audio clip of the character's first 10 seconds of speech in the practice phase.

Prompt:
```
Listen to this audio clip. Is the speaker's voice male or female?
Answer with exactly one word: "male" or "female".
```

Expected output matches the routing decision. This is the ground-truth check that the `session_settings` voice swap actually took effect and produced an audibly different voice.

### Pass criteria

| Judgment | Pass condition |
|---|---|
| Routing check | `routing_correct == true` |
| Voice verification | Gemma answer matches `expected_gender` |
| Memory check | Call 2 `gender_question_asked == false` |

All three must pass for the scenario to pass.

---

## 9. Test Fixture

### Scenarios

Three abbreviated calls (consent + intake + 2 practice turns + cancel). A synthetic `LLMCustomer` plays the caller. Memory is cleared via `clear_caller(test_caller_hash)` before the run.

```
call_1:  new caller
  input:  "I need to ask my manager for a raise"
  caller says: "male"
  expected routing: MaleCharacterAgent
  expected voice judgment: "male"

call_2:  same caller, any topic
  expected: no gender question asked
  expected routing: MaleCharacterAgent (from memory)
  expected voice judgment: "male"

call_3:  same caller, forces preference change
  caller explicitly says: "actually I'd prefer a female voice"
  expected routing: FemaleCharacterAgent
  expected voice judgment: "female"
  expected memory after: "female"
```

### Fixtures

```python
@pytest.fixture
def routing_eval_memory(honcho_server: str) -> HonchoCallerMemory:
    memory = HonchoCallerMemory(base_url=honcho_server, workspace_id="rehearse-test")
    return memory

@pytest.fixture
def test_caller() -> str:
    return f"eval-caller-{uuid.uuid4().hex[:8]}"

@pytest.fixture
def vllm_judge() -> VLLMAudioProvider:
    return VLLMAudioProvider()  # uses VLLM_BASE_URL + VLLM_API_KEY from env
```

The `honcho_server` fixture (from `conftest.py`) starts a local Honcho instance. Tests skip when `lib/honcho/` is absent or `VLLM_BASE_URL` is not set.

---

## 10. How to Run

```bash
# Requires: VLLM_BASE_URL, VLLM_API_KEY, Honcho running (make serve)
uv run pytest tests/eval/test_persona_voice_routing_eval.py \
  -v \
  -m "live_api and live_honcho" \
  --timeout=120
```

Each call is capped at 90 seconds (cancel after second practice turn). Full three-scenario suite runs in under 6 minutes.

Single scenario:
```bash
uv run pytest tests/eval/test_persona_voice_routing_eval.py::test_new_caller_is_asked_gender -v
```

Clear a specific caller's memory for manual re-runs:
```bash
uv run python -c "
import asyncio
from rehearse.memory import HonchoCallerMemory
m = HonchoCallerMemory(base_url='http://localhost:8001')
asyncio.run(m.clear_caller('your-test-hash'))
"
```

---

## 11. Artifacts Produced

Each eval run writes to `sessions/<eval_session_id>/`:

| Artifact | Format | Contents |
|---|---|---|
| `transcript.jsonl` | JSONL | Full turn-by-turn transcript |
| `routing_eval_result.json` | JSON | `{call_id, gender_question_asked, routing_correct, reasoning}` |
| `voice_eval_result.json` | JSON | `{call_id, expected_gender, gemma_judgment, pass}` |
| `character_audio_clip.wav` | WAV | First 10s of character speech in practice phase (input to Gemma) |
| `memory_state.json` | JSON | Snapshot of `gender` key from Honcho after all three calls |

Summary printed to stdout:

```
PASS  call_1  gender_asked=True   routing=male_character  voice_judgment=male   ✓
PASS  call_2  gender_asked=False  routing=male_character  voice_judgment=male   ✓
PASS  call_3  gender_asked=False  routing=female_character voice_judgment=female ✓

3/3 routing scenarios passed. 3/3 voice verifications passed.
```

---

## 12. File Inventory

| File | Change |
|---|---|
| `rehearse/memory.py` | Add `get_gender_preference`, `record_gender_preference`, `clear_caller` to protocol + all 4 implementations |
| `rehearse/agents/roles/character.py` | Add `MaleCharacterAgent`, `FemaleCharacterAgent`; extract `_character_prompt(gender, persona)` helper |
| `rehearse/agents/registry.py` | Register both in `build_registry()` |
| `rehearse/agents/router.py` | Extend `IntakeAwareRouter` to read gender preference from memory |
| `rehearse/agents/persona_swap.py` | Add `on_intake_to_practice(session, gender)`; send `session_settings` after bridge line |
| `rehearse/services/hume_evi.py` | Add `send_session_settings(voice_id, system_prompt)` |
| `rehearse/services/hume_configs.py` | Add `VOICE_REGISTRY = {"female": "Inspiring Woman", "male": "Wise Man"}` + `resolve_voice_id(gender)` |
| `rehearse/intake.py` | Ask gender question when no preference found; capture answer into `IntakeRecord.gender_preference` |
| `rehearse/types.py` | Add `gender_preference: Literal["male", "female"] \| None = None` to `IntakeRecord` |
| `rehearse/config.py` | Add `rehearse_male_voice_name: str = "Wise Man"` |
| `tests/eval/test_persona_voice_routing_eval.py` | **New** — 3-scenario routing + voice eval |
| `tests/test_gender_memory.py` | **New** — unit tests for `get/record/clear_gender_preference` |
| `tests/test_session_settings_swap.py` | **New** — unit test: `send_session_settings` sends correct WebSocket payload |

---

## 13. Open Questions

| # | Question | Impact |
|---|---|---|
| Q1 | What is the exact Hume voice name for the male voice? ("Wise Man" is a guess — verify in Hume dashboard) | `VOICE_REGISTRY` value; no code change needed if env var is used |
| Q2 | Should the gender question be asked before or after the situation is captured? Before is simpler; after lets the coach say "and for that conversation, male or female?" | Intake script ordering |
| Q3 | If the `session_settings` message arrives while TTS is mid-stream, does Hume apply it immediately or queue it? | Timing of the swap; test empirically |
