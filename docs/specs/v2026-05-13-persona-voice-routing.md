# Persona Voice Routing: Memory-Backed Practice Partner Selection

**Status:** wip
**Date:** 2026-05-13
**Owner:** Josh Zastrow
**Depends on:** `v2026-05-12-agent-design-patterns.md`

---

## 1. Problem

Every call to Rehearse assigns the same generic practice partner: a neutral voice
with no history and no memory of who the caller has practiced with before.

When a caller says *"I need to call my dad and tell him I crashed his car — he's
going to be pretty mad,"* the system should recognize that the practice partner is
a specific person — a father — not a menu option. It should not ask the caller to
choose a gender. It should not start from zero on the second call.

The problem is threefold:

1. **Counterparty inference is rule-based.** `classify_gender_preference()` matches
   keywords. It fails for "my dad", "Sarah my assistant", or anything outside a
   hardcoded list.

2. **Personas are static code.** `PersonaRegistry` is a Python list. Adding a new
   persona requires a code change and a deploy. It cannot grow to N personas.

3. **Nothing is remembered.** Each call bootstraps a fresh character with no
   knowledge of who this caller practiced with last time, or how that character
   behaved, or what the caller learned.

---

## 2. Outcomes

| # | Outcome | Acceptance test |
|---|---|---|
| O1 | Counterparty gender and name inferred from transcript without keyword matching | LLM extraction returns correct gender for "my dad", "Sarah my assistant", "my manager Tom" |
| O2 | Gender question only asked when the transcript has no signal | Intake transcript with clear counterparty → no question asked; ambiguous transcript → question asked |
| O3 | First-encounter counterparty bootstraps a persona and soul from description | `PeerStore.bootstrap()` called on first encounter; `PersonaRecord` + soul stored in Honcho |
| O4 | Second encounter retrieves the same persona from Honcho memory | `PeerStore.search()` finds prior persona via Dialectic; same voice and soul used |
| O5 | Practice partner has a Honcho Peer with observations accumulating across sessions | Character peer exists in Honcho; Deriver produces observations after session 3+ |
| O6 | Soul document at session start = bootstrap soul + Dialectic synthesis | `PeerStore.get_soul()` returns merged text; Layer 2 grows with sessions |
| O7 | Audio gender eval judge verifies voice actually changed | Gemma 3 via Modal judges WAV clip; `voice_eval_result.json` produced per eval run |
| O8 | Eval infra costs ~$0 when idle | Modal serverless GPU scales to zero between eval runs |

---

## 3. Architectural decisions

### 3.1 Inference model split: Haiku for real-time, Gemma for eval

Two distinct inference workloads with incompatible latency requirements:

| Workload | Model | Infra | Why |
|---|---|---|---|
| Counterparty extraction (call-time) | Claude Haiku 4.5 | Anthropic API | <200ms, no GPU, $0.00025/1K tok |
| Soul bootstrapping (post-intake, async) | Claude Haiku 4.5 | Anthropic API | Creative quality sufficient, same latency |
| Honcho Dialectic synthesis | Honcho internal | Honcho infrastructure | Already handled |
| Audio gender eval judge | Gemma 3 27B | Modal serverless GPU | Audio-native, cheap for batch, OK cold start |
| Soul quality scoring (eval) | Gemma 3 27B | Modal serverless GPU | Text eval, same infra |

**Real-time path uses Haiku API exclusively.** No GPU required on the critical
call path. Latency budget for counterparty extraction: ≤ 400ms (runs during the
bridge utterance). Haiku averages 180ms at this token count.

**Eval path uses Gemma 3 27B on Modal.** Modal serverless GPU provides:
- Cold start: ~15-30s on first invocation in a session (acceptable for evals)
- Warm start: ~150ms per inference
- Cost: ~$0.001/GPU-second on A100; ~$0.006 per eval call at 6s inference
- Scale to zero between runs — zero cost when idle
- The existing `VLLMAudioProvider` points at `VLLM_BASE_URL`; Modal provides this
  endpoint. Zero code change to the provider.

**Why not always-on GPU:**

A reserved A100 runs ~$2,800/month. An eval suite that runs 3x/day at 6 minutes
each costs ~$1.08/day on Modal — about $32/month. The break-even is ~87x/day.
We are nowhere near that threshold in development.

**Why not use Haiku for audio eval:**

Haiku does not accept audio input. The gender verification check requires a model
that can classify voice from a WAV clip. Gemma 3 via vLLM on Modal is the only
in-house option that satisfies both requirements.

### 3.2 PeerStore, not PersonaRegistry

The entity being managed is a **Honcho Peer** — a practice partner who persists
across sessions and accumulates a representation over time. The class that manages
these peers is `PeerStore`. Naming it `PersonaStore` obscures the underlying
model: each practice partner IS a Honcho Peer, and the store creates, finds, and
reads Honcho Peers.

`PersonaRegistry` (the static Python list) is deleted. It cannot represent N
characters and offers no path to memory-backed evolution.

---

## 4. Data contracts

### 4.1 Honcho peer model

| Peer | ID format | `observe_me` | `observe_others` | Purpose |
|---|---|---|---|---|
| Caller | `{caller_hash}` | `True` | `False` | Deriver builds caller representation |
| Coach | `rehearse_coach` | `False` | `True` | Deterministic; not modeled |
| Character | `character_{caller_hash[:8]}_{persona_slug}` | `True` | `True` | Deriver builds character representation; character learns caller patterns |

Character peers are created at bootstrap and reused across all sessions for that
caller-character pair. A caller who practices with "dad" twice has one character
peer whose representation deepens after each session.

### 4.2 Caller peer metadata schema

All persona records are stored as structured metadata on the caller's Honcho peer:

```json
{
  "consented": true,
  "gender": "male",
  "personas": {
    "male_callers_father": {
      "id": "male_callers_father",
      "name": "David",
      "description": "caller's father, upset about crashed car",
      "gender": "male",
      "voice_name": "Wise Man",
      "soul": "# David — Soul\n\n## Identity\n...",
      "sessions": 3,
      "created_at": "2026-05-13T14:00:00Z",
      "soul_refreshed_at": "2026-05-13T14:00:00Z"
    }
  }
}
```

`soul` is the bootstrap soul (Layer 1). It is refreshed every 5 sessions by
re-running the soul generator with accumulated Dialectic insights as input.

### 4.3 Session message attribution

Practice phase messages must be attributed to the character peer — not the coach
peer — so the Deriver builds the character's representation from their words.

`store_session()` receives the transcript as `[{"role", "content", "speaker"}]`.
Split by `speaker` tag:

| `speaker` value | Attributed to |
|---|---|
| `"user"` | `caller_peer` |
| `"coach"` | `coach_peer` |
| `"character"` | `character_peer` |
| anything else | `coach_peer` |

### 4.4 Soul document structure

Six sections. Stored as plain Markdown. Generated once by Haiku from the base
template (`rehearse/personas/souls/base_character.md`) and the caller's
description. Refreshed every 5 sessions using Dialectic synthesis as additional
input.

```
# [Name] — Soul

## Identity        — essential nature; what shaped them
## Values          — what they'd never compromise
## How they communicate  — rhythm, tells, silences
## Emotional landscape   — triggers, how they soften/harden
## Relationship to the caller  — what they want for this relationship
## Edges           — where the character ends; what breaks the frame
```

---

## 5. Component interfaces

### 5.1 `PeerStore`

```python
# rehearse/personas/peer_store.py

class PeerStore:
    """Manages practice partner Honcho Peers: creation, retrieval, soul."""

    def __init__(self, honcho: Honcho, llm_client: AsyncAnthropic) -> None: ...

    async def search(
        self,
        caller_hash: str,
        description: str,
    ) -> PersonaRecord | None:
        """Find a prior persona matching this description via Dialectic.

        Query: "Has this caller practiced with a character described as
        [description]? Return their persona_id and name if yes."

        Returns None on first encounter or ambiguous match.
        Latency: 200-400ms (Dialectic call). Run at IntakeComplete.
        """

    async def bootstrap(
        self,
        *,
        caller_hash: str,
        description: str,
        gender: Literal["male", "female"],
        name: str | None,
    ) -> PersonaRecord:
        """Create a new practice partner peer and generate their soul.

        1. Generate soul document via Haiku (base template + description)
        2. Create character Honcho peer (character_{caller_hash[:8]}_{slug})
        3. Store PersonaRecord in caller peer metadata["personas"]
        4. Return PersonaRecord

        Latency: ~600ms (soul generation). Run async after IntakeComplete.
        """

    async def get_soul(
        self,
        caller_hash: str,
        persona_id: str,
    ) -> str:
        """Return the two-layer soul for a persona.

        Layer 1: Bootstrap soul from caller peer metadata (static).
        Layer 2: Dialectic synthesis from character peer (grows with sessions).

        Merged: "{layer_1}\n\n## What practice has revealed:\n{layer_2}"
        Layer 2 is empty on session 1; meaningful by session 3-5.

        Latency: ~300ms (one Dialectic call). Called once per session start.
        """

    async def record_session(
        self,
        caller_hash: str,
        persona_id: str,
        session_id: str,
        messages: list[dict],
    ) -> None:
        """Store session transcript with correct peer attribution.

        Creates Honcho session with caller + coach + character peers.
        Splits message attribution by speaker tag.
        Increments session counter; triggers soul refresh at multiples of 5.
        """

    async def maybe_refresh_soul(
        self,
        caller_hash: str,
        persona_id: str,
    ) -> None:
        """Refresh the bootstrap soul if session count is a multiple of 5.

        Non-blocking: runs post-call. Queries character peer Dialectic for
        accumulated observations and regenerates soul with that context.
        """
```

### 5.2 `CounterpartyExtraction` (data class)

```python
@dataclass
class CounterpartyExtraction:
    gender: Literal["male", "female", "unknown"]
    name: str | None
    description: str
    gender_is_inferred: bool  # False only if transcript has no signal at all

    @property
    def needs_gender_question(self) -> bool:
        return not self.gender_is_inferred or self.gender == "unknown"
```

### 5.3 `PersonaRoutingAgent`

```python
class PersonaRoutingAgent:
    """Extracts counterparty context and resolves a PersonaRecord."""

    def __init__(
        self,
        peer_store: PeerStore,
        *,
        client: AsyncAnthropic,
        caller_hash: str,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None: ...

    async def extract(self, transcript: str) -> CounterpartyExtraction:
        """One LLM call. Returns counterparty gender, name, description.
        max_tokens=80, temperature=0. Latency: ~150ms.
        """

    async def select(self, transcript: str) -> PersonaRecord:
        """Extract → search Honcho → bootstrap on miss → return PersonaRecord."""
```

### 5.4 Modal eval judge

The `VLLMAudioProvider` (`rehearse/eval/providers/vllm.py`) is unchanged. It
calls any OpenAI-compatible endpoint at `VLLM_BASE_URL`. For eval runs, set
`VLLM_BASE_URL` to the Modal deployment endpoint.

**Modal deployment (`modal_app/gemma_judge.py`):**

```python
import modal

app = modal.App("rehearse-gemma-judge")
image = modal.Image.debian_slim().pip_install("vllm>=0.4", "huggingface_hub")

@app.cls(
    gpu="A100-80GB",
    image=image,
    timeout=300,
    container_idle_timeout=60,  # scale to zero after 60s idle
)
class GemmaJudge:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM("google/gemma-3-27b-it", dtype="bfloat16")

    @modal.web_endpoint(method="POST")
    async def chat_completions(self, request: dict) -> dict:
        """OpenAI-compatible /v1/chat/completions endpoint.
        VLLMAudioProvider points at this via VLLM_BASE_URL."""
        ...
```

**Configuration for eval runs:**
```bash
VLLM_BASE_URL=https://your-org--rehearse-gemma-judge.modal.run/v1
VLLM_API_KEY=<modal-api-key>
```

No changes to `VLLMAudioProvider`. No changes to eval tests. The endpoint is
drop-in compatible with the existing OpenAI client in the provider.

---

## 6. Per-call flows

### 6.1 First encounter — "I need to call my dad"

```
INTAKE:
  Transcript: "I need to call my dad and tell him I crashed his car."
  IntakeProcessor collects user turns.
  No gender question (inference will handle it).

INTAKE COMPLETE:
  PersonaRoutingAgent.extract(transcript)
  → {gender: "male", name: null,
     description: "caller's father, upset about car crash",
     gender_is_inferred: true}

  PersonaRoutingAgent.select():
  → peer_store.search(caller_hash, "caller's father, upset...")
    → Dialectic: "Has this caller practiced with this character?" → None
  → peer_store.bootstrap(caller_hash, description, gender="male", name=None)
    → Haiku generates soul: "David is a man who built something..."
    → character peer created: character_abc123ef_callers_father
    → soul + record stored in caller peer metadata["personas"]
  → returns PersonaRecord(id="male_callers_father", voice="Wise Man", ...)

  session.selected_persona_id = "male_callers_father"

PRACTICE TRANSITION:
  peer_store.get_soul(caller_hash, "male_callers_father")
  → Layer 1: bootstrap soul (David, father, accountability...)
  → Layer 2: character peer Dialectic → "" (no sessions yet)
  → returns Layer 1 only

  PersonaSwapCoordinator speaks bridge line.
  session_settings: {voice_id: <wise_man_id>, system_prompt: soul + scene}

END OF CALL:
  peer_store.record_session(caller_hash, "male_callers_father", session_id, transcript)
  → Honcho session created with caller + coach + character peers
  → Practice messages attributed to character_abc123ef_callers_father
  → Deriver queued to process character peer observations
  → peer_store.maybe_refresh_soul() — sessions=1, no refresh yet
```

### 6.2 Return call — same caller, same topic

```
INTAKE COMPLETE:
  PersonaRoutingAgent.select():
  → peer_store.search(caller_hash, "caller's father, upset about car")
    → Dialectic: "Yes — practiced with male_callers_father (David)"
    → returns stored PersonaRecord
  → No bootstrap needed.

PRACTICE START:
  peer_store.get_soul(caller_hash, "male_callers_father")
  → Layer 1: same bootstrap soul
  → Layer 2: character peer Dialectic (session 3+):
    "This father goes silent before escalating. Direct apology de-escalates.
    Softened twice when caller acknowledged feelings before explaining."
  → returns Layer 1 + "## What practice has revealed:\n{Layer 2}"
```

### 6.3 New topic, same caller — "I want to speak with my executive assistant Sarah"

```
PersonaRoutingAgent.extract():
→ {gender: "female", name: "Sarah",
   description: "caller's executive assistant",
   gender_is_inferred: true}

peer_store.search(caller_hash, "executive assistant named Sarah") → None

peer_store.bootstrap(caller_hash, "executive assistant", gender="female", name="Sarah")
→ soul: "Sarah is professional and warm. She manages up as naturally as she manages down..."
→ character peer: character_abc123ef_sarah_exec_assistant
→ stored under caller peer: metadata["personas"]["female_sarah_exec_assistant"]
```

---

## 7. Soul document: writing guide

A soul document defines who the practice partner **is** — not what they do in
the scene, but who they choose to be. Every section should be answerable from
inside the character: *if this person had to make a choice the script didn't
cover, what would they do?*

**Six sections. Wrong vs right:**

| Section | Wrong | Right |
|---|---|---|
| Identity | "David is a manager." | "David built the team from scratch. He treats every failure as a personal accusation." |
| Values | "Values hard work and honesty." | "Would rather be lied to than pitied. Hates wasted time more than anything." |
| Communication | "Direct communicator." | "Short sentences when calm. Goes quiet right before he gets angry. Never raises his voice." |
| Emotional landscape | "Gets frustrated when challenged." | "Disappointment looks like silence. He doesn't yell — he withdraws, then comes back harder." |
| Relationship | "Has authority over the caller." | "Sees the caller as an extension of himself. Proud of them in ways he rarely says out loud." |
| Edges | "May break character if asked." | "Won't cry. Won't apologize first. Will stop if the caller says they need a break." |

**Rules:**
- Specificity beats generality at every level.
- Contradiction is realistic — a character can be both proud and cold. Don't flatten.
- The Edges section is the agent's safety rail. Explicit edges let the model stay
  in character confidently because it knows exactly where the ground is.
- Write from the inside. The document should read as written *about* the person,
  not as instructions *to* an AI.

**Base template:** `rehearse/personas/souls/base_character.md`

---

## 8. Soul evolution via Honcho Deriver

The soul is not static. It evolves as Honcho's Deriver processes messages from
the character peer across sessions.

### 8.1 What the Deriver learns

After 3-5 sessions, observations about the character peer include:

- *"This character escalates with silence, not volume."*
- *"Direct apology ('I was careless') de-escalates consistently; indirect apology ('that wasn't ideal') does not."*
- *"The character softened on two occasions when the caller acknowledged feelings before explaining."*
- *"Caller's defensive posture consistently triggers a harder response from this character."*

These are extracted automatically by the Deriver from the character peer's
messages across sessions. No manual annotation required.

### 8.2 Two-layer soul assembly

```
Layer 1 (static)   Bootstrap soul — who this character is before any practice
Layer 2 (dynamic)  character_peer.chat("What patterns have you observed in this
                   character across sessions?") — grows with Deriver observations

Merged output:
  {layer_1}

  ## What practice has revealed:
  {layer_2}
```

Layer 2 is queried at PRACTICE start. It is empty on session 1 and meaningful
by session 3-5. If the Dialectic returns nothing (new character, no observations),
only Layer 1 is used and the call proceeds normally.

### 8.3 Soul refresh cadence

Every 5 sessions, the bootstrap soul (Layer 1) is regenerated using the Dialectic
synthesis as additional context. This keeps Layer 1 from diverging too far from
what has been observed.

Trigger: `PeerStore.record_session()` increments `metadata["personas"][id]["sessions"]`.
At multiples of 5, queue `maybe_refresh_soul()` as a background task post-call.
The refresh is non-blocking — the next session picks up the updated soul.

### 8.4 `observe_others` activation

`character_peer.observe_others=True` is set from session 3 onwards. Sessions 1-2
establish a baseline for both peers; early observations are noisy. From session 3,
the character peer builds its own representation of the caller — eventually knowing
"this caller over-explains when nervous" and adapting accordingly.

---

## 9. Eval judge specification

### 9.1 Three-scenario eval suite

```
call_1  New caller, any topic
        → PersonaRoutingAgent extracts counterparty
        → peer_store.bootstrap() called (first encounter)
        → soul generated and stored
        Judge checks: counterparty extracted correctly, bootstrap fired

call_2  Same caller, same counterparty
        → peer_store.search() finds prior persona
        → No bootstrap
        → Same voice used
        Judge checks: no re-bootstrap, same persona_id in session

call_3  Same caller, new counterparty (different person entirely)
        → search returns None
        → bootstrap fires for new persona
        → Different voice if different gender
        Judge checks: new persona created, correct voice
```

### 9.2 Gemma audio judge (via Modal)

**Routing correctness judgment (text):**

Input: intake transcript + `routing_eval_result.json`

```
You are evaluating a voice coaching call. Given the intake transcript and the
routing decision made, determine whether the routing was correct.

Transcript: {transcript}
Routing decision: {result}

Return JSON: {"routing_correct": bool, "reasoning": str}
```

**Voice verification judgment (audio):**

Input: WAV clip of first 10s of character speech in practice phase.

```
Listen to this audio. Is the speaker male or female?
Answer with exactly one word: "male" or "female".
```

Expected answer matches `persona.gender`.

### 9.3 Running evals

```bash
# Deploy Modal judge once (idempotent)
modal deploy modal_app/gemma_judge.py

# Run eval suite
VLLM_BASE_URL=https://your-org--rehearse-gemma-judge.modal.run/v1 \
VLLM_API_KEY=<key> \
uv run pytest tests/eval/test_persona_voice_routing_eval.py \
  -v -m "live_api and live_honcho" --timeout=180
```

---

## 10. Non-functional requirements

| Requirement | Target | Fallback |
|---|---|---|
| Counterparty extraction latency | ≤ 400ms | Falls back to gender question if extraction times out |
| Soul retrieval at session start | ≤ 500ms (Layer 1 + Layer 2) | Uses Layer 1 only if Dialectic times out |
| Soul bootstrap latency | ≤ 800ms (async, post-intake) | Uses default soul template on timeout |
| `session_settings` voice swap | Before bridge line ends | No voice swap; character prompt still correct |
| Modal cold start | ≤ 30s (eval only) | Retry with 30s timeout; acceptable for batch eval |
| Cost per eval run (3 scenarios) | ≤ $0.10 | N/A |
| Cost when idle | $0.00 (Modal scales to zero) | N/A |

---

## 11. Implementation phases

### Phase 1 — Core routing + PeerStore (this PR)

Delivers: O1, O2, O3, O4 from §2.

**Changes:**

| File | Change |
|---|---|
| `rehearse/personas/peer_store.py` | **New** — `PeerStore` with `search`, `bootstrap`, `get_soul`, `record_session`, `maybe_refresh_soul` |
| `rehearse/personas/souls/base_character.md` | **Done** — base soul template |
| `rehearse/agents/persona_routing_agent.py` | Replace `list_personas` tool with `extract()` + `PeerStore.search/bootstrap` |
| `rehearse/agents/persona_selection_recorder.py` | Gate gender question on `CounterpartyExtraction.needs_gender_question` |
| `rehearse/agents/roles/character.py` | `system_prompt()` calls `peer_store.get_soul()` at session start |
| `rehearse/memory.py` | Add `get_raw_metadata` / `set_raw_metadata` to `CallerMemory` protocol + all implementations |
| `rehearse/personas/__init__.py` | Remove `classify_gender_preference`; keep coach elicitation prompt |
| `rehearse/intake.py` | Remove keyword detection; gender question gated by `PersonaSelectionRecorder` |

**Tests:**

| File | What it covers |
|---|---|
| `tests/test_peer_store.py` | search, bootstrap, get_soul (layers), record_session attribution |
| `tests/test_persona_routing_agent.py` | LLM extraction correctness; search-before-bootstrap order |
| `tests/test_intake_gender_question.py` | Question fires on ambiguous transcript; silent on "my dad" |

### Phase 2 — Soul evolution + Deriver (follow-on)

Delivers: O5, O6 from §2.

**Changes:**

- `PeerStore.record_session()`: add character peer with `observe_me=True`; split message attribution by `speaker` tag
- `PeerStore.get_soul()`: add Layer 2 Dialectic query on character peer
- `PeerStore.maybe_refresh_soul()`: implement refresh at multiples of 5 sessions
- `CallerMemory.store_session()`: pass `speaker` tags through so PeerStore can split attribution

**Tests:**

- `tests/test_peer_store_evolution.py`: session 1 has empty Layer 2; session 5 triggers refresh
- `tests/test_agent_memory_multisession.py`: end-to-end soul evolution over 3 sessions

### Phase 3 — Modal eval judge

Delivers: O7, O8 from §2.

**Changes:**

- `modal_app/gemma_judge.py`: Gemma 3 27B on A100, OpenAI-compatible endpoint
- `tests/eval/test_persona_voice_routing_eval.py`: 3-scenario eval using `VLLMAudioProvider`
- `.env.example`: add `VLLM_BASE_URL` + `VLLM_API_KEY` for eval runs

---

## 12. Artifacts produced by eval

| Artifact | Format | Contents |
|---|---|---|
| `transcript.jsonl` | JSONL | Full transcript per call |
| `routing_eval_result.json` | JSON | `{call_id, persona_id, bootstrap_fired, routing_correct, reasoning}` |
| `voice_eval_result.json` | JSON | `{call_id, expected_gender, gemma_judgment, pass}` |
| `character_audio_clip.wav` | WAV | First 10s of character speech (Gemma input) |
| `peer_store_state.json` | JSON | Caller peer metadata snapshot after all calls |
| `soul_evolution.json` | JSON | Layer 1 + Layer 2 per call, showing Deriver growth |

---

## 13. Open questions (resolved)

| # | Question | Resolution |
|---|---|---|
| Q1 | Should souls evolve? | Yes — via Deriver on character peer. §8 documents the mechanism. |
| Q2 | Wise Man voice name? | Must verify in Hume dashboard before Phase 1 ships. Env var `REHEARSE_MALE_VOICE_NAME` overrides. |
| Q3 | `session_settings` timing? | Test empirically in Phase 1. If it applies mid-TTS, shorten bridge line to 1 sentence. |
| Q4 | When to enable `observe_others`? | Session 3 onwards. Sessions 1-2 are noisy; session 3 has enough signal. |
