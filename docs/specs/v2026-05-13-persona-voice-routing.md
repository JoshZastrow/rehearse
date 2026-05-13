# Persona Voice Routing: Memory-Backed Practice Partner Selection

**Status:** wip
**Date:** 2026-05-13
**Owner:** Josh Zastrow
**Depends on:**
- `v2026-05-12-agent-design-patterns.md` — `AgentRouter`, `CallerMemory`, `IntakeAwareRouter`

---

## 1. Vision

Every practice partner should feel like a real person — not a generic male or female voice, but someone with a specific name, history, and way of being in the world. A caller practicing a conversation with their dad gets their actual dad's energy. Someone practicing with their executive assistant Sarah gets the Sarah they described, built from what they said about her, persistent across calls.

The system should never ask for information it already has. If the caller says "I need to call my dad and tell him I crashed his car — he's going to be pretty mad," the intake agent knows who they're practicing with and what kind of character is needed. No gender question. No menu. The routing agent infers everything from the conversation and bootstraps a persona from a soul template on first encounter. Subsequent calls find the same persona in Honcho memory.

This spec designs the infrastructure that makes N personas possible without seeding N cases:
- **Soul documents** define who a persona is (not what they do)
- **PersonaStore** backed by Honcho stores bootstrapped personas and finds them semantically
- **LLM extraction** replaces keyword-based gender inference
- **PersonaRoutingAgent** searches memory first, bootstraps on miss

---

## 2. What the current design gets right and wrong

**Right:**
- `PersonaRoutingAgent` runs at `IntakeComplete` with a tool call — the right hook point
- `session_settings` swaps the voice mid-call without disconnecting
- `CallerMemory` with Honcho is the right persistence layer

**Wrong:**
- `classify_gender_preference()` is a keyword matcher — fails for "my dad", "Sarah my assistant", any relationship the keyword list doesn't cover
- `PersonaRegistry` is a static Python list — can't grow to N personas without code changes
- The gender question fires even when the counterparty is obvious from context
- `list_personas` tool queries a hardcoded registry, not Honcho memory

---

## 3. Soul documents

A soul document defines who a practice partner is — not their job description, not their instructions, but their identity. It answers the question: if this character existed in the world, who would they choose to be?

The reference format is Claude's own soul document (Anthropic's model spec). It covers identity, values, psychological groundedness, consistency across contexts, and emotional landscape. A Rehearse persona soul document follows the same structure, adapted to a character who exists for one call at a time.

### 3.1 Soul document format

```markdown
# [Name] — Soul

## Identity

Who [Name] is at their core. Not their role or job title — their essential nature.
What shaped them. What they carry into every room.

Example: "David is a man who built something from nothing and doesn't forget the cost.
He's proud — not arrogant — and holds people to standards he holds himself to first."

## Values

What they care about most. What they'd never compromise on.
What they'd sacrifice to protect.

## How they communicate

Their rhythm. Their tells. What they say when they're comfortable versus threatened.
How they use silence. How they use humor (if at all). What they avoid saying directly.

## Emotional landscape

Their default state. What activates them — the triggers that shift their tone.
How they handle pushback. What makes them soften. What makes them harder.
What they look like when they're scared versus angry versus proud.

## Relationship to the caller

How they see the person across from them.
What they want for that relationship — consciously and not.
What they'd never do to them, even in conflict.

## Edges

Where the character ends. What breaks the frame.
What this persona won't do, even if asked.
What pulls them out of themselves.
```

### 3.2 The base soul template

`rehearse/personas/souls/base_character.md` is the template used when bootstrapping an entirely new persona. It contains placeholder text in each section that the LLM fills in from the caller's description.

The bootstrapper receives:
- The counterparty description (extracted from intake transcript by LLM)
- The gender
- The name (if given)
- The relationship context

It generates a complete soul document by specializing the base template. The result is stored in Honcho under this caller's peer.

### 3.3 Soul retrieval at session start

At the start of the PRACTICE phase, the character agent retrieves the soul from Honcho and injects it into the system prompt:

```python
# In CharacterAgent.system_prompt():
soul = await self._persona_store.get_soul(
    caller_hash=session.phone_number_hash,
    persona_id=session.selected_persona_id,
)
if soul:
    prompt = f"You are {persona.name}.\n\n{soul}\n\n---\n\n{character_prompt}"
else:
    prompt = character_prompt
```

The soul is fetched once per session and cached. It is not re-fetched on every CLM turn.

The soul document is prepended to the character prompt, not appended. It sets who the character is before the character prompt says what they're doing. The character prompt handles the scene; the soul handles the person playing the scene.

---

## 4. PersonaStore — Honcho-backed, N personas

`PersonaStore` replaces the static `PersonaRegistry`. It stores and retrieves bootstrapped personas using Honcho peer metadata and the Dialectic for semantic search.

```python
# rehearse/personas/store.py

class PersonaStore:
    """Honcho-backed persona storage with bootstrap-on-miss."""

    def __init__(self, memory: CallerMemory) -> None:
        self._memory = memory

    async def search(
        self,
        caller_hash: str,
        description: str,
    ) -> PersonaRecord | None:
        """Find a previously used persona matching this description.

        Uses Honcho's Dialectic to answer: "has this caller practiced with
        a character matching [description]? If so, return their persona id
        and details."

        Returns None on first encounter or if no match is found.
        """
        result = await self._memory.prefetch(
            caller_hash,
            f"Has this caller previously practiced with a character described as: {description}? "
            f"If yes, return the persona_id and their name."
        )
        if not result:
            return None
        return self._parse_persona_from_recall(result)

    async def store(self, caller_hash: str, persona: PersonaRecord) -> None:
        """Persist a persona record to Honcho metadata for future recall."""
        # Stored as a serialized JSON blob in peer metadata under "personas"
        current = await self._memory.get_raw_metadata(caller_hash)
        personas = current.get("personas", {})
        personas[persona.id] = persona.to_dict()
        await self._memory.set_raw_metadata(caller_hash, {**current, "personas": personas})

    async def get_soul(
        self,
        caller_hash: str,
        persona_id: str,
    ) -> str | None:
        """Retrieve the soul document for a specific persona."""
        current = await self._memory.get_raw_metadata(caller_hash)
        personas = current.get("personas", {})
        record = personas.get(persona_id, {})
        return record.get("soul")

    async def bootstrap(
        self,
        *,
        caller_hash: str,
        description: str,
        gender: str,
        name: str | None,
        llm_client: Any,
    ) -> PersonaRecord:
        """Bootstrap a new persona from the caller's description and a soul template.

        Generates a soul document using the LLM, assigns a voice, creates a
        PersonaRecord, and stores it in Honcho for future recall.
        """
        soul = await self._generate_soul(
            description=description,
            gender=gender,
            name=name,
            llm_client=llm_client,
        )
        persona_id = f"{gender}_{_slug(name or description)}"
        voice_name = "Inspiring Woman" if gender == "female" else "Wise Man"
        persona = PersonaRecord(
            id=persona_id,
            name=name or _infer_name(description, gender),
            description=description,
            gender=gender,
            voice_name=voice_name,
            system_prompt_template=soul,
            tags=[gender, "bootstrapped"],
        )
        await self.store(caller_hash, persona)
        return persona

    async def _generate_soul(
        self,
        *,
        description: str,
        gender: str,
        name: str | None,
        llm_client: Any,
    ) -> str:
        """Use the LLM to specialize the base soul template for this character."""
        base_template = _load_base_soul_template()
        prompt = (
            f"Generate a soul document for a practice partner with the following description:\n\n"
            f"Name: {name or 'unknown'}\n"
            f"Gender: {gender}\n"
            f"Description: {description}\n\n"
            f"Use this template:\n\n{base_template}\n\n"
            f"Fill in each section based on the description. Be specific and grounded. "
            f"This person should feel real, not like a character archetype."
        )
        response = await llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
```

---

## 5. PersonaRoutingAgent — general-purpose extraction

The routing agent no longer uses keyword matching or a static tool. It does two things in sequence:

**Step 1 — Extract counterparty context from intake transcript.**

```python
extraction_prompt = """
Read this intake transcript and extract information about the practice partner.

Transcript:
{transcript}

Return JSON:
{
  "gender": "male" | "female" | "unknown",
  "name": "<first name if mentioned, else null>",
  "description": "<one sentence: who is this person to the caller>",
  "gender_is_inferred": true | false
}

Examples:
- "I need to call my dad" → {"gender": "male", "name": null, "description": "caller's father", "gender_is_inferred": true}
- "I want to speak with my executive assistant Sarah" → {"gender": "female", "name": "Sarah", "description": "caller's executive assistant", "gender_is_inferred": true}
- "I need to practice with someone" → {"gender": "unknown", "name": null, "description": "unspecified", "gender_is_inferred": false}

Only set gender_is_inferred to false if the transcript gives no signal at all.
"""
```

**Step 2 — Search or bootstrap.**

```python
async def select(self, transcript: str) -> PersonaRecord:
    # 1. Extract counterparty context from transcript
    extraction = await self._extract(transcript)
    gender = extraction.get("gender", "unknown")
    name = extraction.get("name")
    description = extraction.get("description", "")

    # 2. Search Honcho for a prior persona matching this description
    if self._caller_hash:
        prior = await self._persona_store.search(self._caller_hash, description)
        if prior:
            return prior

    # 3. Bootstrap a new persona from the description
    if self._caller_hash and description and description != "unspecified":
        return await self._persona_store.bootstrap(
            caller_hash=self._caller_hash,
            description=description,
            gender=gender if gender != "unknown" else "female",
            name=name,
            llm_client=self._llm_client,
        )

    # 4. Final fallback: default for detected gender
    resolved_gender = gender if gender in ("male", "female") else "female"
    registry = PersonaRegistry(PERSONA_REGISTRY)
    return registry.default(gender=resolved_gender)
```

---

## 6. Gender question gating

Because the routing agent extracts gender from the transcript using the LLM, the intake coach only asks the question when extraction returns `"gender_is_inferred": false`.

The `PersonaSelectionRecorder` checks the extraction result stored on the session before deciding whether to ask. If gender is already known from context, it sets `session.selected_persona_id` directly without prompting. If not, it leaves `gender_preference = None` and the coach asks naturally.

The coach prompt is already written to ask only once and accept any answer. This gating just prevents it from asking when the answer is already in the conversation.

---

## 7. Full per-call flow

```
Caller: "I need to call my dad and tell him I crashed his car. He's going to be pretty mad."

INTAKE PHASE:
  IntakeProcessor watches transcript, collects user turns.
  Coach does NOT ask gender preference (dad is clearly male).

INTAKE COMPLETE:
  PersonaSelectionRecorder fires.
  PersonaRoutingAgent.select(transcript):
    → extract({gender: male, name: null, description: "caller's father, upset about crashed car"})
    → persona_store.search(caller_hash, "caller's father, upset about crashed car")
      → Honcho: "has this caller practiced with a father figure before?"
      → First time: not found
    → persona_store.bootstrap(
          caller_hash, description, gender=male, name=None, ...)
      → LLM generates soul document for: "a father reacting to his child's mistake"
      → PersonaRecord(id="male_callers_father_upset", voice="Wise Man", soul=...)
      → Stored in Honcho
  session.selected_persona_id = "male_callers_father_upset"

PRACTICE TRANSITION:
  PersonaSwapCoordinator speaks: "Just a moment while I connect you with your practice partner."
  Simultaneously sends session_settings:
    → voice_id = resolve_voice_id("Wise Man")
    → system_prompt = soul + character_prompt

  The practice partner picks up as the caller's dad — gruff, disappointed, but recognizably a father.

SECOND CALL, SAME TOPIC:
  PersonaRoutingAgent.select(transcript):
    → extract({gender: male, description: "caller's father..."})
    → persona_store.search(caller_hash, "...father...")
      → Honcho finds "male_callers_father_upset"
      → Returns stored PersonaRecord with same soul
  Same voice. Same character. Consistent across calls.
```

---

## 8. How to write a soul document

A soul document is written when a persona is bootstrapped. The LLM writes it from the base template using what the caller said about the person. But developers can also write souls manually for any character they want to pre-define.

**The base template is at:** `rehearse/personas/souls/base_character.md`

**Writing guidance:**

The soul document describes who a person **is**, not what they **do**. Every section should answer: if this person existed in the world and had to make a choice no script told them about, what would they do?

| Section | Wrong | Right |
|---|---|---|
| Identity | "David is a manager who oversees a team." | "David built the team from scratch and treats every failure as a personal accusation." |
| Values | "Values honesty and hard work." | "Would rather be lied to than pitied. Hates when people waste his time more than anything." |
| Communication | "Direct communicator." | "Speaks in short sentences when calm. Goes quiet right before he gets angry. Never raises his voice." |
| Emotional landscape | "Gets frustrated when challenged." | "His disappointment looks like silence. He doesn't yell — he withdraws. Then he comes back harder." |
| Relationship | "Has authority over the caller." | "Sees the caller as an extension of himself. Proud of them in ways he rarely says out loud." |
| Edges | "May break character if asked nicely." | "Won't cry. Won't apologize first. Will stop if the caller says they need a break." |

**What makes a soul document work:**

- Specificity beats generality. "Hates wasted time" is more useful than "values efficiency."
- Contradiction is realistic. A character can be both proud and cold. Don't flatten them.
- The edges section is a gift to the LLM. Tell it exactly what won't happen — so it knows where the character's ground is.
- Write from the inside. The document should read as if written *about* the person, not as instructions *to* an AI playing them.

---

## 9. System prompt injection at session start

When the PRACTICE phase begins, the soul document is loaded and injected once. This is the only time Honcho is called for soul retrieval; after this point it's cached.

```
[Character prompt — what this character is doing in this scene]

---

[Soul document — who this character chooses to be]
```

The soul comes second so the character prompt frames the immediate context and the soul grounds the emotional truth. The LLM reads the soul as background identity that informs how the scene plays out, not as additional instructions to follow.

If no soul document exists (fallback), only the character prompt is used. The practice still works — it just lacks the persistent identity layer.

---

## 10. What changes from the current implementation

| Current | Replace with |
|---|---|
| `classify_gender_preference()` keyword matcher | LLM extraction in `PersonaRoutingAgent` — works for any relationship, any name |
| `PersonaRegistry` static Python list | `PersonaStore` backed by Honcho — bootstraps on miss, recalls on return |
| `list_personas` tool queries hardcoded registry | `search_personas` queries Honcho + `bootstrap_persona` on miss |
| Coach always asks gender question | Coach only asks when extraction returns `gender_is_inferred: false` |
| `MaleCharacterAgent` / `FemaleCharacterAgent` (gender-only dispatch) | `PersonaStore` result dispatches to the bootstrapped persona |
| `get_gender_preference(caller_hash)` flat key-value | `PersonaStore.search(caller_hash, description)` semantic lookup |

---

## 11. File inventory (revised)

| File | Change |
|---|---|
| `rehearse/personas/store.py` | **New** — `PersonaStore` with `search`, `bootstrap`, `store`, `get_soul` |
| `rehearse/personas/souls/base_character.md` | **New** — base soul template with placeholder sections |
| `rehearse/agents/persona_routing_agent.py` | Replace keyword extraction with LLM extraction + `PersonaStore` lookup |
| `rehearse/agents/persona_selection_recorder.py` | Pass `PersonaStore` to routing agent; gate gender question on extraction result |
| `rehearse/agents/roles/character.py` | `CharacterAgent.system_prompt()` fetches soul from `PersonaStore` |
| `rehearse/memory.py` | Add `get_raw_metadata` / `set_raw_metadata` to `CallerMemory` for persona storage |
| `rehearse/personas/__init__.py` | Remove keyword-based `classify_gender_preference`; keep coach elicitation prompt |
| `rehearse/intake.py` | Remove `classify_gender_preference` call + coach question detection logic |
| `tests/test_persona_store.py` | **New** — search, bootstrap, soul retrieval |
| `tests/test_persona_routing_agent.py` | Update to test LLM extraction + store lookup |
| `tests/test_intake_gender_question.py` | Update: gender question only fires when extraction has no signal |

---

## 12. Open questions

| # | Question |
|---|---|
| Q1 | Should souls be versioned in Honcho? If a caller's description of their dad changes across sessions, should the soul update or stay fixed? |
| Q2 | What is the right Honcho voice name for the male voice ("Wise Man" assumed — needs verification in Hume dashboard)? |
| Q3 | Does `session_settings` apply before or after the current TTS utterance completes? Test empirically — affects bridge line timing. |
| Q4 | Should `PersonaStore.search()` use the Dialectic (LLM-backed) or a flat metadata lookup? Dialectic is more flexible but adds latency (~200ms). |
