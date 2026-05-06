# LLM-classified persona routing from inbound SMS

**Status:** draft
**Date:** 2026-05-06
**Owner:** Josh Zastrow
**Related code:** `rehearse/telephony.py`, `rehearse/services/hume_evi.py`, `rehearse/services/hume_configs.py`, `rehearse/session.py`

## Goal

Pick the right Hume EVI config (voice + greeting + prompt) for each call from
the inbound SMS body, before the call connects. The routing is decided once
at SMS time by a small LLM classifier, recorded on the session manifest, and
read by `HumeEVIClient._connect` to pick the right config id.

## Non-goals

- Mid-call persona switching. The persona chosen at SMS time runs end-to-end.
- Implementing the existing `HumeEVIClient.swap_config` stub.
- Persona inference from the live audio/transcript stream.
- Hand-engineered persona-specific prompts. The new `relationship_coach`
  ships with a near-copy of the default prompt; prompt-engineering is a
  separate follow-up.
- Scripted phase-handoff bridge lines (option B from the time-card brainstorm
  — separate spec).

## Design

### Schema additions

`HumePersonaConfig` (in `rehearse/services/hume_configs.py`) gains:

```python
routing_description: str
```

A one-sentence description of when this persona should be picked. Used only
to build the classifier prompt; never POSTed to Hume. Required field, no
default — every persona must declare its routing intent.

`Session` (in `rehearse/types.py`) gains:

```python
persona_key: str = "default"
```

Default for backward compatibility with sessions created before this lift.

### Persona registry expansion

`PERSONAS` gains a second entry:

```python
"relationship_coach": HumePersonaConfig(
    persona_key="relationship_coach",
    display_name="Rehearse Coach (relationship)",
    routing_description=(
        "Romantic relationships, partners, dating, breakups, "
        "intimacy issues, marriage, divorce."
    ),
    voice=HumeVoice(name="Inspiring Woman", provider="HUME_AI"),  # same voice in v1
    language_model=HumeLanguageModel(
        provider="ANTHROPIC", model="claude-sonnet-4-20250514"
    ),
    prompt_text=_DEFAULT_PROMPT,  # near-copy with one-line override below
    on_new_chat=HumeEventMessage(
        enabled=True, text="Hey, glad you called. What's going on?"
    ),
    # all other fields default
)
```

The default persona also gets a `routing_description`:

```python
routing_description=(
    "General conversation rehearsal — work conversations, family, "
    "negotiations, pitches, anything that isn't romantic."
),
```

The `relationship_coach` prompt is the default `_DEFAULT_PROMPT` verbatim
plus one prepended sentence: *"You're acting as a relationship coach. The
caller is rehearsing a conversation with a romantic partner."* This is a
deliberate stub — full prompt-engineering for this persona is a follow-up
(see Out of scope).

### Persona router

New module `rehearse/agents/persona_router.py`:

```python
async def infer_persona_key(
    sms_body: str,
    personas: dict[str, HumePersonaConfig],
    *,
    anthropic_client: AsyncAnthropic | None,
    model: str,
    fallback: str = "default",
) -> str:
    """Classify an SMS body into one of the registered persona keys.

    Returns `fallback` when:
    - `anthropic_client` is None (no API key configured),
    - the SMS body is empty or the literal string "<inbound-call>",
    - the classifier raises any exception,
    - the classifier returns a key not in `personas`.

    All fallback paths are logged (structlog) with the SMS body and the
    reason so we can audit miss-classifications.
    """
```

The classifier prompt is built from `personas` — it lists each
`persona_key` + `routing_description` and asks the model to return *only*
the chosen key as plain text. Uses Claude Haiku (`claude-haiku-4-5-20251001`)
via the existing Anthropic client. `max_tokens=20`, `temperature=0`.

### Wiring at SMS time

In `rehearse/telephony.py::twilio_sms`, between the existing `orchestrator.start(trigger)`
and the background `_place` task:

```python
persona_key = await infer_persona_key(
    Body,
    PERSONAS,
    anthropic_client=anthropic_client,  # cached at app startup
    model=config.anthropic_model,
)
await orchestrator.set_persona_key(handle.session_id, persona_key)
```

`set_persona_key` is a new method on the orchestrator that mutates the
session manifest via the existing `update_session` store hook. The
inference runs in the foreground because it's ~200 ms and gating
`place_call` on it makes the persona decision deterministic by the time the
call rings the user's phone.

The shared `AsyncAnthropic` client is constructed at app startup
(`rehearse/app.py`) when `config.anthropic_api_key` is set, and passed into
the telephony route registration. When the key is absent, `None` is passed
and inference is skipped.

### Wiring at connect time

`HumeEVIClient.__init__` gains:

```python
def __init__(
    self,
    *,
    api_key: str,
    config_id: str,                       # existing — used as fallback
    persona_key: str = "default",         # NEW
    bus: FrameBus,
    session_id: str,
    ...
) -> None:
    ...
    self._fallback_config_id = config_id
    self._persona_key = persona_key
```

`_connect` resolves the actual config id at call time:

```python
resolved_config_id = select_config_id(
    self._persona_key,
    fallback=self._fallback_config_id,
)
```

and passes that into the connect call instead of `self._config_id`.

Wherever `HumeEVIClient` is constructed today (look at `rehearse/pipeline.py`
and the orchestrator), pass `persona_key=session.persona_key`.

### Fallback semantics summary

- LLM client unavailable → `"default"`.
- Empty / `"<inbound-call>"` SMS body → `"default"` (no signal to classify).
- LLM exception or unknown key → `"default"`, logged.
- `"default"` not in `PERSONAS` → `select_config_id` returns the env
  `hume_config_id`, preserving today's behavior.
- Mapping file missing or persona key absent from mapping → env
  `hume_config_id` (already implemented in `select_config_id`).

## Test plan

`tests/test_persona_router.py`:

- Happy path: SMS "I want to rehearse breaking up with my partner" → `"relationship_coach"` (mocked classifier).
- Happy path: SMS "I'm asking my boss for a raise" → `"default"`.
- Fallback: classifier raises → `"default"` returned, error logged.
- Fallback: classifier returns `"unknown_persona"` → `"default"` returned, warning logged.
- Fallback: `anthropic_client=None` → `"default"` returned, no LLM call attempted.
- Fallback: `sms_body=""` → `"default"` returned, no LLM call attempted.
- Fallback: `sms_body="<inbound-call>"` → `"default"` returned, no LLM call attempted.

`tests/test_telephony_r1.py` (extend existing tests):

- POST `/twilio/sms` with relationship-themed body causes the persisted
  session manifest to have `persona_key="relationship_coach"` (classifier
  mocked).
- POST `/twilio/sms` with the classifier monkeypatched to raise still
  succeeds (returns 200, session has `persona_key="default"`).

`tests/test_hume_evi.py`:

- `HumeEVIClient(persona_key="relationship_coach", ...)` with a mapping file
  containing `"relationship_coach": "cfg_xyz"` calls connect with
  `config_id="cfg_xyz"`.
- Same client with no mapping file uses the fallback `config_id`.

`tests/test_hume_configs.py`:

- Both `"default"` and `"relationship_coach"` are in `PERSONAS`.
- Both have a non-empty `routing_description`.

## Migration

After this lift lands, run `uv run rehearse-hume sync` to push the new
`relationship_coach` config to the Hume workspace. The default config is
already synced from the previous lift; only the new persona will trigger a
`CREATE` action. The mapping file gains a `"relationship_coach"` entry.

If the operator hasn't run `sync`, calls still work — `select_config_id`
falls back to the env `hume_config_id` for any persona key not in the
mapping file. The default persona is unaffected.

## Out of scope (follow-ups)

- Persona-specific prompt engineering for `relationship_coach`. Ships with
  a near-copy of the default + one framing sentence; refining the prompt
  for the relationship use case is a separate task.
- Re-routing mid-call when the SMS classification was wrong. Requires
  implementing `swap_config`; deferred to a future spec.
- Persona-specific voices. v1 uses the same `Inspiring Woman` voice for
  both personas to keep this lift bounded.
- Non-LLM offline-mode classifier. The `"default"` fallback when no
  Anthropic key is configured is sufficient for local dev.
- Persona-aware time card (`rehearse/agents/timecard.py` already takes
  phase + budgets; persona doesn't influence pacing).
- Confidence thresholds / "ambiguous" handling. v1 always picks the
  classifier's first answer.
