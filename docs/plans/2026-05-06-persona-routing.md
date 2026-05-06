# Persona Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route each call to the right Hume EVI persona by classifying the inbound SMS body with Claude Haiku before the outbound call connects.

**Architecture:** A new `rehearse/agents/persona_router.py` exposes `infer_persona_key`. `Session.persona_key` carries the decision into the websocket handler, where `HumeEVIClient` resolves it to a config id via the existing `select_config_id` helper. A new `relationship_coach` persona ships in `PERSONAS` so routing has somewhere to go besides default.

**Tech Stack:** Python 3.11+, pydantic v2, `anthropic` async client (Claude Haiku), pytest, structlog.

**Spec:** `docs/specs/v2026-05-06-persona-routing.md`

---

## File Structure

- **Create:** `rehearse/agents/persona_router.py` — `infer_persona_key` + classifier prompt builder.
- **Create:** `tests/test_persona_router.py` — happy path + 5 fallback scenarios.
- **Modify:** `rehearse/services/hume_configs.py` — add `routing_description` field to `HumePersonaConfig`, add `_RELATIONSHIP_PROMPT`, add `relationship_coach` entry to `PERSONAS`, add `routing_description` to the existing `default` entry.
- **Modify:** `rehearse/types.py` — add `persona_key: str = "default"` to `Session`.
- **Modify:** `rehearse/services/hume_evi.py` — add `persona_key` kwarg to `HumeEVIClient.__init__`, resolve via `select_config_id` in `_connect`.
- **Modify:** `rehearse/session.py` — add `set_persona_key(session_id, key)` to `SessionOrchestrator`.
- **Modify:** `rehearse/telephony.py` — construct shared `AsyncAnthropic` client at app wiring time, call `infer_persona_key` after `orchestrator.start`, load `Session` before constructing `HumeEVIClient` and pass `persona_key=session.persona_key`.
- **Modify:** `tests/test_hume_configs.py` — assert both personas registered with non-empty `routing_description`.
- **Modify:** `tests/test_hume_evi.py` — assert `persona_key` resolves to mapped config id.
- **Modify:** `tests/test_telephony_r1.py` — assert SMS classification path persists `persona_key` and degrades gracefully on classifier failure.

---

## Task 1: Add `routing_description` + relationship_coach persona

**Files:**
- Modify: `rehearse/services/hume_configs.py`
- Modify: `tests/test_hume_configs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hume_configs.py`:

```python
def test_relationship_coach_persona_is_registered():
    assert "relationship_coach" in PERSONAS
    persona = PERSONAS["relationship_coach"]
    assert persona.persona_key == "relationship_coach"
    assert persona.display_name == "Rehearse Coach (relationship)"
    assert persona.routing_description != ""
    assert "relationship" in persona.routing_description.lower()


def test_default_persona_has_routing_description():
    persona = PERSONAS["default"]
    assert persona.routing_description != ""
    assert "general" in persona.routing_description.lower() or "any" in persona.routing_description.lower()


def test_relationship_prompt_extends_default_with_framing():
    persona = PERSONAS["relationship_coach"]
    default_prompt = PERSONAS["default"].prompt_text
    assert default_prompt in persona.prompt_text
    assert "relationship coach" in persona.prompt_text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hume_configs.py -v`
Expected: 3 failures (AttributeError on `routing_description`, KeyError on `relationship_coach`).

- [ ] **Step 3: Add `routing_description` to the schema**

In `rehearse/services/hume_configs.py`, add to `HumePersonaConfig` (between `display_name` and `evi_version`):

```python
    routing_description: str
```

(No default — every persona must declare its routing intent.)

- [ ] **Step 4: Add `routing_description` to the existing default persona**

In the `PERSONAS["default"]` entry, add the field right after `display_name=...`:

```python
        routing_description=(
            "General conversation rehearsal — work conversations, family, "
            "negotiations, pitches, anything that isn't romantic."
        ),
```

- [ ] **Step 5: Add `_RELATIONSHIP_PROMPT` and the `relationship_coach` entry**

Below `_DEFAULT_PROMPT`, add:

```python
_RELATIONSHIP_PROMPT = (
    "You're acting as a relationship coach. The caller is rehearsing a "
    "conversation with a romantic partner.\n\n" + _DEFAULT_PROMPT
)
```

In the `PERSONAS` dict, add a second entry alongside `"default"`:

```python
    "relationship_coach": HumePersonaConfig(
        persona_key="relationship_coach",
        display_name="Rehearse Coach (relationship)",
        routing_description=(
            "Romantic relationships, partners, dating, breakups, "
            "intimacy issues, marriage, divorce."
        ),
        evi_version="4-mini",
        voice=HumeVoice(name="Inspiring Woman", provider="HUME_AI"),
        language_model=HumeLanguageModel(
            provider="ANTHROPIC",
            model="claude-sonnet-4-20250514",
        ),
        prompt_text=_RELATIONSHIP_PROMPT,
        on_new_chat=HumeEventMessage(
            enabled=True, text="Hey, glad you called. What's going on?"
        ),
        on_max_duration_timeout=HumeEventMessage(enabled=True, text=None),
        on_inactivity_timeout=HumeEventMessage(enabled=False, text=None),
        timeouts=HumeTimeouts(max_duration_secs=300, inactivity_secs=122),
        turn_detection=HumeTurnDetection(
            end_of_turn_silence_ms=500,
            prefix_padding_ms=300,
            speech_detection_threshold=0.4,
        ),
        interruption_min_ms=800,
        nudges_enabled=True,
        nudges_interval_secs=8,
        builtin_tools=["web_search", "hang_up"],
    ),
```

- [ ] **Step 6: Add `routing_description` to `_COMPARED_FIELDS` decision**

`routing_description` is for the classifier prompt only — it must NOT be sent to Hume and must NOT be in the diff comparison (Hume has no field for it). Verify `_COMPARED_FIELDS` does not contain `"routing_description"`. Verify `_to_create_kwargs` does not pass it. No code change should be needed if you appended the field correctly; just confirm.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_hume_configs.py tests/test_hume_configs_cli.py -v`
Expected: all pass (existing 11 + 3 new = 14, plus the CLI tests unchanged).

- [ ] **Step 8: Commit**

```bash
git add rehearse/services/hume_configs.py tests/test_hume_configs.py
git commit -m "add relationship_coach persona and routing_description field"
```

---

## Task 2: Add `persona_key` to `Session`

**Files:**
- Modify: `rehearse/types.py`
- Modify: `tests/test_session_storage.py` (or `tests/test_types.py`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_types.py`:

```python
from datetime import UTC, datetime

from rehearse.types import Session


def test_session_default_persona_key():
    session = Session(created_at=datetime.now(UTC))
    assert session.persona_key == "default"


def test_session_accepts_explicit_persona_key():
    session = Session(created_at=datetime.now(UTC), persona_key="relationship_coach")
    assert session.persona_key == "relationship_coach"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_types.py -v`
Expected: failures — `Session` rejects unknown field `persona_key` (Strict config) or has no such attribute.

- [ ] **Step 3: Add the field**

In `rehearse/types.py`, in the `Session` class definition (around line 243-250), add right after `phase_timings: list[PhaseTiming] = Field(default_factory=list)`:

```python
    persona_key: str = "default"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: 2 new tests pass.

- [ ] **Step 5: Run the full suite to catch any pydantic-strict fallout**

Run: `uv run pytest -q`
Expected: green. Pre-existing tests build `Session` via kwargs without `persona_key`; the default value handles that.

- [ ] **Step 6: Commit**

```bash
git add rehearse/types.py tests/test_types.py
git commit -m "add persona_key to Session manifest with default value"
```

---

## Task 3: Implement `persona_router`

**Files:**
- Create: `rehearse/agents/persona_router.py`
- Create: `tests/test_persona_router.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_persona_router.py`:

```python
"""Verify the SMS-body persona classifier and its fallback paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rehearse.agents.persona_router import infer_persona_key
from rehearse.services.hume_configs import PERSONAS


def _fake_client(reply_text: str) -> MagicMock:
    """Build a MagicMock AsyncAnthropic client that returns one reply."""
    client = MagicMock()
    message = MagicMock()
    message.content = [MagicMock(text=reply_text)]
    client.messages.create = AsyncMock(return_value=message)
    return client


@pytest.mark.asyncio
async def test_returns_classifier_choice():
    client = _fake_client("relationship_coach")
    key = await infer_persona_key(
        "I want to rehearse breaking up with my partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "relationship_coach"


@pytest.mark.asyncio
async def test_returns_default_when_classifier_picks_default():
    client = _fake_client("default")
    key = await infer_persona_key(
        "I'm asking my boss for a raise",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_falls_back_when_no_client():
    key = await infer_persona_key(
        "anything",
        PERSONAS,
        anthropic_client=None,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_falls_back_when_classifier_raises():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))
    key = await infer_persona_key(
        "I need to talk to my partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_falls_back_when_classifier_returns_unknown_key():
    client = _fake_client("not_a_real_persona")
    key = await infer_persona_key(
        "I need to talk to my partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_skips_classifier_for_empty_body():
    client = MagicMock()
    client.messages.create = AsyncMock()
    key = await infer_persona_key(
        "",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"
    client.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_classifier_for_inbound_call_marker():
    client = MagicMock()
    client.messages.create = AsyncMock()
    key = await infer_persona_key(
        "<inbound-call>",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"
    client.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_strips_classifier_response_whitespace_and_punctuation():
    client = _fake_client("  relationship_coach.  \n")
    key = await infer_persona_key(
        "partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "relationship_coach"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_persona_router.py -v`
Expected: ImportError — `rehearse.agents.persona_router` does not exist.

- [ ] **Step 3: Implement the router**

Create `rehearse/agents/persona_router.py`:

```python
"""Classify an inbound SMS body into one of the registered persona keys.

Called once per call at SMS-trigger time, before the outbound call connects.
The chosen key is stored on the session manifest and read by `HumeEVIClient`
at connect time to pick the right Hume EVI config.

All failure modes (no client, empty body, API error, unknown key) fall back
to `"default"` — the call still happens, just with the default persona.
"""

from __future__ import annotations

from typing import Any

import structlog

from rehearse.services.hume_configs import HumePersonaConfig

log = structlog.get_logger(__name__)

_DEFAULT_KEY = "default"
_SKIP_BODIES: frozenset[str] = frozenset({"", "<inbound-call>"})
_STRIP_CHARS = " \t\n\r.,!?\"'"


async def infer_persona_key(
    sms_body: str,
    personas: dict[str, HumePersonaConfig],
    *,
    anthropic_client: Any | None,
    model: str,
    fallback: str = _DEFAULT_KEY,
) -> str:
    """Return the persona key the classifier picked for this SMS body."""
    body = sms_body.strip()
    if anthropic_client is None:
        log.info("persona_router.skip", reason="no_client")
        return fallback
    if body in _SKIP_BODIES:
        log.info("persona_router.skip", reason="empty_or_inbound_marker")
        return fallback

    prompt = _build_prompt(body, personas)
    try:
        message = await anthropic_client.messages.create(
            model=model,
            max_tokens=20,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        log.warning("persona_router.error", body=body, error=str(exc))
        return fallback

    raw = _extract_text(message)
    candidate = raw.strip(_STRIP_CHARS).lower()
    if candidate not in personas:
        log.warning(
            "persona_router.unknown_key", body=body, raw=raw, candidate=candidate
        )
        return fallback
    log.info("persona_router.picked", body=body, key=candidate)
    return candidate


def _build_prompt(sms_body: str, personas: dict[str, HumePersonaConfig]) -> str:
    """Render the classifier prompt from the persona registry."""
    options = "\n".join(
        f"- {persona.persona_key}: {persona.routing_description}"
        for persona in personas.values()
    )
    return (
        "Pick the best persona for this rehearsal request.\n\n"
        "Options:\n"
        f"{options}\n\n"
        f'Request: "{sms_body}"\n\n'
        "Reply with ONLY the persona key (one word from the list above), "
        "nothing else."
    )


def _extract_text(message: Any) -> str:
    """Pull the first text block out of an Anthropic Messages API response."""
    content = getattr(message, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_persona_router.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add rehearse/agents/persona_router.py tests/test_persona_router.py
git commit -m "add SMS-body persona classifier with default fallback"
```

---

## Task 4: `SessionOrchestrator.set_persona_key`

**Files:**
- Modify: `rehearse/session.py`
- Modify: `tests/test_session_storage.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session_storage.py`:

```python
import pytest

from rehearse.types import ConsentState


@pytest.mark.asyncio
async def test_orchestrator_set_persona_key_persists(tmp_path):
    from rehearse.session import SessionOrchestrator, TriggerEvent, utcnow
    from rehearse.storage import LocalFilesystemStore
    from rehearse.types import Session

    store = LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")
    orch = SessionOrchestrator(store=store)
    handle = await orch.start(
        TriggerEvent(
            from_number="+15555550100", body="anything", received_at=utcnow()
        )
    )

    await orch.set_persona_key(handle.session_id, "relationship_coach")

    payload = await store.read(handle.session_id, "session.json")
    session = Session.model_validate_json(payload)
    assert session.persona_key == "relationship_coach"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_session_storage.py::test_orchestrator_set_persona_key_persists -v`
Expected: AttributeError — `set_persona_key` does not exist.

- [ ] **Step 3: Add the method**

In `rehearse/session.py`, in the `SessionOrchestrator` class, add a new method (placement: near the other persistence-mutating methods like `attach_call`):

```python
    async def set_persona_key(self, session_id: str, persona_key: str) -> None:
        """Persist the chosen persona key on the session manifest."""

        def _set(session: Session) -> Session:
            session.persona_key = persona_key
            return session

        await self._store.update_session(session_id, _set)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_session_storage.py -v`
Expected: all pass (including the new test).

- [ ] **Step 5: Commit**

```bash
git add rehearse/session.py tests/test_session_storage.py
git commit -m "add SessionOrchestrator.set_persona_key"
```

---

## Task 5: `HumeEVIClient` resolves config_id from `persona_key`

**Files:**
- Modify: `rehearse/services/hume_evi.py`
- Modify: `tests/test_hume_evi.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hume_evi.py` (add imports as needed):

```python
import json

import pytest

from rehearse.services.hume_evi import HumeEVIClient


@pytest.mark.asyncio
async def test_connect_uses_mapped_config_id_for_persona(tmp_path, monkeypatch):
    from rehearse.bus import FrameBus
    from rehearse.services import hume_configs

    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps({"relationship_coach": "cfg_relationship", "default": "cfg_default"})
    )
    monkeypatch.setattr(hume_configs, "MAPPING_PATH_DEFAULT", mapping)

    captured: dict = {}

    class _FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def _connect_fn(**kwargs):
        captured.update(kwargs)
        return _FakeSocket()

    bus = FrameBus()
    client = HumeEVIClient(
        api_key="k",
        config_id="cfg_env_fallback",
        persona_key="relationship_coach",
        bus=bus,
        session_id="sess_test",
        connect_fn=_connect_fn,
    )
    async with client:
        pass

    assert captured["config_id"] == "cfg_relationship"


@pytest.mark.asyncio
async def test_connect_falls_back_to_env_config_id_when_no_mapping(tmp_path, monkeypatch):
    from rehearse.bus import FrameBus
    from rehearse.services import hume_configs

    monkeypatch.setattr(hume_configs, "MAPPING_PATH_DEFAULT", tmp_path / "missing.json")

    captured: dict = {}

    class _FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def _connect_fn(**kwargs):
        captured.update(kwargs)
        return _FakeSocket()

    bus = FrameBus()
    client = HumeEVIClient(
        api_key="k",
        config_id="cfg_env_fallback",
        persona_key="relationship_coach",
        bus=bus,
        session_id="sess_test",
        connect_fn=_connect_fn,
    )
    async with client:
        pass

    assert captured["config_id"] == "cfg_env_fallback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hume_evi.py -v -k persona`
Expected: TypeError — `HumeEVIClient.__init__` does not accept `persona_key`.

- [ ] **Step 3: Modify `HumeEVIClient`**

In `rehearse/services/hume_evi.py`, change `__init__`:

```python
    def __init__(
        self,
        *,
        api_key: str,
        config_id: str,
        bus: FrameBus,
        session_id: str,
        persona_key: str = "default",
        connect_fn: Callable[..., Any] | None = None,
        reconnect_backoff_s: float = 0.1,
    ) -> None:
        """Store connection settings and test seams for one Hume session."""
        self._api_key = api_key
        self._fallback_config_id = config_id
        self._persona_key = persona_key
        self._bus = bus
        self._session_id = session_id
        self._connect_fn = (
            connect_fn or AsyncHumeClient(api_key=api_key).empathic_voice.chat.connect
        )
        self._reconnect_backoff_s = reconnect_backoff_s
        self._stack: AsyncExitStack | None = None
        self._socket: Any = None
        self._started_at = time.monotonic()
        self._utterance_counter = 0
```

(Note the `_fallback_config_id` rename — there's no longer a single `_config_id`.)

Add the import at the top of the file:

```python
from rehearse.services.hume_configs import select_config_id
```

Change `_connect` to resolve the id at connect time:

```python
    async def _connect(self) -> None:
        """Open a fresh Hume chat websocket and store the socket object."""
        resolved_config_id = select_config_id(
            self._persona_key, fallback=self._fallback_config_id
        )
        self._stack = AsyncExitStack()
        self._socket = await self._stack.enter_async_context(
            self._connect_fn(
                config_id=resolved_config_id,
                api_key=self._api_key,
                session_settings={
                    "custom_session_id": self._session_id,
                    "audio": {
                        "channels": 1,
                        "encoding": "linear16",
                        "sample_rate": 16_000,
                    },
                },
            )
        )
```

If `swap_config` references `self._config_id`, update it to use
`self._fallback_config_id` (it's a `NotImplementedError` stub today, so the
substitution is purely cosmetic).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hume_evi.py -v`
Expected: all pass (existing tests + 2 new).

- [ ] **Step 5: Commit**

```bash
git add rehearse/services/hume_evi.py tests/test_hume_evi.py
git commit -m "resolve Hume config_id from persona_key at connect time"
```

---

## Task 6: Wire `telephony.py` end-to-end

**Files:**
- Modify: `rehearse/telephony.py`
- Modify: `rehearse/app.py` (construct shared Anthropic client; pass it down)
- Modify: `tests/test_telephony_r1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_telephony_r1.py`:

```python
@pytest.mark.asyncio
async def test_sms_persists_classified_persona_key(tmp_path, monkeypatch):
    """SMS handler runs the persona classifier and stores the result."""
    from rehearse.agents import persona_router
    from rehearse.types import Session

    async def _fake_infer(body, personas, *, anthropic_client, model, fallback="default"):
        if "partner" in body.lower():
            return "relationship_coach"
        return "default"

    monkeypatch.setattr(persona_router, "infer_persona_key", _fake_infer)

    client, _orch, store = _make_app_client(tmp_path, monkeypatch)  # see existing helper
    response = client.post(
        "/twilio/sms",
        data={"From": "+15555550100", "Body": "I want to rehearse with my partner"},
    )
    assert response.status_code == 200

    session_id = _last_session_id(store)
    payload = await store.read(session_id, "session.json")
    assert Session.model_validate_json(payload).persona_key == "relationship_coach"


@pytest.mark.asyncio
async def test_sms_handles_classifier_failure_gracefully(tmp_path, monkeypatch):
    """SMS still 200s and session uses default persona when classifier raises."""
    from rehearse.agents import persona_router
    from rehearse.types import Session

    async def _broken_infer(body, personas, *, anthropic_client, model, fallback="default"):
        raise RuntimeError("classifier down")

    monkeypatch.setattr(persona_router, "infer_persona_key", _broken_infer)

    client, _orch, store = _make_app_client(tmp_path, monkeypatch)
    response = client.post(
        "/twilio/sms",
        data={"From": "+15555550100", "Body": "anything"},
    )
    # The whole endpoint must not 500 if classification breaks — the call must still happen.
    # Catch the failure inside the handler, treat it as fallback="default".
    assert response.status_code == 200

    session_id = _last_session_id(store)
    payload = await store.read(session_id, "session.json")
    assert Session.model_validate_json(payload).persona_key == "default"
```

If `_make_app_client` and `_last_session_id` helpers don't already exist in
`tests/test_telephony_r1.py`, look at the existing tests in the file for the
pattern they use to construct a `TestClient` against the FastAPI app and to
locate the freshly-created session id (likely scanning `store` for the
single session directory). Use whatever pattern matches the existing style.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_telephony_r1.py -v -k persona`
Expected: failures — `infer_persona_key` is never called from the SMS handler today.

- [ ] **Step 3: Construct the shared Anthropic client at app wiring**

In `rehearse/app.py`, where the FastAPI app is constructed, build an
`AsyncAnthropic` client when `config.anthropic_api_key` is set, and pass it
into the telephony route registration. The exact integration depends on
your `create_app` / route-registration shape — read it first.

The general pattern:

```python
from anthropic import AsyncAnthropic

# inside create_app(config: RuntimeConfig):
anthropic_client: AsyncAnthropic | None = (
    AsyncAnthropic(api_key=config.anthropic_api_key)
    if config.anthropic_api_key
    else None
)

mount_telephony_routes(
    app,
    config=config,
    orchestrator=orchestrator,
    client=twilio_client,
    anthropic_client=anthropic_client,   # NEW
)
```

If `mount_telephony_routes` (or whatever the registration function is called)
doesn't take a kwargs bag today, add the parameter; this is the shape the
spec calls for.

- [ ] **Step 4: Wire `infer_persona_key` into the SMS handler**

In `rehearse/telephony.py::twilio_sms`, add the inference call after
`handle = await orchestrator.start(trigger)` and before the background
`_place` task:

```python
        from rehearse.agents.persona_router import infer_persona_key
        from rehearse.services.hume_configs import PERSONAS

        try:
            persona_key = await infer_persona_key(
                Body,
                PERSONAS,
                anthropic_client=anthropic_client,
                model=config.anthropic_model,
            )
        except Exception:
            log.exception("twilio.sms.persona_inference_failed", session_id=handle.session_id)
            persona_key = "default"
        await orchestrator.set_persona_key(handle.session_id, persona_key)
```

The redundant try/except wraps a function whose contract already says it
never raises — but the test
`test_sms_handles_classifier_failure_gracefully` monkeypatches the function
to raise. Belt and suspenders here is fine: the SMS endpoint must never
500 because of classification.

(Adjust `from rehearse.agents.persona_router import infer_persona_key` to
the top of the file for cleanliness if it doesn't conflict with circular
imports.)

- [ ] **Step 5: Plumb `persona_key` to `HumeEVIClient`**

In `rehearse/telephony.py`, in the websocket handler at line ~213, before
constructing `HumeEVIClient`, load the `Session` manifest and read
`persona_key`:

```python
        session_payload = await orchestrator.store.read(session_id, "session.json")
        session_obj = Session.model_validate_json(session_payload)
        persona_key = session_obj.persona_key
```

Then pass it to `HumeEVIClient`:

```python
            async with TwilioStream(ws) as twilio, HumeEVIClient(
                api_key=config.hume_api_key,
                config_id=config.hume_config_id,
                persona_key=persona_key,
                bus=bus,
                session_id=session_id,
            ) as hume:
```

(Add `from rehearse.types import Session` to the imports if not present.)

- [ ] **Step 6: Run new tests + suite**

Run: `uv run pytest tests/test_telephony_r1.py -v`
Expected: all pass (existing + 2 new).

Then: `uv run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add rehearse/telephony.py rehearse/app.py tests/test_telephony_r1.py
git commit -m "wire persona inference into SMS handler and HumeEVI connect"
```

---

## Task 7: Lint + final sweep

- [ ] **Step 1: Run ruff over all changed files**

Run:
```bash
uv run ruff check rehearse/agents/persona_router.py rehearse/services/hume_configs.py rehearse/services/hume_evi.py rehearse/session.py rehearse/telephony.py rehearse/app.py rehearse/types.py tests/test_persona_router.py tests/test_hume_configs.py tests/test_hume_evi.py tests/test_telephony_r1.py tests/test_session_storage.py tests/test_types.py
```

Expected: clean.

- [ ] **Step 2: Full suite one more time**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -u
git commit -m "lint fixes for persona routing"
```

(Skip if no fixes.)

---

## Task 8: Manual verification (operator, post-merge)

- [ ] **Step 1: Sync the new persona to Hume**

Run: `uv run rehearse-hume diff` — expect `CREATE relationship_coach (Rehearse Coach (relationship))`.
Run: `uv run rehearse-hume sync` — confirm `sessions/.hume_configs.json` now contains both `default` and `relationship_coach`.

- [ ] **Step 2: Place two test calls**

- SMS body: `"I want to practice asking my boss for a raise"` → call should connect with the default persona's voice and greeting.
- SMS body: `"I need to talk to my partner about our relationship"` → call should connect with the `relationship_coach` greeting (`"Hey, glad you called. What's going on?"`).

If the second call uses the wrong greeting, check `sessions/<id>/session.json` for the persisted `persona_key` — that tells you whether the classifier or the wiring is at fault.

---

## Self-review notes

- Spec sections covered: schema additions (Task 1, Task 2), persona registry expansion (Task 1), persona router with all 7 fallback paths (Task 3), wiring at SMS time (Task 4 + Task 6 steps 3-4), wiring at connect time (Task 5 + Task 6 step 5), test plan (Tasks 1, 3, 5, 6).
- Out-of-scope items (mid-call swap, persona-specific prompt engineering, persona-specific voices, non-LLM offline classifier, confidence thresholds) are NOT in any task.
- The migration step in the spec ("run `rehearse-hume sync` after the lift lands") is in Task 8.
- `routing_description` is deliberately excluded from `_COMPARED_FIELDS` and `_to_create_kwargs` (Task 1 step 6) — it's classifier-only, never sent to Hume.
- `set_persona_key` uses the existing `update_session` store hook so the manifest write is atomic and follows the established pattern (same as `attach_call`).
