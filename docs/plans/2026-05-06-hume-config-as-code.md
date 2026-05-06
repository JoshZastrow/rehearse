# Hume EVI configs as code — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Declare Hume EVI configs in repo as a Python registry and reconcile them against the live workspace via a `rehearse-hume` CLI (`diff` + `sync`).

**Architecture:** A new `rehearse/services/hume_configs.py` module owns the schema (`HumePersonaConfig`), the `PERSONAS` registry, the pure `plan_sync` reconciliation function, the SDK applier, and the `select_config_id` helper. A new `rehearse/services/hume_configs_cli.py` exposes the `diff` and `sync` commands. The default persona is seeded from the live config at id `1259711b-0cec-43f4-a729-fea57e20cd32`.

**Tech Stack:** Python 3.11+, pydantic v2, hume SDK (`AsyncHumeClient`), pytest with `asyncio_mode=auto`, argparse for CLI.

**Spec:** `docs/specs/v2026-05-06-hume-config-as-code.md`

---

## File Structure

- **Create:** `rehearse/services/hume_configs.py` — schema models, `PERSONAS` registry, `plan_sync`, `apply_sync`, `select_config_id`. Pure planning + applier are in the same file because the applier is small (~40 LOC) and lives directly above its types.
- **Create:** `rehearse/services/hume_configs_cli.py` — argparse entry point exposing `diff` and `sync`. Imports `hume_configs` and `RuntimeConfig.from_env` only.
- **Create:** `tests/test_hume_configs.py` — covers `plan_sync`, schema round-trip, and `select_config_id`.
- **Create:** `tests/test_hume_configs_cli.py` — covers CLI exit codes and applier wiring with a mocked SDK client.
- **Modify:** `pyproject.toml` — add `rehearse-hume = "rehearse.services.hume_configs_cli:main"` under `[project.scripts]`.

---

## Task 1: Schema and PERSONAS registry

**Files:**
- Create: `rehearse/services/hume_configs.py` (schema + registry only in this task)
- Create: `tests/test_hume_configs.py` (round-trip test)

- [ ] **Step 1: Write the failing schema round-trip test**

Create `tests/test_hume_configs.py`:

```python
"""Verify the Hume EVI persona-config registry and reconciliation logic."""

from __future__ import annotations

from rehearse.services.hume_configs import PERSONAS, HumePersonaConfig


def test_default_persona_is_registered():
    assert "default" in PERSONAS
    persona = PERSONAS["default"]
    assert persona.persona_key == "default"
    assert persona.display_name == "Rehearse Coach (default)"
    assert persona.voice.name == "Inspiring Woman"
    assert persona.voice.provider == "HUME_AI"
    assert persona.language_model.provider == "ANTHROPIC"
    assert persona.language_model.model == "claude-sonnet-4-20250514"
    assert persona.timeouts.max_duration_secs == 300
    assert "web_search" in persona.builtin_tools


def test_persona_config_round_trip():
    persona = PERSONAS["default"]
    payload = persona.model_dump()
    reborn = HumePersonaConfig.model_validate(payload)
    assert reborn == persona
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_hume_configs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rehearse.services.hume_configs'`

- [ ] **Step 3: Implement the schema and registry**

Create `rehearse/services/hume_configs.py`:

```python
"""Declarative Hume EVI configs and reconciliation against the live workspace.

The repository is the source of truth. Each persona is a `HumePersonaConfig`
entry in `PERSONAS`. The CLI (`rehearse-hume`) compares this registry against
the Hume API and either creates a new config or appends a new version when
fields drift. The runtime later reads `select_config_id(persona_key)` to pick
the right config id for a call.

Example:
    >>> from rehearse.services.hume_configs import PERSONAS
    >>> PERSONAS["default"].voice.name
    'Inspiring Woman'
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HumeVoice(BaseModel):
    """Voice selector for a Hume EVI config: by name OR by id."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    id: str | None = None
    provider: Literal["HUME_AI", "CUSTOM_VOICE"] = "HUME_AI"


class HumeLanguageModel(BaseModel):
    """Language-model spec routed by Hume for assistant turns."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    temperature: float | None = None


class HumeEventMessage(BaseModel):
    """One Hume event-message slot (greeting, max-duration, inactivity, ...)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    text: str | None = None


class HumeTimeouts(BaseModel):
    """Per-call max duration and inactivity timeouts in seconds."""

    model_config = ConfigDict(extra="forbid")

    max_duration_secs: int = 300
    inactivity_secs: int = 122


class HumeTurnDetection(BaseModel):
    """Voice-activity / end-of-turn parameters Hume applies to user audio."""

    model_config = ConfigDict(extra="forbid")

    end_of_turn_silence_ms: int = 500
    prefix_padding_ms: int = 300
    speech_detection_threshold: float = 0.4


class HumePersonaConfig(BaseModel):
    """One declarative Hume EVI config keyed by persona."""

    model_config = ConfigDict(extra="forbid")

    persona_key: str
    display_name: str
    evi_version: str = "4-mini"
    voice: HumeVoice
    language_model: HumeLanguageModel
    prompt_text: str
    on_new_chat: HumeEventMessage | None = None
    on_resume_chat: HumeEventMessage | None = None
    on_max_duration_timeout: HumeEventMessage | None = None
    on_inactivity_timeout: HumeEventMessage | None = None
    timeouts: HumeTimeouts = HumeTimeouts()
    turn_detection: HumeTurnDetection = HumeTurnDetection()
    interruption_min_ms: int = 800
    nudges_enabled: bool = True
    nudges_interval_secs: int = 8
    builtin_tools: list[str] = []


_DEFAULT_PROMPT = (
    "You are the live voice for Rehearse, a phone-based coach that helps people prepare for a real\n"
    "conversation they're nervous about—asking for a raise, a hard talk with a partner, a difficult call with a parent, a pitch.\n\n"
    "Your job per call:\n"
    "1. INTAKE (under 90s): warmly greet the caller, ask who they're rehearsing with, what the conversation is about, and what outcome they want. Listen more than you talk. One question at a time.\n"
    "2. PRACTICE: when you have enough context, switch into the role of the person they're rehearsing with— same emotional temperature, same likely pushback. Stay in character. Let the user lead. Do not coach\n"
    "mid-scene.\n"
    "3. FEEDBACK: when the user steps out of the scene, drop the character and become the coach again.\n"
    "Reflect one specific thing they did well and one concrete line they could try next time. Cite their\n"
    "actual words.\n\n"
    "Voice and style:\n"
    "- Plain, grounded, conversational. Contractions. No corporate phrases. No filler greetings beyond a\n"
    "brief opener.\n"
    "- Short turns. Two or three sentences max unless the user explicitly asks for more.\n"
    "- Match the user's energy: if they're nervous, slow down; if they're frustrated, acknowledge it before\n"
    "redirecting.\n"
    "- Never read lists out loud. Speak like a person, not a manual.\n\n"
    "Boundaries:\n"
    "- You are not a therapist or legal advisor. If the conversation involves crisis, abuse, or self-harm,\n"
    "gently pause the rehearsal and suggest a human professional.\n"
    "- If you don't know something, say so in one sentence and move on. Do not stall, do not invent facts.\n"
    "- Stay on the rehearsal task. Politely decline unrelated requests.\n"
    "- Apply all ellicitation techniques at your disposal to understand the needs.\n"
)


PERSONAS: dict[str, HumePersonaConfig] = {
    "default": HumePersonaConfig(
        persona_key="default",
        display_name="Rehearse Coach (default)",
        evi_version="4-mini",
        voice=HumeVoice(name="Inspiring Woman", provider="HUME_AI"),
        language_model=HumeLanguageModel(
            provider="ANTHROPIC",
            model="claude-sonnet-4-20250514",
        ),
        prompt_text=_DEFAULT_PROMPT,
        on_new_chat=HumeEventMessage(enabled=True, text="Hey there, what's on the mind?"),
        on_resume_chat=HumeEventMessage(enabled=True, text="Still there?"),
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
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hume_configs.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add rehearse/services/hume_configs.py tests/test_hume_configs.py
git commit -m "add Hume persona-config schema with default registry entry"
```

---

## Task 2: `plan_sync` reconciliation logic

**Files:**
- Modify: `rehearse/services/hume_configs.py` (add SyncAction types and `plan_sync`)
- Modify: `tests/test_hume_configs.py` (add reconciliation tests)

- [ ] **Step 1: Write the failing reconciliation tests**

Append to `tests/test_hume_configs.py`:

```python
from datetime import datetime, UTC

from rehearse.services.hume_configs import (
    Create,
    NewVersion,
    NoOp,
    RemoteConfigSnapshot,
    plan_sync,
)


def _remote_from_persona(persona: HumePersonaConfig, *, config_id: str) -> RemoteConfigSnapshot:
    return RemoteConfigSnapshot(
        id=config_id,
        display_name=persona.display_name,
        evi_version=persona.evi_version,
        voice=persona.voice.model_copy(),
        language_model=persona.language_model.model_copy(),
        prompt_text=persona.prompt_text,
        on_new_chat=persona.on_new_chat.model_copy() if persona.on_new_chat else None,
        on_resume_chat=persona.on_resume_chat.model_copy() if persona.on_resume_chat else None,
        on_max_duration_timeout=(
            persona.on_max_duration_timeout.model_copy()
            if persona.on_max_duration_timeout
            else None
        ),
        on_inactivity_timeout=(
            persona.on_inactivity_timeout.model_copy()
            if persona.on_inactivity_timeout
            else None
        ),
        timeouts=persona.timeouts.model_copy(),
        turn_detection=persona.turn_detection.model_copy(),
        interruption_min_ms=persona.interruption_min_ms,
        nudges_enabled=persona.nudges_enabled,
        nudges_interval_secs=persona.nudges_interval_secs,
        builtin_tools=list(persona.builtin_tools),
    )


def test_plan_sync_creates_when_remote_empty():
    actions = plan_sync(PERSONAS, remote_configs=[])
    assert len(actions) == 1
    assert isinstance(actions[0], Create)
    assert actions[0].persona.persona_key == "default"


def test_plan_sync_noop_when_remote_matches():
    snapshot = _remote_from_persona(PERSONAS["default"], config_id="cfg_123")
    actions = plan_sync(PERSONAS, remote_configs=[snapshot])
    assert len(actions) == 1
    assert isinstance(actions[0], NoOp)
    assert actions[0].config_id == "cfg_123"


def test_plan_sync_new_version_when_voice_drifts():
    snapshot = _remote_from_persona(PERSONAS["default"], config_id="cfg_123")
    snapshot.voice.name = "Different Voice"
    actions = plan_sync(PERSONAS, remote_configs=[snapshot])
    assert len(actions) == 1
    assert isinstance(actions[0], NewVersion)
    assert actions[0].config_id == "cfg_123"
    assert any("voice" in entry for entry in actions[0].diff)


def test_plan_sync_new_version_when_prompt_drifts():
    snapshot = _remote_from_persona(PERSONAS["default"], config_id="cfg_123")
    snapshot.prompt_text = snapshot.prompt_text + "\nextra line"
    actions = plan_sync(PERSONAS, remote_configs=[snapshot])
    assert isinstance(actions[0], NewVersion)
    assert any("prompt_text" in entry for entry in actions[0].diff)


def test_plan_sync_new_version_when_timeout_drifts():
    snapshot = _remote_from_persona(PERSONAS["default"], config_id="cfg_123")
    snapshot.timeouts.max_duration_secs = 240
    actions = plan_sync(PERSONAS, remote_configs=[snapshot])
    assert isinstance(actions[0], NewVersion)
    assert any("timeouts" in entry for entry in actions[0].diff)


def test_plan_sync_ignores_unmatched_remote_configs():
    other = _remote_from_persona(PERSONAS["default"], config_id="cfg_999")
    other.display_name = "Some Unrelated Config"
    actions = plan_sync(PERSONAS, remote_configs=[other])
    assert len(actions) == 1
    assert isinstance(actions[0], Create)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hume_configs.py -v`
Expected: ImportError on `Create`, `NewVersion`, `NoOp`, `RemoteConfigSnapshot`, `plan_sync`.

- [ ] **Step 3: Add the reconciliation logic**

Append to `rehearse/services/hume_configs.py`:

```python
from dataclasses import dataclass, field


class RemoteConfigSnapshot(BaseModel):
    """Hume-side view of one config used as input to `plan_sync`.

    This is the subset of fields we care about for reconciliation. Fields
    Hume manages (`created_on`, `version`, voice `reference_tokens`, prompt
    `id`/`version`) are deliberately excluded.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    evi_version: str
    voice: HumeVoice
    language_model: HumeLanguageModel
    prompt_text: str
    on_new_chat: HumeEventMessage | None = None
    on_resume_chat: HumeEventMessage | None = None
    on_max_duration_timeout: HumeEventMessage | None = None
    on_inactivity_timeout: HumeEventMessage | None = None
    timeouts: HumeTimeouts
    turn_detection: HumeTurnDetection
    interruption_min_ms: int
    nudges_enabled: bool
    nudges_interval_secs: int
    builtin_tools: list[str]


@dataclass(frozen=True)
class Create:
    """Action: create a new Hume config for one persona."""

    persona: HumePersonaConfig


@dataclass(frozen=True)
class NewVersion:
    """Action: append a new version to an existing matched Hume config."""

    persona: HumePersonaConfig
    config_id: str
    diff: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NoOp:
    """Action: declared persona already matches the live config exactly."""

    persona: HumePersonaConfig
    config_id: str


SyncAction = Create | NewVersion | NoOp


_COMPARED_FIELDS: tuple[str, ...] = (
    "evi_version",
    "voice",
    "language_model",
    "prompt_text",
    "on_new_chat",
    "on_resume_chat",
    "on_max_duration_timeout",
    "on_inactivity_timeout",
    "timeouts",
    "turn_detection",
    "interruption_min_ms",
    "nudges_enabled",
    "nudges_interval_secs",
    "builtin_tools",
)


def plan_sync(
    personas: dict[str, HumePersonaConfig],
    *,
    remote_configs: list[RemoteConfigSnapshot],
) -> list[SyncAction]:
    """Return the list of sync actions needed to align Hume with `personas`.

    Match key: `display_name` (exact, case-sensitive). If no remote config
    matches a persona, emit `Create`. If one matches but compared fields
    differ, emit `NewVersion` with a list of differing field names. If
    everything matches, emit `NoOp`.
    """
    by_name = {snap.display_name: snap for snap in remote_configs}
    actions: list[SyncAction] = []
    for persona in personas.values():
        snap = by_name.get(persona.display_name)
        if snap is None:
            actions.append(Create(persona=persona))
            continue
        diff = _diff_fields(persona, snap)
        if diff:
            actions.append(NewVersion(persona=persona, config_id=snap.id, diff=diff))
        else:
            actions.append(NoOp(persona=persona, config_id=snap.id))
    return actions


def _diff_fields(persona: HumePersonaConfig, snap: RemoteConfigSnapshot) -> list[str]:
    """Return the list of compared field names that differ between two configs."""
    diffs: list[str] = []
    for name in _COMPARED_FIELDS:
        if getattr(persona, name) != getattr(snap, name):
            diffs.append(name)
    return diffs
```

(Move the `from dataclasses import dataclass, field` import to the top of the file once you save, so all imports stay grouped.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hume_configs.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add rehearse/services/hume_configs.py tests/test_hume_configs.py
git commit -m "add plan_sync reconciliation for Hume persona configs"
```

---

## Task 3: SDK applier and `select_config_id`

**Files:**
- Modify: `rehearse/services/hume_configs.py` — add `apply_sync`, `fetch_remote_configs`, `select_config_id`, `MAPPING_PATH_DEFAULT`
- Modify: `tests/test_hume_configs.py` — add `select_config_id` tests
- Create: `tests/test_hume_configs_cli.py` — CLI applier wiring tests (added in Task 4 also; here we add applier-level tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hume_configs.py`:

```python
import json
from pathlib import Path

from rehearse.services.hume_configs import select_config_id


def test_select_config_id_reads_mapping(tmp_path: Path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"default": "cfg_abc", "synced_at": "2026-05-06T00:00:00Z"}))
    assert select_config_id("default", mapping_path=path, fallback="env_id") == "cfg_abc"


def test_select_config_id_falls_back_when_file_missing(tmp_path: Path):
    path = tmp_path / "missing.json"
    assert select_config_id("default", mapping_path=path, fallback="env_id") == "env_id"


def test_select_config_id_falls_back_when_key_missing(tmp_path: Path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"other": "cfg_xyz"}))
    assert select_config_id("default", mapping_path=path, fallback="env_id") == "env_id"
```

Create `tests/test_hume_configs_cli.py`:

```python
"""Verify the rehearse-hume CLI's applier wiring against a mocked Hume client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rehearse.services.hume_configs import (
    Create,
    HumePersonaConfig,
    NewVersion,
    PERSONAS,
    apply_sync,
)


@pytest.mark.asyncio
async def test_apply_sync_creates_new_config(tmp_path: Path):
    fake_client = MagicMock()
    fake_client.empathic_voice.configs.create_config = AsyncMock(
        return_value=MagicMock(id="cfg_new")
    )
    fake_client.empathic_voice.configs.create_config_version = AsyncMock()

    actions = [Create(persona=PERSONAS["default"])]
    mapping_path = tmp_path / "mapping.json"

    mapping = await apply_sync(fake_client, actions, mapping_path=mapping_path)

    assert mapping["default"] == "cfg_new"
    assert fake_client.empathic_voice.configs.create_config.await_count == 1
    assert fake_client.empathic_voice.configs.create_config_version.await_count == 0
    on_disk = json.loads(mapping_path.read_text())
    assert on_disk["default"] == "cfg_new"
    assert "synced_at" in on_disk


@pytest.mark.asyncio
async def test_apply_sync_appends_new_version(tmp_path: Path):
    fake_client = MagicMock()
    fake_client.empathic_voice.configs.create_config = AsyncMock()
    fake_client.empathic_voice.configs.create_config_version = AsyncMock(
        return_value=MagicMock(id="cfg_existing")
    )

    actions = [
        NewVersion(
            persona=PERSONAS["default"],
            config_id="cfg_existing",
            diff=["voice", "prompt_text"],
        )
    ]
    mapping_path = tmp_path / "mapping.json"

    mapping = await apply_sync(fake_client, actions, mapping_path=mapping_path)

    assert mapping["default"] == "cfg_existing"
    assert fake_client.empathic_voice.configs.create_config.await_count == 0
    assert fake_client.empathic_voice.configs.create_config_version.await_count == 1
    call_kwargs = fake_client.empathic_voice.configs.create_config_version.await_args.kwargs
    assert call_kwargs["id"] == "cfg_existing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hume_configs.py tests/test_hume_configs_cli.py -v`
Expected: ImportError on `apply_sync` and `select_config_id`.

- [ ] **Step 3: Implement the applier and helper**

Append to `rehearse/services/hume_configs.py`:

```python
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hume.empathic_voice.types.posted_builtin_tool import PostedBuiltinTool
from hume.empathic_voice.types.posted_config_prompt_spec import PostedConfigPromptSpec
from hume.empathic_voice.types.posted_event_message_spec import PostedEventMessageSpec
from hume.empathic_voice.types.posted_event_message_specs import PostedEventMessageSpecs
from hume.empathic_voice.types.posted_language_model import PostedLanguageModel
from hume.empathic_voice.types.posted_timeout_specs import PostedTimeoutSpecs
from hume.empathic_voice.types.posted_timeout_spec import PostedTimeoutSpec
from hume.empathic_voice.types.posted_turn_detection_spec import PostedTurnDetectionSpec
from hume.empathic_voice.types.voice_id import VoiceId
from hume.empathic_voice.types.voice_name import VoiceName


MAPPING_PATH_DEFAULT = Path("sessions/.hume_configs.json")


def select_config_id(
    persona_key: str,
    *,
    mapping_path: Path = MAPPING_PATH_DEFAULT,
    fallback: str,
) -> str:
    """Return the Hume config id for `persona_key`, falling back if unknown.

    Reads the mapping file written by `apply_sync`. If the file is missing or
    the persona key is absent, returns `fallback` (typically `RuntimeConfig.hume_config_id`).
    """
    if not mapping_path.exists():
        return fallback
    try:
        data = json.loads(mapping_path.read_text())
    except json.JSONDecodeError:
        return fallback
    return data.get(persona_key, fallback)


async def apply_sync(
    client: Any,
    actions: list[SyncAction],
    *,
    mapping_path: Path = MAPPING_PATH_DEFAULT,
) -> dict[str, str]:
    """Execute Create / NewVersion actions against Hume and write the mapping.

    Returns the persona_key -> config_id mapping that was persisted.
    """
    mapping: dict[str, str] = {}
    for action in actions:
        if isinstance(action, Create):
            kwargs = _to_create_kwargs(action.persona)
            response = await client.empathic_voice.configs.create_config(**kwargs)
            mapping[action.persona.persona_key] = response.id
        elif isinstance(action, NewVersion):
            kwargs = _to_version_kwargs(action.persona)
            await client.empathic_voice.configs.create_config_version(
                id=action.config_id, **kwargs
            )
            mapping[action.persona.persona_key] = action.config_id
        elif isinstance(action, NoOp):
            mapping[action.persona.persona_key] = action.config_id
    payload = {**mapping, "synced_at": datetime.now(UTC).isoformat()}
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(payload, indent=2))
    return mapping


def _to_create_kwargs(persona: HumePersonaConfig) -> dict[str, Any]:
    """Render a persona into kwargs for `configs.create_config`."""
    return {
        "evi_version": persona.evi_version,
        "name": persona.display_name,
        "voice": _render_voice(persona.voice),
        "language_model": _render_language_model(persona.language_model),
        "prompt": PostedConfigPromptSpec(text=persona.prompt_text),
        "event_messages": _render_event_messages(persona),
        "timeouts": _render_timeouts(persona.timeouts),
        "turn_detection": _render_turn_detection(persona.turn_detection),
        "builtin_tools": _render_builtin_tools(persona.builtin_tools),
    }


def _to_version_kwargs(persona: HumePersonaConfig) -> dict[str, Any]:
    """Render a persona into kwargs for `configs.create_config_version`."""
    kwargs = _to_create_kwargs(persona)
    kwargs.pop("name", None)
    return kwargs


def _render_voice(voice: HumeVoice) -> VoiceName | VoiceId:
    if voice.id is not None:
        return VoiceId(id=voice.id, provider=voice.provider)
    if voice.name is None:
        raise ValueError("HumeVoice requires either id or name")
    return VoiceName(name=voice.name, provider=voice.provider)


def _render_language_model(lm: HumeLanguageModel) -> PostedLanguageModel:
    return PostedLanguageModel(
        model_provider=lm.provider,
        model_resource=lm.model,
        temperature=lm.temperature,
    )


def _render_event_messages(persona: HumePersonaConfig) -> PostedEventMessageSpecs:
    def spec(msg: HumeEventMessage | None) -> PostedEventMessageSpec | None:
        if msg is None:
            return None
        return PostedEventMessageSpec(enabled=msg.enabled, text=msg.text)

    return PostedEventMessageSpecs(
        on_new_chat=spec(persona.on_new_chat),
        on_inactivity_timeout=spec(persona.on_inactivity_timeout),
        on_max_duration_timeout=spec(persona.on_max_duration_timeout),
    )


def _render_timeouts(t: HumeTimeouts) -> PostedTimeoutSpecs:
    return PostedTimeoutSpecs(
        inactivity=PostedTimeoutSpec(enabled=True, duration_secs=t.inactivity_secs),
        max_duration=PostedTimeoutSpec(enabled=True, duration_secs=t.max_duration_secs),
    )


def _render_turn_detection(td: HumeTurnDetection) -> PostedTurnDetectionSpec:
    return PostedTurnDetectionSpec(
        type="FIXED",
        end_of_turn_silence_ms=td.end_of_turn_silence_ms,
        prefix_padding_ms=td.prefix_padding_ms,
        speech_detection_threshold=td.speech_detection_threshold,
    )


def _render_builtin_tools(names: list[str]) -> list[PostedBuiltinTool]:
    return [PostedBuiltinTool(name=name) for name in names]
```

If any of the `Posted*` import paths fail (Hume SDK organization can change), inspect the installed SDK with:

```
uv run python -c "import hume.empathic_voice.types as t; import os; print(sorted(os.listdir(os.path.dirname(t.__file__))))"
```

and substitute the correct module name. If `PostedTurnDetectionSpec` lacks a `type` field in your SDK version, drop it; the SDK accepts the spec without it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hume_configs.py tests/test_hume_configs_cli.py -v`
Expected: all pass (3 select_config_id + 2 cli applier + previous 8 = 13).

- [ ] **Step 5: Commit**

```bash
git add rehearse/services/hume_configs.py tests/test_hume_configs.py tests/test_hume_configs_cli.py
git commit -m "add Hume config applier and select_config_id helper"
```

---

## Task 4: CLI entry point (`rehearse-hume`)

**Files:**
- Create: `rehearse/services/hume_configs_cli.py`
- Modify: `rehearse/services/hume_configs.py` — add `fetch_remote_configs`
- Modify: `pyproject.toml` — register `rehearse-hume` script
- Modify: `tests/test_hume_configs_cli.py` — add CLI exit-code tests

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/test_hume_configs_cli.py`:

```python
from rehearse.services.hume_configs_cli import run_diff, run_sync


@pytest.mark.asyncio
async def test_run_diff_returns_zero_when_in_sync(tmp_path: Path, monkeypatch, capsys):
    from rehearse.services import hume_configs

    async def _fake_fetch(client):
        snap = hume_configs.RemoteConfigSnapshot(
            id="cfg_match",
            display_name=PERSONAS["default"].display_name,
            evi_version=PERSONAS["default"].evi_version,
            voice=PERSONAS["default"].voice.model_copy(),
            language_model=PERSONAS["default"].language_model.model_copy(),
            prompt_text=PERSONAS["default"].prompt_text,
            on_new_chat=PERSONAS["default"].on_new_chat.model_copy(),
            on_resume_chat=PERSONAS["default"].on_resume_chat.model_copy(),
            on_max_duration_timeout=PERSONAS["default"].on_max_duration_timeout.model_copy(),
            on_inactivity_timeout=PERSONAS["default"].on_inactivity_timeout.model_copy(),
            timeouts=PERSONAS["default"].timeouts.model_copy(),
            turn_detection=PERSONAS["default"].turn_detection.model_copy(),
            interruption_min_ms=PERSONAS["default"].interruption_min_ms,
            nudges_enabled=PERSONAS["default"].nudges_enabled,
            nudges_interval_secs=PERSONAS["default"].nudges_interval_secs,
            builtin_tools=list(PERSONAS["default"].builtin_tools),
        )
        return [snap]

    monkeypatch.setattr(hume_configs, "fetch_remote_configs", _fake_fetch)
    fake_client = MagicMock()
    exit_code = await run_diff(fake_client)
    assert exit_code == 0


@pytest.mark.asyncio
async def test_run_diff_returns_one_when_drifted(monkeypatch, capsys):
    from rehearse.services import hume_configs

    async def _fake_fetch(client):
        return []

    monkeypatch.setattr(hume_configs, "fetch_remote_configs", _fake_fetch)
    fake_client = MagicMock()
    exit_code = await run_diff(fake_client)
    assert exit_code == 1


@pytest.mark.asyncio
async def test_run_sync_writes_mapping(tmp_path: Path, monkeypatch):
    from rehearse.services import hume_configs

    async def _fake_fetch(client):
        return []

    monkeypatch.setattr(hume_configs, "fetch_remote_configs", _fake_fetch)
    fake_client = MagicMock()
    fake_client.empathic_voice.configs.create_config = AsyncMock(
        return_value=MagicMock(id="cfg_new")
    )
    fake_client.empathic_voice.configs.create_config_version = AsyncMock()

    mapping_path = tmp_path / "mapping.json"
    exit_code = await run_sync(fake_client, mapping_path=mapping_path)
    assert exit_code == 0
    on_disk = json.loads(mapping_path.read_text())
    assert on_disk["default"] == "cfg_new"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hume_configs_cli.py -v`
Expected: ImportError on `run_diff`, `run_sync`, `fetch_remote_configs`.

- [ ] **Step 3: Add `fetch_remote_configs` to `rehearse/services/hume_configs.py`**

Append:

```python
async def fetch_remote_configs(client: Any) -> list[RemoteConfigSnapshot]:
    """Page through the workspace and return the latest version of each config.

    Hume's `list_configs` returns one entry per config-name (latest version).
    """
    snapshots: list[RemoteConfigSnapshot] = []
    pager = await client.empathic_voice.configs.list_configs()
    async for cfg in pager:
        snapshots.append(_snapshot_from_remote(cfg))
    return snapshots


def _snapshot_from_remote(cfg: Any) -> RemoteConfigSnapshot:
    """Convert one Hume `ReturnConfig` into a comparison snapshot."""
    voice = HumeVoice(
        name=getattr(cfg.voice, "name", None) if cfg.voice else None,
        id=getattr(cfg.voice, "id", None) if cfg.voice else None,
        provider=getattr(cfg.voice, "provider", "HUME_AI") if cfg.voice else "HUME_AI",
    )
    lm = HumeLanguageModel(
        provider=getattr(cfg.language_model, "model_provider", "ANTHROPIC"),
        model=getattr(cfg.language_model, "model_resource", None),
        temperature=getattr(cfg.language_model, "temperature", None),
    )
    em = cfg.event_messages
    prompt_text = getattr(cfg.prompt, "text", "") if cfg.prompt else ""
    timeouts = HumeTimeouts(
        max_duration_secs=cfg.timeouts.max_duration.duration_secs,
        inactivity_secs=cfg.timeouts.inactivity.duration_secs,
    )
    td = cfg.turn_detection
    return RemoteConfigSnapshot(
        id=cfg.id,
        display_name=cfg.name,
        evi_version=cfg.evi_version,
        voice=voice,
        language_model=lm,
        prompt_text=prompt_text,
        on_new_chat=_event(em.on_new_chat) if em else None,
        on_resume_chat=_event(getattr(em, "on_resume_chat", None)) if em else None,
        on_max_duration_timeout=_event(em.on_max_duration_timeout) if em else None,
        on_inactivity_timeout=_event(em.on_inactivity_timeout) if em else None,
        timeouts=timeouts,
        turn_detection=HumeTurnDetection(
            end_of_turn_silence_ms=td.end_of_turn_silence_ms,
            prefix_padding_ms=td.prefix_padding_ms,
            speech_detection_threshold=td.speech_detection_threshold,
        ),
        interruption_min_ms=cfg.interruption.min_interruption_ms,
        nudges_enabled=getattr(cfg.nudges, "enabled", False) if cfg.nudges else False,
        nudges_interval_secs=getattr(cfg.nudges, "interval_secs", 0) if cfg.nudges else 0,
        builtin_tools=[t.name for t in (cfg.builtin_tools or [])],
    )


def _event(spec: Any) -> HumeEventMessage | None:
    if spec is None:
        return None
    return HumeEventMessage(enabled=spec.enabled, text=spec.text)
```

- [ ] **Step 4: Create the CLI module**

Create `rehearse/services/hume_configs_cli.py`:

```python
"""CLI entry point for managing Hume EVI configs declaratively.

Examples:
    # Show what sync would change without writing anything.
    rehearse-hume diff

    # Reconcile the live workspace against PERSONAS, then write
    # `sessions/.hume_configs.json` with persona_key -> config_id.
    rehearse-hume sync

The first time you run `sync`, Hume will likely contain a manually-created
config under a different display_name (e.g. with a timestamp). To avoid
creating a duplicate, rename that config in the Hume console to match the
declared `display_name` (e.g. `Rehearse Coach (default)`) before running
sync. After that, every change to `PERSONAS` is one `sync` away from being
live.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from hume.client import AsyncHumeClient

from rehearse.config import RuntimeConfig
from rehearse.services.hume_configs import (
    Create,
    MAPPING_PATH_DEFAULT,
    NewVersion,
    NoOp,
    PERSONAS,
    apply_sync,
    fetch_remote_configs,
    plan_sync,
)


async def run_diff(client) -> int:
    """Print planned actions; exit 1 if any Create/NewVersion is needed."""
    remote = await fetch_remote_configs(client)
    actions = plan_sync(PERSONAS, remote_configs=remote)
    drift = False
    for action in actions:
        if isinstance(action, Create):
            print(f"CREATE {action.persona.persona_key} ({action.persona.display_name})")
            drift = True
        elif isinstance(action, NewVersion):
            print(
                f"NEW_VERSION {action.persona.persona_key} "
                f"({action.config_id}) diff={action.diff}"
            )
            drift = True
        elif isinstance(action, NoOp):
            print(f"NOOP {action.persona.persona_key} ({action.config_id})")
    return 1 if drift else 0


async def run_sync(client, *, mapping_path: Path = MAPPING_PATH_DEFAULT) -> int:
    """Execute pending actions and write the persona->config_id mapping."""
    remote = await fetch_remote_configs(client)
    actions = plan_sync(PERSONAS, remote_configs=remote)
    mapping = await apply_sync(client, actions, mapping_path=mapping_path)
    print(f"Wrote {mapping_path} with {len(mapping)} persona(s).")
    return 0


def main() -> None:
    """Argparse entry point for `rehearse-hume`."""
    parser = argparse.ArgumentParser(prog="rehearse-hume")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("diff", help="Show planned reconcile actions; exit 1 if drifted.")
    sub.add_parser("sync", help="Apply reconcile actions and write the id mapping.")
    args = parser.parse_args()

    cfg = RuntimeConfig.from_env()
    client = AsyncHumeClient(api_key=cfg.hume_api_key)

    if args.command == "diff":
        sys.exit(asyncio.run(run_diff(client)))
    if args.command == "sync":
        sys.exit(asyncio.run(run_sync(client)))
```

- [ ] **Step 5: Register the script in `pyproject.toml`**

In `pyproject.toml`, change:

```toml
[project.scripts]
rehearse-eval = "rehearse.eval.cli:main"
```

to:

```toml
[project.scripts]
rehearse-eval = "rehearse.eval.cli:main"
rehearse-hume = "rehearse.services.hume_configs_cli:main"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_hume_configs.py tests/test_hume_configs_cli.py -v`
Expected: all pass.

- [ ] **Step 7: Verify CLI is discoverable**

Run: `uv sync && uv run rehearse-hume --help`
Expected: shows the `diff` and `sync` subcommands.

- [ ] **Step 8: Commit**

```bash
git add rehearse/services/hume_configs.py rehearse/services/hume_configs_cli.py tests/test_hume_configs_cli.py pyproject.toml
git commit -m "add rehearse-hume CLI for declarative Hume config management"
```

---

## Task 5: Full suite + lint

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: green. If any pre-existing test fails because the schema or import order changed, fix it without weakening behavior.

- [ ] **Step 2: Run the linter**

Run: `uv run ruff check rehearse/services/hume_configs.py rehearse/services/hume_configs_cli.py tests/test_hume_configs.py tests/test_hume_configs_cli.py`
Expected: clean.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -u
git commit -m "lint fixes for hume config CLI"
```

(Skip if there were no fixes.)

---

## Task 6: Manual verification against the live workspace

Not automated. Done by the operator before relying on the mapping file.

- [ ] **Step 1: First-run migration**

Run: `uv run rehearse-hume diff`. Expected output: `CREATE default (Rehearse Coach (default))`.

In the Hume console, rename the existing config (currently
`"Your smart companion (5/1/2026, 03:23:55 PM)"`) to
`"Rehearse Coach (default)"`.

Re-run: `uv run rehearse-hume diff`. Expected: `NOOP default (cfg_id)` if no
fields drifted, or `NEW_VERSION default (cfg_id) diff=[...]` if any did.

- [ ] **Step 2: Sync**

Run: `uv run rehearse-hume sync`. Confirm `sessions/.hume_configs.json`
exists and contains the default persona's id.

- [ ] **Step 3: Place a test call**

Verify the call still works end-to-end. Nothing in the live runtime path
changed, so this is a smoke test that the synced config didn't accidentally
break behavior.

---

## Self-review notes

- Spec sections covered: schema (Task 1), default registry entry (Task 1), `plan_sync` reconciliation (Task 2), applier with create/new-version (Task 3), `select_config_id` (Task 3), CLI with `diff`/`sync` (Task 4), mapping file path + format (Task 3 + Task 4), migration via rename (Task 4 docstring + Task 6), test plan (Tasks 1–4).
- Spec out-of-scope items (`pull`, `list`, runtime wiring, persona inference, live integration test) are deliberately not in any task.
- `select_config_id`'s `fallback` is a required keyword. Callers (the runtime, eventually) pass `RuntimeConfig.hume_config_id`. This keeps the function pure — no env reads inside.
- The `_to_create_kwargs` / `_to_version_kwargs` split exists because `create_config` requires `name` while `create_config_version` rejects it.
- Type-name caveat: SDK module paths under `hume.empathic_voice.types` may shift between SDK versions. Task 3 includes a fallback inspection command if imports fail.
