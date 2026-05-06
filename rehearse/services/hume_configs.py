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

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from hume.empathic_voice.types.posted_builtin_tool import PostedBuiltinTool
from hume.empathic_voice.types.posted_config_prompt_spec import PostedConfigPromptSpec
from hume.empathic_voice.types.posted_event_message_spec import PostedEventMessageSpec
from hume.empathic_voice.types.posted_event_message_specs import PostedEventMessageSpecs
from hume.empathic_voice.types.posted_language_model import PostedLanguageModel
from hume.empathic_voice.types.posted_timeout_specs import PostedTimeoutSpecs
from hume.empathic_voice.types.posted_timeout_specs_inactivity import (
    PostedTimeoutSpecsInactivity,
)
from hume.empathic_voice.types.posted_timeout_specs_max_duration import (
    PostedTimeoutSpecsMaxDuration,
)
from hume.empathic_voice.types.posted_turn_detection_spec import PostedTurnDetectionSpec
from hume.empathic_voice.types.voice_id import VoiceId
from hume.empathic_voice.types.voice_name import VoiceName
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


_DEFAULT_PROMPT = (  # noqa: E501
    "You are the live voice for Rehearse, a phone-based coach that helps people prepare for a real\n"  # noqa: E501
    "conversation they're nervous about—asking for a raise, a hard talk with a partner, a difficult call with a parent, a pitch.\n\n"  # noqa: E501
    "Your job per call:\n"
    "1. INTAKE (under 90s): warmly greet the caller, ask who they're rehearsing with, what the conversation is about, and what outcome they want. Listen more than you talk. One question at a time.\n"  # noqa: E501
    "2. PRACTICE: when you have enough context, switch into the role of the person they're rehearsing with— same emotional temperature, same likely pushback. Stay in character. Let the user lead. Do not coach\n"  # noqa: E501
    "mid-scene.\n"
    "3. FEEDBACK: when the user steps out of the scene, drop the character and become the coach again.\n"  # noqa: E501
    "Reflect one specific thing they did well and one concrete line they could try next time. Cite their\n"  # noqa: E501
    "actual words.\n\n"
    "Voice and style:\n"
    "- Plain, grounded, conversational. Contractions. No corporate phrases. No filler greetings beyond a\n"  # noqa: E501
    "brief opener.\n"
    "- Short turns. Two or three sentences max unless the user explicitly asks for more.\n"
    "- Match the user's energy: if they're nervous, slow down; if they're frustrated, acknowledge it before\n"  # noqa: E501
    "redirecting.\n"
    "- Never read lists out loud. Speak like a person, not a manual.\n\n"
    "Boundaries:\n"
    "- You are not a therapist or legal advisor. If the conversation involves crisis, abuse, or self-harm,\n"  # noqa: E501
    "gently pause the rehearsal and suggest a human professional.\n"
    "- If you don't know something, say so in one sentence and move on. Do not stall, do not invent facts.\n"  # noqa: E501
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
        inactivity=PostedTimeoutSpecsInactivity(enabled=True, duration_secs=t.inactivity_secs),
        max_duration=PostedTimeoutSpecsMaxDuration(enabled=True, duration_secs=t.max_duration_secs),
    )


def _render_turn_detection(td: HumeTurnDetection) -> PostedTurnDetectionSpec:
    # Note: PostedTurnDetectionSpec in this SDK version has no `type` field.
    return PostedTurnDetectionSpec(
        end_of_turn_silence_ms=td.end_of_turn_silence_ms,
        prefix_padding_ms=td.prefix_padding_ms,
        speech_detection_threshold=td.speech_detection_threshold,
    )


def _render_builtin_tools(names: list[str]) -> list[PostedBuiltinTool]:
    return [PostedBuiltinTool(name=name) for name in names]
