"""Verify the Hume EVI persona-config registry and reconciliation logic."""

from __future__ import annotations

import json
from pathlib import Path

from rehearse.services.hume_configs import (
    PERSONAS,
    Create,
    HumePersonaConfig,
    NewVersion,
    NoOp,
    RemoteConfigSnapshot,
    plan_sync,
    select_config_id,
)


def test_default_persona_is_registered():
    assert "default" in PERSONAS
    persona = PERSONAS["default"]
    assert persona.persona_key == "default"
    assert persona.display_name == "Rehearse Coach (default)"
    assert persona.voice.name == "Inspiring Woman"
    assert persona.voice.provider == "HUME_AI"
    assert persona.language_model.provider == "CUSTOM_LANGUAGE_MODEL"
    assert persona.language_model.model is not None
    assert persona.language_model.model.endswith("/chat/completions")
    assert persona.timeouts.max_duration_secs == 300
    assert "web_search" in persona.builtin_tools


def test_all_personas_route_through_custom_language_model() -> None:
    """Every shipped persona must route Hume's per-turn LLM through our CLM
    webhook — otherwise Hume calls Anthropic directly with one static prompt
    and our phase-aware coach/character/feedback_coach swap never runs.
    """
    for key, persona in PERSONAS.items():
        assert persona.language_model.provider == "CUSTOM_LANGUAGE_MODEL", (
            f"persona {key!r} has provider={persona.language_model.provider!r}; "
            "must be CUSTOM_LANGUAGE_MODEL for per-phase prompt swap"
        )


def test_persona_config_round_trip():
    persona = PERSONAS["default"]
    payload = persona.model_dump()
    reborn = HumePersonaConfig.model_validate(payload)
    assert reborn == persona


def _remote_from_persona(persona: HumePersonaConfig, *, config_id: str) -> RemoteConfigSnapshot:
    return RemoteConfigSnapshot(
        id=config_id,
        display_name=persona.display_name,
        evi_version=persona.evi_version,
        voice=persona.voice.model_copy(),
        language_model=persona.language_model.model_copy(),
        prompt_text=persona.prompt_text,
        on_new_chat=persona.on_new_chat.model_copy() if persona.on_new_chat else None,
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
    assert len(actions) == len(PERSONAS)
    assert all(isinstance(a, Create) for a in actions)
    assert any(a.persona.persona_key == "default" for a in actions)


def test_plan_sync_noop_when_remote_matches():
    snapshots = [_remote_from_persona(p, config_id=f"cfg_{k}") for k, p in PERSONAS.items()]
    actions = plan_sync(PERSONAS, remote_configs=snapshots)
    assert len(actions) == len(PERSONAS)
    assert all(isinstance(a, NoOp) for a in actions)
    default_action = next(a for a in actions if a.persona.persona_key == "default")
    assert default_action.config_id == "cfg_default"


def test_plan_sync_new_version_when_voice_drifts():
    snapshot = _remote_from_persona(PERSONAS["default"], config_id="cfg_123")
    snapshot.voice.name = "Different Voice"
    actions = plan_sync(PERSONAS, remote_configs=[snapshot])
    default_action = next(a for a in actions if a.persona.persona_key == "default")
    assert isinstance(default_action, NewVersion)
    assert default_action.config_id == "cfg_123"
    assert any("voice" in entry for entry in default_action.diff)


def test_plan_sync_ignores_prompt_drift_for_custom_lm_personas():
    """Custom-LM personas don't post prompt_text to Hume, so the field always
    looks empty on the remote side. Comparing it would create a permanent
    drift loop. Verify the diff skips prompt_text for those personas.
    """
    persona = PERSONAS["default"]
    assert persona.language_model.provider == "CUSTOM_LANGUAGE_MODEL"
    snapshot = _remote_from_persona(persona, config_id="cfg_123")
    snapshot.prompt_text = ""  # what Hume actually returns for custom-LM configs
    actions = plan_sync(PERSONAS, remote_configs=[snapshot])
    default_action = next(a for a in actions if a.persona.persona_key == "default")
    assert isinstance(default_action, NoOp)


def test_plan_sync_new_version_when_prompt_drifts_for_native_lm():
    """Personas that DO post prompt_text (any non-custom provider) must still
    surface prompt drift as NEW_VERSION.
    """
    persona = PERSONAS["default"].model_copy(
        update={
            "language_model": PERSONAS["default"].language_model.model_copy(
                update={"provider": "ANTHROPIC", "model": "claude-sonnet-4"}
            )
        }
    )
    snapshot = _remote_from_persona(persona, config_id="cfg_123")
    snapshot.prompt_text = snapshot.prompt_text + "\nextra line"
    actions = plan_sync({"default": persona}, remote_configs=[snapshot])
    default_action = next(a for a in actions if a.persona.persona_key == "default")
    assert isinstance(default_action, NewVersion)
    assert any("prompt_text" in entry for entry in default_action.diff)


def test_plan_sync_new_version_when_timeout_drifts():
    snapshot = _remote_from_persona(PERSONAS["default"], config_id="cfg_123")
    snapshot.timeouts.max_duration_secs = 240
    actions = plan_sync(PERSONAS, remote_configs=[snapshot])
    default_action = next(a for a in actions if a.persona.persona_key == "default")
    assert isinstance(default_action, NewVersion)
    assert any("timeouts" in entry for entry in default_action.diff)


def test_plan_sync_ignores_unmatched_remote_configs():
    other = _remote_from_persona(PERSONAS["default"], config_id="cfg_999")
    other.display_name = "Some Unrelated Config"
    actions = plan_sync(PERSONAS, remote_configs=[other])
    assert all(isinstance(a, Create) for a in actions)


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
    desc = persona.routing_description.lower()
    assert "general" in desc or "any" in desc


def test_relationship_prompt_extends_default_with_framing():
    persona = PERSONAS["relationship_coach"]
    default_prompt = PERSONAS["default"].prompt_text
    assert default_prompt in persona.prompt_text
    assert "relationship coach" in persona.prompt_text.lower()
