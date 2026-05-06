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
    assert persona.language_model.provider == "ANTHROPIC"
    assert persona.language_model.model == "claude-sonnet-4-20250514"
    assert persona.timeouts.max_duration_secs == 300
    assert "web_search" in persona.builtin_tools


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
