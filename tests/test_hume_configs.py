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
