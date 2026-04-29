"""Test deterministic intake extraction and character compilation helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from rehearse.personas import build_intake_record, character_system_prompt, compile_character

_NOW = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)


def test_build_intake_record_extracts_core_negotiation_fields() -> None:
    """Negotiation-like intake text should map to useful structured fields."""
    intake = build_intake_record(
        session_id="session-1",
        user_turns=[
            "I need help negotiating a job offer with a recruiter.",
            "I want to ask for more salary and equity but stay calm.",
        ],
        captured_at=_NOW,
    )

    assert intake.counterparty_relationship == "recruiter"
    assert "job offer" in intake.situation.lower()
    assert "financial terms" in intake.stakes.lower()
    assert intake.desired_tone == "calm"


def test_compile_character_builds_persona_prompt_from_intake() -> None:
    """Compiled personas should carry the intake context into the prompt."""
    intake = build_intake_record(
        session_id="session-2",
        user_turns=[
            "I need help with my manager.",
            "I want to ask for more support without sounding defensive.",
        ],
        captured_at=_NOW,
    )

    persona = compile_character(intake, compiled_at=_NOW)

    assert persona.relationship == "manager"
    assert "manager" in persona.personality_prompt.lower()
    assert persona.hot_buttons
    assert persona.likely_reactions
    assert character_system_prompt(persona) == persona.personality_prompt
