"""Unit tests for ConversationBackend protocol, ManagedBackend, and factory."""

from __future__ import annotations

import pytest
from rehearse.types import Speaker


def test_persona_spec_is_constructable():
    from rehearse.backends.base import PersonaSpec
    spec = PersonaSpec(
        name="Alex",
        gender="male",
        system_prompt="You are Alex.",
        voice_ref="voice-123",
    )
    assert spec["name"] == "Alex"
    assert spec["gender"] == "male"
    assert spec["voice_ref"] == "voice-123"


def test_persona_spec_voice_ref_is_optional():
    from rehearse.backends.base import PersonaSpec
    spec = PersonaSpec(name="Sam", gender="female", system_prompt="You are Sam.", voice_ref=None)
    assert spec["voice_ref"] is None


def test_conversation_backend_protocol_exists():
    from rehearse.backends.base import ConversationBackend
    # Structural check: protocol has the five required methods
    assert hasattr(ConversationBackend, "start")
    assert hasattr(ConversationBackend, "send_caller_audio")
    assert hasattr(ConversationBackend, "inject_speech")
    assert hasattr(ConversationBackend, "swap_persona")
    assert hasattr(ConversationBackend, "close")
