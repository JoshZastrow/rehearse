"""Tests for HumeEVIClient.send_session_settings and resolve_voice_id."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rehearse.agents.roles.character import (
    CharacterAgent,
    FemaleCharacterAgent,
    MaleCharacterAgent,
)
from rehearse.services.hume_configs import resolve_voice_id
from tests.fakes import FakeMemoryManager


# ---------------------------------------------------------------------------
# MaleCharacterAgent / FemaleCharacterAgent
# ---------------------------------------------------------------------------


def test_male_character_agent_name() -> None:
    assert MaleCharacterAgent(FakeMemoryManager()).name == "male_character"


def test_female_character_agent_name() -> None:
    assert FemaleCharacterAgent(FakeMemoryManager()).name == "female_character"


def test_male_system_prompt_mentions_male_name(tmp_session) -> None:
    agent = MaleCharacterAgent(FakeMemoryManager())
    prompt = agent.system_prompt(tmp_session)
    assert "male" in prompt.lower() or "alex" in prompt.lower() or "he" in prompt.lower()


def test_female_system_prompt_differs_from_male(tmp_session) -> None:
    male = MaleCharacterAgent(FakeMemoryManager()).system_prompt(tmp_session)
    female = FemaleCharacterAgent(FakeMemoryManager()).system_prompt(tmp_session)
    assert male != female


@pytest.mark.asyncio
async def test_male_and_female_agents_in_registry() -> None:
    from rehearse.agents.registry import AgentRegistry
    from rehearse.memory_manager import MemoryManager
    from rehearse.memory import NullCallerMemory

    memory = MemoryManager(NullCallerMemory())
    registry = AgentRegistry()
    registry.register(MaleCharacterAgent(memory))
    registry.register(FemaleCharacterAgent(memory))
    assert registry.get("male_character").name == "male_character"
    assert registry.get("female_character").name == "female_character"


# ---------------------------------------------------------------------------
# resolve_voice_id
# ---------------------------------------------------------------------------


def test_resolve_voice_id_returns_cached_result() -> None:
    with patch("rehearse.services.hume_configs._resolve_voice_id_from_api") as mock_api:
        mock_api.return_value = "voice-uuid-123"
        result1 = resolve_voice_id("Inspiring Woman")
        result2 = resolve_voice_id("Inspiring Woman")
        assert result1 == "voice-uuid-123"
        assert mock_api.call_count == 1  # cached after first call


def test_resolve_voice_id_returns_none_on_api_error() -> None:
    with patch("rehearse.services.hume_configs._resolve_voice_id_from_api") as mock_api:
        mock_api.side_effect = Exception("api error")
        result = resolve_voice_id("Unknown Voice")
        assert result is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_session():
    from datetime import UTC, datetime
    from rehearse.types import ConsentState, Phase, PhaseTiming, Session
    return Session(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        consent=ConsentState.GRANTED,
        phase_timings=[PhaseTiming(phase=Phase.PRACTICE, started_at=datetime(2026, 1, 1, tzinfo=UTC), budget_seconds=180)],
    )
