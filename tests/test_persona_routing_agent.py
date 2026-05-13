"""Tests for PersonaRoutingAgent — tool-call-based persona selection."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rehearse.agents.persona_routing_agent import PersonaRoutingAgent
from rehearse.personas.registry import PERSONA_REGISTRY, PersonaRegistry


def _registry() -> PersonaRegistry:
    return PersonaRegistry(PERSONA_REGISTRY)


def _make_agent(registry: PersonaRegistry, *, tool_persona_id: str) -> PersonaRoutingAgent:
    """Build a PersonaRoutingAgent whose LLM always picks `tool_persona_id`."""
    client = AsyncMock()
    # Simulate: model calls list_personas tool, then returns persona_id
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "list_personas"
    tool_use_block.id = "tu_1"
    tool_use_block.input = {}

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = f'{{"persona_id": "{tool_persona_id}"}}'

    first_response = MagicMock()
    first_response.stop_reason = "tool_use"
    first_response.content = [tool_use_block]

    second_response = MagicMock()
    second_response.stop_reason = "end_turn"
    second_response.content = [text_block]

    client.messages.create = AsyncMock(side_effect=[first_response, second_response])
    return PersonaRoutingAgent(registry, client=client)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_returns_male_default_when_gender_hint_male() -> None:
    agent = _make_agent(_registry(), tool_persona_id="male_coach_default")
    result = await agent.select(
        transcript="I need to ask my boss for a raise.",
        gender_hint="male",
    )
    assert result.gender == "male"
    assert result.id == "male_coach_default"


@pytest.mark.asyncio
async def test_agent_returns_female_default_when_gender_hint_female() -> None:
    agent = _make_agent(_registry(), tool_persona_id="female_coach_default")
    result = await agent.select(
        transcript="I need to talk to my partner.",
        gender_hint="female",
    )
    assert result.gender == "female"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_falls_back_to_default_on_unknown_persona_id() -> None:
    agent = _make_agent(_registry(), tool_persona_id="nonexistent_id")
    result = await agent.select(transcript="some situation", gender_hint="male")
    assert result.gender == "male"
    assert "default" in result.tags


@pytest.mark.asyncio
async def test_agent_falls_back_on_llm_exception() -> None:
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=Exception("rate limit"))
    agent = PersonaRoutingAgent(_registry(), client=client)
    result = await agent.select(transcript="some situation", gender_hint="female")
    assert result.gender == "female"
    assert "default" in result.tags


@pytest.mark.asyncio
async def test_agent_falls_back_when_no_gender_hint() -> None:
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=Exception("error"))
    agent = PersonaRoutingAgent(_registry(), client=client)
    result = await agent.select(transcript="some situation", gender_hint=None)
    assert result is not None  # returns some persona, not None


# ---------------------------------------------------------------------------
# Tool call mechanics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_passes_gender_hint_to_tool_call() -> None:
    registry = _registry()
    agent = _make_agent(registry, tool_persona_id="male_coach_default")
    await agent.select(transcript="situation", gender_hint="male")
    # First LLM call should include the list_personas tool definition
    first_call_kwargs = agent._client.messages.create.call_args_list[0].kwargs
    tool_names = [t["name"] for t in first_call_kwargs.get("tools", [])]
    assert "list_personas" in tool_names
