"""Tests for RehearseAgent protocol and the three concrete agent roles."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rehearse.agents.roles.base import RehearseAgent, wrap_memory_context
from rehearse.agents.roles.character import CharacterAgent
from rehearse.agents.roles.feedback import FeedbackCoachAgent
from rehearse.agents.roles.intake import IntakeCoachAgent
from rehearse.types import ConsentState, Phase, PhaseTiming, Session
from tests.fakes import FakeMemoryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(*, caller_hash: str = "abc123", phase: Phase = Phase.INTAKE) -> Session:
    return Session(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        phone_number_hash=caller_hash,
        consent=ConsentState.GRANTED,
        phase_timings=[
            PhaseTiming(phase=phase, started_at=datetime(2026, 1, 1, tzinfo=UTC), budget_seconds=60)
        ],
    )


# ---------------------------------------------------------------------------
# wrap_memory_context helper
# ---------------------------------------------------------------------------


def test_wrap_memory_context_returns_fenced_block() -> None:
    result = wrap_memory_context("Prior: salary negotiation")
    assert "<memory-context>" in result
    assert "</memory-context>" in result
    assert "Prior: salary negotiation" in result
    assert "System note:" in result


def test_wrap_memory_context_returns_empty_for_blank_input() -> None:
    assert wrap_memory_context("") == ""
    assert wrap_memory_context("   ") == ""


# ---------------------------------------------------------------------------
# IntakeCoachAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intake_coach_recall_calls_prefetch_with_domain_query() -> None:
    memory = FakeMemoryManager(prefetch_response="Prior: salary negotiation")
    agent = IntakeCoachAgent(memory)
    result = await agent.recall(_session(caller_hash="caller-abc"))
    assert result == "Prior: salary negotiation"
    assert len(memory.prefetch_calls) == 1
    caller, query = memory.prefetch_calls[0]
    assert caller == "caller-abc"
    assert query == agent._RECALL_QUERY


@pytest.mark.asyncio
async def test_intake_coach_recall_returns_empty_for_new_caller() -> None:
    agent = IntakeCoachAgent(FakeMemoryManager(prefetch_response=""))
    assert await agent.recall(_session(caller_hash="new-caller")) == ""


@pytest.mark.asyncio
async def test_intake_coach_recall_returns_empty_for_missing_caller_hash() -> None:
    agent = IntakeCoachAgent(FakeMemoryManager(prefetch_response="something"))
    assert await agent.recall(_session(caller_hash="")) == ""


def test_intake_coach_system_prompt_includes_base_prompt() -> None:
    agent = IntakeCoachAgent(FakeMemoryManager())
    prompt = agent.system_prompt(_session(), memory_context="")
    assert len(prompt) > 20  # non-empty coaching prompt


def test_intake_coach_system_prompt_wraps_memory_context() -> None:
    agent = IntakeCoachAgent(FakeMemoryManager())
    prompt = agent.system_prompt(_session(), memory_context="prior topics here")
    assert "<memory-context>" in prompt
    assert "prior topics here" in prompt


def test_intake_coach_system_prompt_no_memory_block_when_empty() -> None:
    agent = IntakeCoachAgent(FakeMemoryManager())
    prompt = agent.system_prompt(_session(), memory_context="")
    assert "<memory-context>" not in prompt


@pytest.mark.asyncio
async def test_intake_coach_after_turn_is_noop() -> None:
    memory = FakeMemoryManager()
    agent = IntakeCoachAgent(memory)
    await agent.after_turn(_session(), "user said hi", "agent replied")
    assert memory.stored_sessions == []


# ---------------------------------------------------------------------------
# CharacterAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_character_recall_returns_empty_phase1() -> None:
    memory = FakeMemoryManager(prefetch_response="should not be called")
    agent = CharacterAgent(memory)
    result = await agent.recall(_session(phase=Phase.PRACTICE))
    assert result == ""
    assert memory.prefetch_calls == []


def test_character_system_prompt_is_non_empty() -> None:
    agent = CharacterAgent(FakeMemoryManager())
    prompt = agent.system_prompt(_session())
    assert len(prompt) > 10


def test_character_system_prompt_uses_persona_when_present() -> None:
    from rehearse.types import CounterpartyPersona

    agent = CharacterAgent(FakeMemoryManager())
    session = _session()
    session.persona = CounterpartyPersona(
        session_id="s1",
        name="Alex",
        relationship="difficult manager",
        personality_prompt="You are a skeptical manager.",
        compiled_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    prompt = agent.system_prompt(session)
    assert "difficult manager" in prompt or "manager" in prompt or "skeptical" in prompt


# ---------------------------------------------------------------------------
# FeedbackCoachAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_coach_recall_calls_prefetch_with_domain_query() -> None:
    memory = FakeMemoryManager(prefetch_response="Last session: improved directness")
    agent = FeedbackCoachAgent(memory)
    result = await agent.recall(_session(caller_hash="caller-abc"))
    assert result == "Last session: improved directness"
    caller, query = memory.prefetch_calls[0]
    assert caller == "caller-abc"
    assert query == agent._RECALL_QUERY


@pytest.mark.asyncio
async def test_feedback_coach_recall_returns_empty_for_new_caller() -> None:
    agent = FeedbackCoachAgent(FakeMemoryManager(prefetch_response=""))
    assert await agent.recall(_session()) == ""


def test_feedback_coach_system_prompt_wraps_context() -> None:
    agent = FeedbackCoachAgent(FakeMemoryManager())
    prompt = agent.system_prompt(_session(), memory_context="prior feedback here")
    assert "<memory-context>" in prompt
    assert "prior feedback here" in prompt


@pytest.mark.asyncio
async def test_feedback_coach_after_turn_is_noop_phase1() -> None:
    memory = FakeMemoryManager()
    agent = FeedbackCoachAgent(memory)
    await agent.after_turn(_session(), "user said", "agent said")
    assert memory.stored_sessions == []


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_all_agents_have_name() -> None:
    memory = FakeMemoryManager()
    assert IntakeCoachAgent(memory).name == "intake_coach"
    assert CharacterAgent(memory).name == "character"
    assert FeedbackCoachAgent(memory).name == "feedback_coach"


def test_all_agents_satisfy_rehearse_agent_protocol() -> None:
    memory = FakeMemoryManager()
    for agent in [IntakeCoachAgent(memory), CharacterAgent(memory), FeedbackCoachAgent(memory)]:
        assert isinstance(agent, RehearseAgent)
