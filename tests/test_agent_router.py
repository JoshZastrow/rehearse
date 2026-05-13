"""Tests for AgentRegistry and PhaseRouter."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rehearse.agents.registry import AgentRegistry
from rehearse.agents.roles.character import CharacterAgent
from rehearse.agents.roles.feedback import FeedbackCoachAgent
from rehearse.agents.roles.intake import IntakeCoachAgent
from rehearse.agents.router import PhaseRouter
from rehearse.types import ConsentState, Phase, PhaseTiming, Session
from tests.fakes import FakeMemoryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(phase: Phase = Phase.INTAKE) -> Session:
    return Session(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        phone_number_hash="caller-abc",
        consent=ConsentState.GRANTED,
        phase_timings=[
            PhaseTiming(
                phase=phase,
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                budget_seconds=60,
            )
        ],
    )


def _build_registry() -> AgentRegistry:
    memory = FakeMemoryManager()
    registry = AgentRegistry()
    registry.register(IntakeCoachAgent(memory))
    registry.register(CharacterAgent(memory))
    registry.register(FeedbackCoachAgent(memory))
    return registry


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


def test_registry_returns_named_agent() -> None:
    registry = _build_registry()
    assert registry.get("intake_coach").name == "intake_coach"
    assert registry.get("character").name == "character"
    assert registry.get("feedback_coach").name == "feedback_coach"


def test_registry_falls_back_to_intake_coach_for_unknown_name() -> None:
    registry = _build_registry()
    assert registry.get("nonexistent").name == "intake_coach"


def test_registry_lists_all_registered_names() -> None:
    registry = _build_registry()
    names = registry.names()
    assert "intake_coach" in names
    assert "character" in names
    assert "feedback_coach" in names


def test_registry_register_overwrites_existing_name() -> None:
    memory = FakeMemoryManager()
    registry = AgentRegistry()
    registry.register(IntakeCoachAgent(memory))
    new_agent = IntakeCoachAgent(FakeMemoryManager(prefetch_response="new"))
    registry.register(new_agent)
    assert registry.get("intake_coach") is new_agent


# ---------------------------------------------------------------------------
# PhaseRouter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_router_intake_returns_intake_coach() -> None:
    router = PhaseRouter(_build_registry())
    agent = await router.route(_session(Phase.INTAKE))
    assert agent.name == "intake_coach"


@pytest.mark.asyncio
async def test_phase_router_practice_returns_character() -> None:
    router = PhaseRouter(_build_registry())
    agent = await router.route(_session(Phase.PRACTICE))
    assert agent.name == "character"


@pytest.mark.asyncio
async def test_phase_router_feedback_returns_feedback_coach() -> None:
    router = PhaseRouter(_build_registry())
    agent = await router.route(_session(Phase.FEEDBACK))
    assert agent.name == "feedback_coach"


@pytest.mark.asyncio
async def test_phase_router_no_timings_defaults_to_intake_coach() -> None:
    session = Session(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        phase_timings=[],
    )
    router = PhaseRouter(_build_registry())
    agent = await router.route(session)
    assert agent.name == "intake_coach"


@pytest.mark.asyncio
async def test_phase_router_uses_most_recent_open_timing() -> None:
    """When multiple phases are in timings, the most recent open one wins."""
    session = Session(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        phase_timings=[
            PhaseTiming(
                phase=Phase.INTAKE,
                started_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                ended_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                budget_seconds=60,
            ),
            PhaseTiming(
                phase=Phase.PRACTICE,
                started_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                budget_seconds=180,
            ),
        ],
    )
    router = PhaseRouter(_build_registry())
    agent = await router.route(session)
    assert agent.name == "character"


@pytest.mark.asyncio
async def test_phase_router_accepts_role_hint_but_prefers_session_phase() -> None:
    """role_hint from Hume query param is accepted but phase wins."""
    router = PhaseRouter(_build_registry())
    agent = await router.route(_session(Phase.FEEDBACK), role_hint="coach")
    assert agent.name == "feedback_coach"  # session phase wins
