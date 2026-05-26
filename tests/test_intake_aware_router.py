"""Tests for IntakeAwareRouter — topic-aware practice phase routing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from rehearse.agents.registry import AgentRegistry
from rehearse.agents.roles.character import (
    CharacterAgent,
    FemaleCharacterAgent,
    MaleCharacterAgent,
)
from rehearse.agents.roles.feedback import FeedbackCoachAgent
from rehearse.agents.roles.intake import IntakeCoachAgent
from rehearse.agents.router import IntakeAwareRouter
from rehearse.memory.memory import InMemoryCallerMemory, NullCallerMemory
from rehearse.types import (
    ConsentState,
    IntakeRecord,
    Phase,
    PhaseTiming,
    Session,
)
from tests.fakes import FakeMemoryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(
    phase: Phase = Phase.PRACTICE,
    situation: str | None = "ask my manager for a raise",
    topic_category: str | None = None,
    phone_number_hash: str = "caller-abc",
    selected_persona_id: str | None = None,
) -> Session:
    s = Session(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        phone_number_hash=phone_number_hash,
        consent=ConsentState.GRANTED,
        phase_timings=[
            PhaseTiming(
                phase=phase,
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                budget_seconds=60,
            )
        ],
        selected_persona_id=selected_persona_id,
    )
    if situation is not None:
        s.intake = IntakeRecord(
            session_id="sess-001",
            situation=situation,
            counterparty_relationship="manager",
            counterparty_description="direct manager",
            stakes="career growth",
            user_goal="get a raise",
            topic_category=topic_category,
            captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    return s


def _build_registry() -> AgentRegistry:
    memory = FakeMemoryManager()
    registry = AgentRegistry()
    registry.register(IntakeCoachAgent(memory))
    registry.register(CharacterAgent(memory))
    registry.register(MaleCharacterAgent(memory))
    registry.register(FemaleCharacterAgent(memory))
    registry.register(FeedbackCoachAgent(memory))
    return registry


class _FakeTopicClassifier:
    """Replaces classify_topic with a canned return value."""

    def __init__(self, topic: Literal["work", "relationship", "other"]) -> None:
        self.topic = topic
        self.calls: list[str] = []

    async def __call__(self, situation: str, *, client, model: str = "") -> str:
        self.calls.append(situation)
        return self.topic


# ---------------------------------------------------------------------------
# Non-PRACTICE phases delegate to PhaseRouter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intake_phase_routes_to_intake_coach() -> None:
    registry = _build_registry()
    memory = NullCallerMemory()
    router = IntakeAwareRouter(registry, memory, llm_client=None)
    session = _session(phase=Phase.INTAKE)
    agent = await router.route(session)
    assert agent.name == "intake_coach"


@pytest.mark.asyncio
async def test_feedback_phase_routes_to_feedback_coach() -> None:
    registry = _build_registry()
    memory = NullCallerMemory()
    router = IntakeAwareRouter(registry, memory, llm_client=None)
    session = _session(phase=Phase.FEEDBACK)
    agent = await router.route(session)
    assert agent.name == "feedback_coach"


# ---------------------------------------------------------------------------
# PRACTICE phase — memory has preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_practice_routes_to_male_agent_when_preference_is_male() -> None:
    registry = _build_registry()
    memory = InMemoryCallerMemory()
    await memory.record_agent_preference("caller-abc", "work", "male")

    classifier = _FakeTopicClassifier("work")
    router = IntakeAwareRouter(registry, memory, llm_client=object(), _classify=classifier)

    session = _session(phase=Phase.PRACTICE, situation="ask my manager for a raise")
    agent = await router.route(session)
    assert agent.name == "male_character"


@pytest.mark.asyncio
async def test_practice_routes_to_female_agent_when_preference_is_female() -> None:
    registry = _build_registry()
    memory = InMemoryCallerMemory()
    await memory.record_agent_preference("caller-abc", "relationship", "female")

    classifier = _FakeTopicClassifier("relationship")
    router = IntakeAwareRouter(registry, memory, llm_client=object(), _classify=classifier)

    session = _session(phase=Phase.PRACTICE, situation="talk to my partner about moving in")
    agent = await router.route(session)
    assert agent.name == "female_character"


# ---------------------------------------------------------------------------
# PRACTICE phase — no memory → default to female
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_practice_defaults_to_female_when_no_preference() -> None:
    registry = _build_registry()
    memory = NullCallerMemory()

    classifier = _FakeTopicClassifier("work")
    router = IntakeAwareRouter(registry, memory, llm_client=object(), _classify=classifier)

    session = _session(phase=Phase.PRACTICE)
    agent = await router.route(session)
    assert agent.name == "female_character"


@pytest.mark.asyncio
async def test_practice_defaults_to_female_when_no_intake() -> None:
    registry = _build_registry()
    memory = NullCallerMemory()
    router = IntakeAwareRouter(registry, memory, llm_client=None)

    session = _session(phase=Phase.PRACTICE, situation=None)
    agent = await router.route(session)
    assert agent.name == "female_character"


# ---------------------------------------------------------------------------
# PRACTICE phase — topic already on intake record (skip classify call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_practice_uses_cached_topic_category_from_intake() -> None:
    registry = _build_registry()
    memory = InMemoryCallerMemory()
    await memory.record_agent_preference("caller-abc", "work", "male")

    classifier = _FakeTopicClassifier("work")
    router = IntakeAwareRouter(registry, memory, llm_client=object(), _classify=classifier)

    # topic_category already set on intake — classifier should NOT be called
    session = _session(phase=Phase.PRACTICE, situation="ask for a raise", topic_category="work")
    agent = await router.route(session)
    assert agent.name == "male_character"
    assert len(classifier.calls) == 0  # used cached value


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_practice_falls_back_to_female_on_classify_error() -> None:
    registry = _build_registry()
    memory = NullCallerMemory()

    async def _failing_classify(situation: str, *, client, model: str = "") -> str:
        raise RuntimeError("API error")

    router = IntakeAwareRouter(
        registry, memory, llm_client=object(), _classify=_failing_classify
    )
    session = _session(phase=Phase.PRACTICE)
    agent = await router.route(session)
    assert agent.name == "female_character"


@pytest.mark.asyncio
async def test_practice_falls_back_to_female_on_memory_error() -> None:
    registry = _build_registry()

    class _BrokenMemory:
        async def get_agent_preference(self, *_) -> None:
            raise RuntimeError("Honcho down")

    classifier = _FakeTopicClassifier("work")
    router = IntakeAwareRouter(registry, _BrokenMemory(), llm_client=object(), _classify=classifier)
    session = _session(phase=Phase.PRACTICE)
    agent = await router.route(session)
    assert agent.name == "female_character"
