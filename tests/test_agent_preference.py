"""Tests for per-topic get_agent_preference / record_agent_preference on CallerMemory."""

from __future__ import annotations

import pytest

from rehearse.memory.memory import InMemoryCallerMemory, NullCallerMemory


# ---------------------------------------------------------------------------
# InMemoryCallerMemory — per-topic preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_agent_preference_returns_none_for_new_caller() -> None:
    m = InMemoryCallerMemory()
    assert await m.get_agent_preference("caller-abc", "work") is None


@pytest.mark.asyncio
async def test_get_agent_preference_returns_none_for_unknown_topic() -> None:
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-abc", "work", "male")
    assert await m.get_agent_preference("caller-abc", "relationship") is None


@pytest.mark.asyncio
async def test_record_and_get_agent_preference_male() -> None:
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-abc", "work", "male")
    assert await m.get_agent_preference("caller-abc", "work") == "male"


@pytest.mark.asyncio
async def test_record_and_get_agent_preference_female() -> None:
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-abc", "relationship", "female")
    assert await m.get_agent_preference("caller-abc", "relationship") == "female"


@pytest.mark.asyncio
async def test_agent_preference_per_topic_isolated() -> None:
    """Different topics store separate preferences for the same caller."""
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-abc", "work", "male")
    await m.record_agent_preference("caller-abc", "relationship", "female")
    assert await m.get_agent_preference("caller-abc", "work") == "male"
    assert await m.get_agent_preference("caller-abc", "relationship") == "female"


@pytest.mark.asyncio
async def test_agent_preference_overwritable() -> None:
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-abc", "work", "male")
    await m.record_agent_preference("caller-abc", "work", "female")
    assert await m.get_agent_preference("caller-abc", "work") == "female"


@pytest.mark.asyncio
async def test_agent_preference_isolated_per_caller() -> None:
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-a", "work", "male")
    assert await m.get_agent_preference("caller-b", "work") is None


@pytest.mark.asyncio
async def test_clear_caller_removes_agent_preferences() -> None:
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-abc", "work", "male")
    await m.record_agent_preference("caller-abc", "relationship", "female")
    await m.clear_caller("caller-abc")
    assert await m.get_agent_preference("caller-abc", "work") is None
    assert await m.get_agent_preference("caller-abc", "relationship") is None


@pytest.mark.asyncio
async def test_clear_caller_does_not_affect_other_callers_preferences() -> None:
    m = InMemoryCallerMemory()
    await m.record_agent_preference("caller-a", "work", "male")
    await m.record_agent_preference("caller-b", "work", "female")
    await m.clear_caller("caller-a")
    assert await m.get_agent_preference("caller-b", "work") == "female"


# ---------------------------------------------------------------------------
# NullCallerMemory — per-topic preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_memory_get_agent_preference_returns_none() -> None:
    m = NullCallerMemory()
    assert await m.get_agent_preference("caller-abc", "work") is None


@pytest.mark.asyncio
async def test_null_memory_record_agent_preference_is_noop() -> None:
    m = NullCallerMemory()
    await m.record_agent_preference("caller-abc", "work", "male")
    assert await m.get_agent_preference("caller-abc", "work") is None
