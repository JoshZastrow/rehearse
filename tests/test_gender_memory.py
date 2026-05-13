"""Tests for CallerMemory gender preference methods."""

from __future__ import annotations

import pytest

from rehearse.memory import InMemoryCallerMemory


@pytest.mark.asyncio
async def test_get_gender_preference_returns_none_for_new_caller() -> None:
    m = InMemoryCallerMemory()
    assert await m.get_gender_preference("caller-abc") is None


@pytest.mark.asyncio
async def test_record_and_get_gender_preference_male() -> None:
    m = InMemoryCallerMemory()
    await m.record_gender_preference("caller-abc", "male")
    assert await m.get_gender_preference("caller-abc") == "male"


@pytest.mark.asyncio
async def test_record_and_get_gender_preference_female() -> None:
    m = InMemoryCallerMemory()
    await m.record_gender_preference("caller-abc", "female")
    assert await m.get_gender_preference("caller-abc") == "female"


@pytest.mark.asyncio
async def test_gender_preference_overwritable() -> None:
    m = InMemoryCallerMemory()
    await m.record_gender_preference("caller-abc", "male")
    await m.record_gender_preference("caller-abc", "female")
    assert await m.get_gender_preference("caller-abc") == "female"


@pytest.mark.asyncio
async def test_gender_preference_isolated_per_caller() -> None:
    m = InMemoryCallerMemory()
    await m.record_gender_preference("caller-a", "male")
    assert await m.get_gender_preference("caller-b") is None


@pytest.mark.asyncio
async def test_clear_caller_removes_gender_preference() -> None:
    m = InMemoryCallerMemory()
    await m.record_gender_preference("caller-abc", "male")
    await m.clear_caller("caller-abc")
    assert await m.get_gender_preference("caller-abc") is None


@pytest.mark.asyncio
async def test_clear_caller_does_not_affect_other_callers() -> None:
    m = InMemoryCallerMemory()
    await m.record_gender_preference("caller-a", "male")
    await m.record_gender_preference("caller-b", "female")
    await m.clear_caller("caller-a")
    assert await m.get_gender_preference("caller-b") == "female"


@pytest.mark.asyncio
async def test_clear_caller_also_clears_consent() -> None:
    m = InMemoryCallerMemory()
    await m.record_consent("caller-abc")
    await m.clear_caller("caller-abc")
    assert await m.has_prior_consent("caller-abc") is False
