"""Tests for MemoryManager — thin lifecycle orchestration over CallerMemoryProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rehearse.memory.memory_manager import MemoryManager


def _mock_provider(**overrides):
    provider = AsyncMock()
    provider.prefetch.return_value = overrides.get("prefetch", "")
    provider.store_session.return_value = None
    provider.has_prior_consent.return_value = overrides.get("has_prior_consent", False)
    provider.record_consent.return_value = None
    return provider


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_delegates_to_provider() -> None:
    provider = _mock_provider(prefetch="Prior: salary negotiation")
    mgr = MemoryManager(provider)
    result = await mgr.prefetch("caller-abc", "What has this caller worked on?")
    provider.prefetch.assert_awaited_once_with("caller-abc", "What has this caller worked on?")
    assert result == "Prior: salary negotiation"


@pytest.mark.asyncio
async def test_prefetch_returns_empty_string_on_provider_error() -> None:
    provider = _mock_provider()
    provider.prefetch.side_effect = Exception("network error")
    mgr = MemoryManager(provider)
    result = await mgr.prefetch("caller-abc", "query")
    assert result == ""


@pytest.mark.asyncio
async def test_prefetch_returns_empty_for_empty_caller_hash() -> None:
    provider = _mock_provider(prefetch="should not be called")
    mgr = MemoryManager(provider)
    result = await mgr.prefetch("", "query")
    assert result == ""
    provider.prefetch.assert_not_awaited()


# ---------------------------------------------------------------------------
# store_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_session_delegates_to_provider() -> None:
    provider = _mock_provider()
    mgr = MemoryManager(provider)
    messages = [{"role": "user", "content": "hi"}]
    await mgr.store_session("caller-abc", messages)
    provider.store_session.assert_awaited_once_with("caller-abc", messages)


@pytest.mark.asyncio
async def test_store_session_passes_rehearse_session_id_kwarg() -> None:
    provider = _mock_provider()
    mgr = MemoryManager(provider)
    await mgr.store_session("caller-abc", [], rehearse_session_id="sess-123")
    _, kwargs = provider.store_session.call_args
    assert kwargs.get("rehearse_session_id") == "sess-123"


@pytest.mark.asyncio
async def test_store_session_swallows_provider_error() -> None:
    provider = _mock_provider()
    provider.store_session.side_effect = Exception("db error")
    mgr = MemoryManager(provider)
    await mgr.store_session("caller-abc", [])  # must not raise


# ---------------------------------------------------------------------------
# consent delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_has_prior_consent_delegates_to_provider() -> None:
    provider = _mock_provider(has_prior_consent=True)
    mgr = MemoryManager(provider)
    assert await mgr.has_prior_consent("caller-abc") is True
    provider.has_prior_consent.assert_awaited_once_with("caller-abc")


@pytest.mark.asyncio
async def test_record_consent_delegates_to_provider() -> None:
    provider = _mock_provider()
    mgr = MemoryManager(provider)
    await mgr.record_consent("caller-abc")
    provider.record_consent.assert_awaited_once_with("caller-abc")
