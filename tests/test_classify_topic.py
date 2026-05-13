"""Tests for classify_topic() — situation → topic category classifier."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rehearse.agents.topic import classify_topic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(response_text: str) -> MagicMock:
    """Build a mock AsyncAnthropic client that returns response_text."""
    client = MagicMock()
    text_block = MagicMock()
    text_block.text = response_text
    mock_response = MagicMock()
    mock_response.content = [text_block]
    client.messages.create = AsyncMock(return_value=mock_response)
    return client


# ---------------------------------------------------------------------------
# Work topic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_topic_work_raise() -> None:
    client = _mock_client("work")
    result = await classify_topic("ask my manager for a raise", client=client)
    assert result == "work"


@pytest.mark.asyncio
async def test_classify_topic_work_promotion() -> None:
    client = _mock_client("work")
    result = await classify_topic("talk to my boss about a promotion", client=client)
    assert result == "work"


@pytest.mark.asyncio
async def test_classify_topic_work_performance() -> None:
    client = _mock_client("work")
    result = await classify_topic("address performance concerns with a direct report", client=client)
    assert result == "work"


# ---------------------------------------------------------------------------
# Relationship topic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_topic_relationship_partner() -> None:
    client = _mock_client("relationship")
    result = await classify_topic("talk to my partner about moving in together", client=client)
    assert result == "relationship"


@pytest.mark.asyncio
async def test_classify_topic_relationship_family() -> None:
    client = _mock_client("relationship")
    result = await classify_topic("set boundaries with my parents about holiday visits", client=client)
    assert result == "relationship"


# ---------------------------------------------------------------------------
# Other / fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_topic_other_unrecognized() -> None:
    client = _mock_client("other")
    result = await classify_topic("figure out my life direction", client=client)
    assert result == "other"


@pytest.mark.asyncio
async def test_classify_topic_returns_other_on_api_error() -> None:
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("network error"))
    result = await classify_topic("ask my manager for a raise", client=client)
    assert result == "other"


@pytest.mark.asyncio
async def test_classify_topic_returns_other_on_unexpected_response() -> None:
    client = _mock_client("UNKNOWN_CATEGORY")
    result = await classify_topic("something ambiguous", client=client)
    assert result == "other"


@pytest.mark.asyncio
async def test_classify_topic_returns_other_on_empty_situation() -> None:
    client = _mock_client("other")
    result = await classify_topic("", client=client)
    assert result == "other"


# ---------------------------------------------------------------------------
# Prompt shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_topic_uses_low_token_budget() -> None:
    client = _mock_client("work")
    await classify_topic("ask for a raise", client=client)
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs.get("max_tokens", 0) <= 10


@pytest.mark.asyncio
async def test_classify_topic_uses_zero_temperature() -> None:
    client = _mock_client("work")
    await classify_topic("ask for a raise", client=client)
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs.get("temperature", 1) == 0


# ---------------------------------------------------------------------------
# Live API test (skipped without ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.live_api
@pytest.mark.asyncio
async def test_classify_topic_live_work() -> None:
    import os

    from anthropic import AsyncAnthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    client = AsyncAnthropic(api_key=api_key)
    result = await classify_topic("ask my manager for a salary increase", client=client)
    assert result == "work"
