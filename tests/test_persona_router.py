"""Verify the SMS-body persona classifier and its fallback paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rehearse.agents.persona_router import infer_persona_key
from rehearse.services.hume_configs import PERSONAS


def _fake_client(reply_text: str) -> MagicMock:
    """Build a MagicMock AsyncAnthropic client that returns one reply."""
    client = MagicMock()
    message = MagicMock()
    message.content = [MagicMock(text=reply_text)]
    client.messages.create = AsyncMock(return_value=message)
    return client


@pytest.mark.asyncio
async def test_returns_classifier_choice():
    client = _fake_client("relationship_coach")
    key = await infer_persona_key(
        "I want to rehearse breaking up with my partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "relationship_coach"


@pytest.mark.asyncio
async def test_returns_default_when_classifier_picks_default():
    client = _fake_client("default")
    key = await infer_persona_key(
        "I'm asking my boss for a raise",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_falls_back_when_no_client():
    key = await infer_persona_key(
        "anything",
        PERSONAS,
        anthropic_client=None,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_falls_back_when_classifier_raises():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))
    key = await infer_persona_key(
        "I need to talk to my partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_falls_back_when_classifier_returns_unknown_key():
    client = _fake_client("not_a_real_persona")
    key = await infer_persona_key(
        "I need to talk to my partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"


@pytest.mark.asyncio
async def test_skips_classifier_for_empty_body():
    client = MagicMock()
    client.messages.create = AsyncMock()
    key = await infer_persona_key(
        "",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"
    client.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_classifier_for_inbound_call_marker():
    client = MagicMock()
    client.messages.create = AsyncMock()
    key = await infer_persona_key(
        "<inbound-call>",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "default"
    client.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_strips_classifier_response_whitespace_and_punctuation():
    client = _fake_client("  relationship_coach.  \n")
    key = await infer_persona_key(
        "partner",
        PERSONAS,
        anthropic_client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert key == "relationship_coach"
