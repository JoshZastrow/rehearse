"""Tests for LLMTransport protocol and AnthropicTransport implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rehearse.agents.clm import CLMMessage
from rehearse.transports.anthropic import AnthropicTransport
from rehearse.transports.base import LLMTransport


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_anthropic_transport_implements_protocol() -> None:
    transport = AnthropicTransport(api_key="test")
    assert isinstance(transport, LLMTransport)


# ---------------------------------------------------------------------------
# convert_messages
# ---------------------------------------------------------------------------


def test_convert_messages_normalizes_explicit_roles() -> None:
    transport = AnthropicTransport(api_key="test")
    messages = [
        CLMMessage(role="user", content="hello"),
        CLMMessage(role="assistant", content="hi"),
    ]
    result = transport.convert_messages(messages)
    assert result == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_convert_messages_infers_assistant_from_type() -> None:
    transport = AnthropicTransport(api_key="test")
    messages = [CLMMessage(type="assistant_message", content="I'm the coach.")]
    result = transport.convert_messages(messages)
    assert result[0]["role"] == "assistant"


def test_convert_messages_defaults_unknown_to_user() -> None:
    transport = AnthropicTransport(api_key="test")
    messages = [CLMMessage(role=None, type=None, content="something")]
    result = transport.convert_messages(messages)
    assert result[0]["role"] == "user"


def test_convert_messages_skips_empty_content() -> None:
    transport = AnthropicTransport(api_key="test")
    messages = [
        CLMMessage(role="user", content=""),
        CLMMessage(role="user", content="   "),
        CLMMessage(role="user", content="real"),
    ]
    result = transport.convert_messages(messages)
    assert len(result) == 1
    assert result[0]["content"] == "real"


def test_convert_messages_appends_prosody_scores() -> None:
    transport = AnthropicTransport(api_key="test")
    messages = [
        CLMMessage(
            role="user",
            content="I need to talk to my boss.",
            models={"prosody": {"scores": {"Anxiety": 0.85, "Determination": 0.72, "Joy": 0.10}}},
        )
    ]
    result = transport.convert_messages(messages)
    assert "Prosody cues:" in result[0]["content"]
    assert "Anxiety=0.85" in result[0]["content"]
    assert "Determination=0.72" in result[0]["content"]


def test_convert_messages_skips_null_prosody() -> None:
    transport = AnthropicTransport(api_key="test")
    messages = [CLMMessage(role="user", content="hi", models={"prosody": None})]
    result = transport.convert_messages(messages)
    assert "Prosody" not in result[0]["content"]


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_text_chunks() -> None:
    transport = AnthropicTransport(api_key="test")

    # Replace the internal async client with a fake that yields text
    fake_stream = _FakeAnthropicStream(["Hello ", "world"])
    transport._client = _FakeAnthropicClient(fake_stream)  # type: ignore[assignment]

    chunks = []
    async for chunk in transport.stream(
        system_blocks=[{"type": "text", "text": "You are a coach."}],
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-6",
    ):
        chunks.append(chunk)

    assert chunks == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_stream_skips_empty_chunks() -> None:
    transport = AnthropicTransport(api_key="test")
    fake_stream = _FakeAnthropicStream(["", "hello", ""])
    transport._client = _FakeAnthropicClient(fake_stream)  # type: ignore[assignment]

    chunks = [c async for c in transport.stream(
        system_blocks=[], messages=[], model="claude-sonnet-4-6"
    )]
    assert chunks == ["hello"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTextStream:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._texts:
            raise StopAsyncIteration
        return self._texts.pop(0)


class _FakeAnthropicStream:
    def __init__(self, texts: list[str]) -> None:
        self.text_stream = _FakeTextStream(texts)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeAnthropicClient:
    def __init__(self, fake_stream: _FakeAnthropicStream) -> None:
        self._stream = fake_stream
        self.messages = self

    def stream(self, **kwargs) -> _FakeAnthropicStream:
        return self._stream
