"""Shared test doubles for agent design pattern tests.

All fakes are simple custom classes (no unittest.mock) following the existing
project pattern. Import from here rather than re-defining in each test file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rehearse.agents.clm import CLMChatRequest, CLMMessage


class FakeLLMTransport:
    """Fake transport that returns canned chunks and records the last call."""

    def __init__(
        self,
        chunks: tuple[str, ...] = ("Hello.",),
        *,
        raise_on_stream: bool = False,
    ) -> None:
        self.chunks = chunks
        self.raise_on_stream = raise_on_stream
        self.last_call: dict[str, Any] | None = None

    def convert_messages(self, messages: list[CLMMessage]) -> list[dict]:
        return [
            {"role": m.role or "user", "content": m.content or ""}
            for m in messages
            if m.content
        ]

    async def stream(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> AsyncIterator[str]:
        self.last_call = {"system_blocks": system_blocks, "messages": messages}
        if self.raise_on_stream:
            raise RuntimeError("stream error")
        for chunk in self.chunks:
            yield chunk

    async def generate(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> str:
        return "".join(self.chunks)


class FakeMemoryManager:
    """Fake memory manager for agent role and CLMResponder tests."""

    def __init__(self, prefetch_response: str = "") -> None:
        self.prefetch_response = prefetch_response
        self.prefetch_calls: list[tuple[str, str]] = []
        self.stored_sessions: list[tuple[str, list[dict]]] = []

    async def prefetch(self, caller_hash: str, query: str) -> str:
        self.prefetch_calls.append((caller_hash, query))
        return self.prefetch_response

    async def store_session(
        self,
        caller_hash: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> None:
        self.stored_sessions.append((caller_hash, messages))

    async def has_prior_consent(self, caller_hash: str) -> bool:
        return False

    async def record_consent(self, caller_hash: str) -> None:
        pass
