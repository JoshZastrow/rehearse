"""LLMTransport protocol — the interface every provider transport implements.

A transport owns exactly two things:
  1. Converting CLM messages to the provider's wire format.
  2. Executing the streaming (or buffered) API call.

It does NOT own: prompt assembly, memory retrieval, routing, retry logic,
or session management. Those belong to CLMResponder and the agent layer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from rehearse.agents.clm import CLMMessage


@runtime_checkable
class LLMTransport(Protocol):
    """Convert messages to provider format and stream or generate a response."""

    def convert_messages(self, messages: list[CLMMessage]) -> list[dict]:
        """Convert Hume CLM messages to the provider's native message list."""
        ...

    async def stream(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> AsyncIterator[str]:
        """Stream response text chunks. Yields plain text, not SSE events."""
        ...

    async def generate(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> str:
        """Return the complete response as a string (buffered, non-streaming)."""
        ...
