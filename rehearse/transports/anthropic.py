"""Anthropic Messages API transport.

Owns: message format conversion (CLM → Anthropic), async client, streaming.
Does not own: prompt assembly, memory, routing, retry logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

import structlog
from anthropic import AsyncAnthropic

from rehearse.agents.clm import CLMMessage

log = structlog.get_logger(__name__)


class AnthropicTransport:
    """Converts CLM messages to Anthropic format and executes streaming calls."""

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    def convert_messages(self, messages: list[CLMMessage]) -> list[dict]:
        """Convert Hume CLM messages to Anthropic's message format."""
        normalized: list[dict] = []
        for message in messages:
            role = self._message_role(message)
            content = self._message_content(message)
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    async def stream(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> AsyncIterator[str]:
        """Stream text chunks from the Anthropic Messages API."""
        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_blocks,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text

    async def generate(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> str:
        """Return the complete response as a string."""
        chunks = []
        async for text in self.stream(
            system_blocks=system_blocks,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            chunks.append(text)
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Private helpers (extracted from clm.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _message_role(message: CLMMessage) -> str:
        if message.role in {"user", "assistant"}:
            return message.role
        if message.type == "assistant_message":
            return "assistant"
        return "user"

    @staticmethod
    def _message_content(message: CLMMessage) -> str:
        text = (message.content or "").strip()
        if not text:
            return ""
        prosody_scores = {}
        if isinstance(message.models, dict):
            prosody_scores = (message.models.get("prosody") or {}).get("scores") or {}
        if not prosody_scores:
            return text
        top = sorted(prosody_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        summary = ", ".join(f"{name}={score:.2f}" for name, score in top)
        return f"{text}\n\nProsody cues: {summary}"
