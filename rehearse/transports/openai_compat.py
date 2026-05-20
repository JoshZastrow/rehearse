"""OpenAI-compatible transport — routes through LiteLLM proxy or any vLLM endpoint.

Drop-in replacement for AnthropicTransport. Flattens Anthropic-style
system_blocks into a single system message so callers (NewCLMResponder)
need no changes.

Also provides OpenAICompatLLMClient — a thin wrapper that mimics the
AsyncAnthropic.messages.create() interface so classify_topic() and the
IntakeAwareRouter work unchanged.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from rehearse.agents.clm import CLMMessage

log = structlog.get_logger(__name__)

_STREAM_TIMEOUT = 120.0


class OpenAICompatTransport:
    """Streams from any OpenAI-compatible /chat/completions endpoint.

    Flattens Anthropic-style system_blocks (list[dict] with type/text/cache_control)
    into a plain system message string — cache_control is Anthropic-only and
    ignored by vLLM/LiteLLM.
    """

    def __init__(self, *, base_url: str, api_key: str = "local", model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def convert_messages(self, messages: list[CLMMessage]) -> list[dict]:
        """Convert Hume CLM messages to OpenAI-compat format."""
        normalized: list[dict] = []
        for message in messages:
            role = _message_role(message)
            content = _message_content(message)
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
        """Stream text chunks via SSE from the OpenAI-compat endpoint."""
        full_messages = _build_messages(system_blocks, messages)
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        text = chunk["choices"][0]["delta"].get("content") or ""
                        if text:
                            yield text
                    except (KeyError, json.JSONDecodeError):
                        continue

    async def generate(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> str:
        chunks: list[str] = []
        async for text in self.stream(
            system_blocks=system_blocks,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            chunks.append(text)
        return "".join(chunks)


class OpenAICompatLLMClient:
    """Mimics AsyncAnthropic.messages.create() over an OpenAI-compat endpoint.

    Used by classify_topic() and IntakeAwareRouter so they need no changes.
    Returns an object with .content[0].text matching the Anthropic response shape.
    """

    def __init__(self, *, base_url: str, api_key: str = "local") -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.messages = _MessagesNamespace(self)

    async def _create(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.4,
        **_: Any,
    ) -> _AnthropicShapedResponse:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"] or ""
        return _AnthropicShapedResponse(text)


class _MessagesNamespace:
    """Exposes client.messages.create() to match AsyncAnthropic's interface."""

    def __init__(self, client: OpenAICompatLLMClient) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> _AnthropicShapedResponse:
        return await self._client._create(**kwargs)


class _AnthropicShapedResponse:
    """Minimal Anthropic response shape: .content[0].text."""

    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# Shared message helpers (same logic as AnthropicTransport)
# ---------------------------------------------------------------------------

def _message_role(message: CLMMessage) -> str:
    if message.role in {"user", "assistant"}:
        return message.role
    if message.type == "assistant_message":
        return "assistant"
    return "user"


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


def _build_messages(system_blocks: list[dict], messages: list[dict]) -> list[dict]:
    """Flatten Anthropic system_blocks + messages into OpenAI-compat message list."""
    result: list[dict] = []
    if system_blocks:
        system_text = "\n\n".join(
            block.get("text", "") for block in system_blocks if block.get("text")
        )
        if system_text:
            result.append({"role": "system", "content": system_text})
    result.extend(messages)
    return result
