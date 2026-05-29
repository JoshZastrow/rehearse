"""LLM judge backend selection.

Provides `backend_from_env()` which reads environment variables and returns a
backend with `async def complete(*, system: str, user: str) -> str`.

Priority:
  1. JUDGE_BASE_URL — explicit override for judge endpoint (OpenAI-compatible)
  2. LITELLM_BASE_URL — LiteLLM proxy (shared with production traffic)
  3. ANTHROPIC_API_KEY — direct Anthropic API
"""

from __future__ import annotations

import os
from typing import Any


class JudgeBackendError(RuntimeError):
    """Raised when backend config is missing or a call fails."""


class _OpenAICompatBackend:
    """OpenAI-compatible backend (LiteLLM proxy or any OpenAI-compat endpoint)."""

    def __init__(self, *, base_url: str, api_key: str, model: str, max_tokens: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, *, system: str, user: str) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise JudgeBackendError("httpx is required for OpenAI-compat backend") from exc

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise JudgeBackendError(f"judge backend HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
            except Exception as exc:
                raise JudgeBackendError(f"judge backend request failed: {exc}") from exc
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise JudgeBackendError(f"unexpected judge backend response shape: {data}") from exc


class _AnthropicBackend:
    """Direct Anthropic API backend."""

    def __init__(self, *, api_key: str, model: str, max_tokens: int) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, *, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise JudgeBackendError("anthropic package is required") from exc

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        try:
            resp = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise JudgeBackendError(f"Anthropic judge call failed: {exc}") from exc
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )


def backend_from_env(*, model: str, max_tokens: int) -> Any:
    """Return a judge backend configured from environment variables."""
    judge_url = os.environ.get("JUDGE_BASE_URL")
    litellm_url = os.environ.get("LITELLM_BASE_URL")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if judge_url:
        api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("LITELLM_API_KEY", "sk-rehearse-local")
        judge_model = os.environ.get("JUDGE_MODEL", model)
        return _OpenAICompatBackend(base_url=judge_url, api_key=api_key, model=judge_model, max_tokens=max_tokens)

    if litellm_url:
        api_key = os.environ.get("LITELLM_API_KEY", "sk-rehearse-local")
        litellm_model = os.environ.get("LITELLM_MODEL", model)
        return _OpenAICompatBackend(base_url=litellm_url, api_key=api_key, model=litellm_model, max_tokens=max_tokens)

    if anthropic_key:
        return _AnthropicBackend(api_key=anthropic_key, model=model, max_tokens=max_tokens)

    raise JudgeBackendError(
        "No LLM judge backend configured. "
        "Set JUDGE_BASE_URL, LITELLM_BASE_URL, or ANTHROPIC_API_KEY."
    )
