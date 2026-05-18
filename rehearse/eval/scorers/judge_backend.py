"""Judge backend abstraction for LLMJudge.

Defines a `JudgeBackend` protocol and concrete implementations so providers
can be swapped without touching `LLMJudge` or any scorer.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class JudgeBackend(Protocol):
    async def complete(self, *, system: str, user: str) -> str: ...


class AnthropicBackend:
    """Wraps AsyncAnthropic — exact behavior extracted from the original LLMJudge."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                from rehearse.eval.scorers.llm_judge import LLMJudgeError

                raise LLMJudgeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=key)
        return self._client

    async def complete(self, *, system: str, user: str) -> str:
        from rehearse.eval.scorers.llm_judge import LLMJudgeError

        client = self._get_client()
        try:
            resp = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMJudgeError(f"judge call failed: {type(exc).__name__}: {exc}") from exc

        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )


class OpenAICompatBackend:
    """Calls any OpenAI-compatible /v1/chat/completions endpoint via httpx.

    Works with vLLM, Ollama, llama.cpp, and the Modal-hosted Gemma 4 server
    in infra/judge.py. Uses httpx directly — no openai SDK dependency.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        base_url: str,
        api_key: str = "local",
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = client  # injected in tests; real calls use httpx

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,
            )
        return self._client

    async def complete(self, *, system: str, user: str) -> str:
        from rehearse.eval.scorers.llm_judge import LLMJudgeError

        client = self._get_client()
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload
            )
            resp.raise_for_status()
        except Exception as exc:
            raise LLMJudgeError(f"judge call failed: {type(exc).__name__}: {exc}") from exc

        return resp.json()["choices"][0]["message"]["content"]


class LocalOllamaBackend(OpenAICompatBackend):
    """OpenAICompatBackend pre-configured for a local Ollama instance.

    Example::

        backend = LocalOllamaBackend(model="gemma3:27b")
    """

    def __init__(self, *, model: str, max_tokens: int = 2048, client: Any = None) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            client=client,
        )


_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
_DEFAULT_COMPAT_MODEL = "gemma-3-27b-it"


def backend_from_env(
    *,
    model: str | None = None,
    max_tokens: int = 2048,
) -> JudgeBackend:
    """Build a JudgeBackend from environment variables.

    Priority:
    1. ``JUDGE_BASE_URL`` set → :class:`OpenAICompatBackend`
    2. ``ANTHROPIC_API_KEY`` set → :class:`AnthropicBackend`
    3. Neither → raise :class:`~rehearse.eval.scorers.llm_judge.LLMJudgeError`

    The model is resolved as: *model* param > ``JUDGE_MODEL`` env var > default.
    """
    from rehearse.eval.scorers.llm_judge import LLMJudgeError

    env_model = os.environ.get("JUDGE_MODEL")
    base_url = os.environ.get("JUDGE_BASE_URL")

    if base_url:
        resolved_model = model or env_model or _DEFAULT_COMPAT_MODEL
        return OpenAICompatBackend(
            model=resolved_model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=os.environ.get("JUDGE_API_KEY", "local"),
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        resolved_model = model or env_model or _DEFAULT_ANTHROPIC_MODEL
        return AnthropicBackend(model=resolved_model, max_tokens=max_tokens)

    raise LLMJudgeError("no judge backend configured — set ANTHROPIC_API_KEY or JUDGE_BASE_URL")
