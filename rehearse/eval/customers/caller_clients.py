"""Thin LLM client wrappers for the synthetic eval caller.

Each class implements:
    async def complete(*, system_prompt: str, history: list[tuple[str, str]]) -> str

Pass an instance as llm_client= to EvalCallerParticipant, or use
make_caller_client(model_name) to resolve the right backend by model name.
"""

from __future__ import annotations

import os


class AnthropicCallerClient:
    """Anthropic SDK caller — routes claude-* models."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def complete(self, *, system_prompt: str, history: list[tuple[str, str]]) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        messages = [
            {"role": "user" if role == "provider" else "assistant", "content": text}
            for role, text in history
        ]
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "."})
        resp = await client.messages.create(
            model=self._model,
            max_tokens=256,
            temperature=0.7,
            system=system_prompt,
            messages=messages,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self._usage["prompt_tokens"] += getattr(usage, "input_tokens", 0)
            self._usage["completion_tokens"] += getattr(usage, "output_tokens", 0)
        return resp.content[0].text.strip() if resp.content else ""


class VLLMCallerClient:
    """OpenAI-compatible caller for vLLM-served models.

    Reads CALLER_VLLM_BASE_URL and CALLER_VLLM_API_KEY from the environment.
    Kept separate from VLLM_BASE_URL / VLLM_API_KEY used by the audio judge.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        base_url = os.environ.get("CALLER_VLLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                f"CALLER_VLLM_BASE_URL not set (required for caller model {model!r})"
            )
        self._base_url = base_url
        self._api_key = os.environ.get("CALLER_VLLM_API_KEY", "placeholder")
        self._usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def complete(self, *, system_prompt: str, history: list[tuple[str, str]]) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for role, text in history:
            messages.append({
                "role": "user" if role == "provider" else "assistant",
                "content": text,
            })
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "."})
        resp = await client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            temperature=0.7,
            messages=messages,  # type: ignore[arg-type]
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self._usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self._usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
        choice = resp.choices[0] if resp.choices else None
        return (choice.message.content or "").strip() if choice else ""


class MoshiCallerClient:
    """Placeholder caller for kyutai/moshi.

    Implement complete() once the Moshi endpoint is available.
    Set CALLER_MOSHI_BASE_URL and CALLER_MOSHI_API_KEY in the environment.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def complete(self, *, system_prompt: str, history: list[tuple[str, str]]) -> str:
        raise NotImplementedError(
            f"Caller backend for {self._model!r} is not yet implemented. "
            "Implement MoshiCallerClient.complete() once the endpoint is available."
        )


_REGISTRY: dict[str, type] = {
    "kyutai/moshi": MoshiCallerClient,
}

DEFAULT_CALLER_MODEL = "claude-sonnet-4-6"


def make_caller_client(
    model: str,
) -> AnthropicCallerClient | VLLMCallerClient | MoshiCallerClient:
    """Return the right caller client for a model name.

    Routing:
      claude-*      → AnthropicCallerClient
      kyutai/moshi  → MoshiCallerClient (placeholder)
      anything else → VLLMCallerClient (requires CALLER_VLLM_BASE_URL)
    """
    if model in _REGISTRY:
        return _REGISTRY[model](model)
    if model.startswith("claude-"):
        return AnthropicCallerClient(model)
    return VLLMCallerClient(model)
