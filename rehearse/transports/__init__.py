"""LLM provider transports — format conversion and streaming I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rehearse.config import RuntimeConfig


def transport_from_config(config: RuntimeConfig) -> Any:
    """Return the appropriate LLMTransport for the runtime config.

    Priority:
      LITELLM_BASE_URL set → OpenAICompatTransport (routes through LiteLLM proxy)
      ANTHROPIC_API_KEY set → AnthropicTransport (direct Anthropic API)
    """
    if config.litellm_base_url:
        from rehearse.transports.openai_compat import OpenAICompatTransport
        return OpenAICompatTransport(
            base_url=config.litellm_base_url,
            api_key=config.litellm_api_key or "local",
            model=config.litellm_model,
        )
    if config.anthropic_api_key:
        from rehearse.transports.anthropic import AnthropicTransport
        return AnthropicTransport(api_key=config.anthropic_api_key)
    raise RuntimeError(
        "No LLM transport configured — set LITELLM_BASE_URL or ANTHROPIC_API_KEY"
    )


def llm_client_from_config(config: RuntimeConfig) -> Any:
    """Return an LLM client for non-transport uses (routing, classification).

    Returns an object with .messages.create() matching AsyncAnthropic's shape
    so callers like classify_topic() work unchanged.
    """
    if config.litellm_base_url:
        from rehearse.transports.openai_compat import OpenAICompatLLMClient
        return OpenAICompatLLMClient(
            base_url=config.litellm_base_url,
            api_key=config.litellm_api_key or "local",
        )
    if config.anthropic_api_key:
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=config.anthropic_api_key)
    raise RuntimeError(
        "No LLM client configured — set LITELLM_BASE_URL or ANTHROPIC_API_KEY"
    )
