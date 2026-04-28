"""Audio LLM provider registry."""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.providers.base import AudioLLMProvider
from rehearse.eval.providers.gemini import GeminiAudioProvider
from rehearse.eval.providers.vllm import VLLMAudioProvider

ProviderFactory = Callable[[dict[str, str]], AudioLLMProvider]

PROVIDERS: dict[str, ProviderFactory] = {
    "gemini": lambda slots: GeminiAudioProvider(model_slots=slots),
    "vllm": lambda slots: VLLMAudioProvider(model_slots=slots),
}


def get_provider(name: str, model_slots: dict[str, str]) -> AudioLLMProvider:
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}. registered: {sorted(PROVIDERS)}")
    return PROVIDERS[name](model_slots)


def list_providers() -> list[str]:
    return sorted(PROVIDERS)
