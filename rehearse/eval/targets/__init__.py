"""Compatibility target registry.

New code should import from `rehearse.eval.environments`. This module remains
for older tests/scripts that still speak in targets.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.environments.multimodal_llm import MultimodalLLMEnvironment
from rehearse.eval.protocols import Target
from rehearse.eval.targets.echo import EchoTarget
from rehearse.eval.targets.raw_llm import RawLLMTarget

TargetFactory = Callable[[dict[str, str]], Target]

TARGETS: dict[str, TargetFactory] = {
    "echo": lambda slots: EchoTarget(model_slots=slots),
    "raw-llm": lambda slots: RawLLMTarget(model_slots=slots),
    "multimodal-llm": lambda slots: MultimodalLLMEnvironment(model_slots=slots),
}


def get_target(name: str, model_slots: dict[str, str]) -> Target:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}. registered: {sorted(TARGETS)}")
    return TARGETS[name](model_slots)


def list_targets() -> list[str]:
    return sorted(TARGETS)
