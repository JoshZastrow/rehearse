"""Environment registry.

Environments are the runnable systems under evaluation. The older `targets`
package remains as a compatibility layer.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.environments.multimodal_llm import MultimodalLLMEnvironment
from rehearse.eval.protocols import Environment
from rehearse.eval.targets.echo import EchoTarget
from rehearse.eval.targets.raw_llm import RawLLMTarget

EnvironmentFactory = Callable[[dict[str, str]], Environment]

ENVIRONMENTS: dict[str, EnvironmentFactory] = {
    "echo": lambda slots: EchoTarget(model_slots=slots),
    "raw-llm": lambda slots: RawLLMTarget(model_slots=slots),
    "multimodal-llm": lambda slots: MultimodalLLMEnvironment(model_slots=slots),
}


def get_environment(name: str, model_slots: dict[str, str]) -> Environment:
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown environment {name!r}. registered: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[name](model_slots)


def list_environments() -> list[str]:
    return sorted(ENVIRONMENTS)
