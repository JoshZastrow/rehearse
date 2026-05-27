"""Environment registry.

Environments are the runnable systems under evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.environments.audio_fixture import AudioFixtureEnvironment
from rehearse.eval.environments.live_audio_sandbox import LiveAudioSandboxEnvironment
from rehearse.eval.environments.media_probe import MediaProbeEnvironment
from rehearse.eval.environments.production_replay import ProductionReplayEnvironment
from rehearse.eval.environments.runtime_sandbox import RuntimeSandboxEnvironment
from rehearse.eval.environments.utils.echo import EchoEnvironment
from rehearse.eval.environments.utils.text_probe import TextProbeEnvironment
from rehearse.eval.protocols import Environment

EnvironmentFactory = Callable[[dict[str, str]], Environment]

ENVIRONMENTS: dict[str, EnvironmentFactory] = {
    "echo": lambda slots: EchoEnvironment(model_slots=slots),
    "raw-llm": lambda slots: TextProbeEnvironment(model_slots=slots),
    "multimodal-llm": lambda slots: MediaProbeEnvironment(model_slots=slots),
    "audio-fixture": lambda slots: AudioFixtureEnvironment(model_slots=slots),
    "live-audio-sandbox": lambda slots: LiveAudioSandboxEnvironment(model_slots=slots),
    "production-replay": lambda slots: ProductionReplayEnvironment(model_slots=slots),
    "runtime-sandbox": lambda slots: RuntimeSandboxEnvironment(model_slots=slots),
}


def get_environment(name: str, model_slots: dict[str, str]) -> Environment:
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown environment {name!r}. registered: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[name](model_slots)


def list_environments() -> list[str]:
    return sorted(ENVIRONMENTS)
