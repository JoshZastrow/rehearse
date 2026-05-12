"""Environment registry.

Environments are the runnable systems under evaluation. The older `targets`
package remains as a compatibility layer.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.environments.audio_fixture import AudioFixtureEnvironment
from rehearse.eval.environments.live_audio_sandbox import LiveAudioSandboxEnvironment
from rehearse.eval.environments.live_rollout_audio import LiveRolloutAudioEnvironment
from rehearse.eval.environments.multimodal_llm import MultimodalLLMEnvironment
from rehearse.eval.environments.production_replay import ProductionReplayEnvironment
from rehearse.eval.environments.runtime_sandbox import RuntimeSandboxEnvironment
from rehearse.eval.environments.voice_agent_sandbox import VoiceAgentSandboxEnvironment
from rehearse.eval.protocols import Environment
from rehearse.eval.targets.echo import EchoTarget
from rehearse.eval.targets.raw_llm import RawLLMTarget

EnvironmentFactory = Callable[[dict[str, str]], Environment]


def _deprecated_voice_agent_sandbox(slots: dict[str, str]) -> VoiceAgentSandboxEnvironment:
    import sys
    print(
        "DeprecationWarning: voice-agent-sandbox is deprecated; use runtime-sandbox.",
        file=sys.stderr,
    )
    return VoiceAgentSandboxEnvironment(model_slots=slots)


ENVIRONMENTS: dict[str, EnvironmentFactory] = {
    "echo": lambda slots: EchoTarget(model_slots=slots),
    "raw-llm": lambda slots: RawLLMTarget(model_slots=slots),
    "multimodal-llm": lambda slots: MultimodalLLMEnvironment(model_slots=slots),
    "voice-agent-sandbox": _deprecated_voice_agent_sandbox,
    "audio-fixture": lambda slots: AudioFixtureEnvironment(model_slots=slots),
    "live-rollout-audio": lambda slots: LiveRolloutAudioEnvironment(model_slots=slots),
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
