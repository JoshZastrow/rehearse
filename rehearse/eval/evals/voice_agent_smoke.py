"""Voice-agent sandbox smoke eval."""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.datasets.voice_agent_smoke import VoiceAgentSmokeDataset
from rehearse.eval.protocols import BenchmarkExample, RolloutResult, Scorer
from rehearse.types import RubricScore


class _SandboxCompletionScorer:
    name = "sandbox_completion"
    dimension = "voice_agent_sandbox_completion"

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        runtime_output = (rollout.payload or {}).get("runtime_output", {})
        min_turns = int(example.expected.get("min_runtime_turns", 1))
        turns_received = int(runtime_output.get("turns_received", 0))
        value = 1.0 if rollout.status == "ok" and turns_received >= min_turns else 0.0
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=value,
                scorer="deterministic",
                rationale=f"runtime_turns={turns_received} min_runtime_turns={min_turns}",
            )
        ]


class VoiceAgentSmokeEval:
    name = "voice-agent-smoke"
    version = "v0"
    supported_environments = frozenset({"voice-agent-sandbox"})
    preferred_environment = "voice-agent-sandbox"

    def __init__(self) -> None:
        self.dataset = VoiceAgentSmokeDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        return [_SandboxCompletionScorer()]

    def rollout_timeout_s(self) -> int:
        return 10
