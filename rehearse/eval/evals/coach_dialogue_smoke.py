"""Coach-dialogue smoke eval.

LLM-vs-LLM dialogue rollout on the existing voice-agent-sandbox env. Reuses
the SandboxCompletionScorer from voice-agent-smoke since v0 only checks that
both sides produced enough turns. Trajectory-quality scoring lands in RLE3
per `docs/specs/v2026-04-29-mme-seeded-rl-sandbox-eval.md`.
"""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.datasets.coach_dialogue_smoke import CoachDialogueSmokeDataset
from rehearse.eval.evals.voice_agent_smoke import _SandboxCompletionScorer
from rehearse.eval.protocols import BenchmarkExample, Scorer


class CoachDialogueSmokeEval:
    name = "coach-dialogue-smoke"
    version = "v0"
    supported_environments = frozenset({"voice-agent-sandbox"})
    preferred_environment = "voice-agent-sandbox"

    def __init__(self) -> None:
        self.dataset = CoachDialogueSmokeDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        return [_SandboxCompletionScorer()]

    def rollout_timeout_s(self) -> int:
        return 120
