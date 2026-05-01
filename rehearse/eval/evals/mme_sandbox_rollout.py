"""MME-seeded RL sandbox rollout eval.

Composes:
  - MMERolloutSeedDataset (manifest-driven, MME-clip-keyed scenarios)
  - voice-agent-sandbox environment (LLM-vs-LLM dialogue, customer/coach=llm)
  - TrajectoryJudgeScorer (Claude Opus, emits the 3 reward dimensions)

This is the product-quality eval for v0. Capability-level emotion perception
lives in the direct `mme-emotion` eval; this one asks whether the rehearse
agent handles a multi-turn coaching conversation well given an MME-grounded
opening affect.
"""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.datasets.mme_rollout_seeds import MMERolloutSeedDataset
from rehearse.eval.protocols import BenchmarkExample, Scorer
from rehearse.eval.scorers.llm_judge import TrajectoryJudgeScorer


class MMESandboxRolloutEval:
    name = "mme-sandbox-rollout"
    version = "v0"
    supported_environments = frozenset({"voice-agent-sandbox"})
    preferred_environment = "voice-agent-sandbox"

    def __init__(self) -> None:
        self.dataset = MMERolloutSeedDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        return [TrajectoryJudgeScorer()]

    def rollout_timeout_s(self) -> int:
        return 180
