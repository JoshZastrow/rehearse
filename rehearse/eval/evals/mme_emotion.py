"""MME-Emotion eval: 10-clip dataset + recognition scorer."""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.datasets.mme_emotion import MMEEmotionDataset
from rehearse.eval.protocols import BenchmarkExample, Scorer
from rehearse.eval.scorers.deterministic import MMERecognitionScorer


class MMEEmotionEval:
    name = "mme-emotion"
    version = "v0-10clip"
    supported_environments = frozenset({"multimodal-llm"})
    preferred_environment = "multimodal-llm"

    def __init__(self) -> None:
        self.dataset = MMEEmotionDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        return [MMERecognitionScorer()]

    def rollout_timeout_s(self) -> int:
        return 90
