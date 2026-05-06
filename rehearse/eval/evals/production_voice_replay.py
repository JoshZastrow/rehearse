"""production-voice-replay — score real consented sessions with the audio stack.

Pairs `ProductionSessionsDataset` with the same audio + naturalness
scorers used by `voice-judges-smoke`, but over real recorded calls
re-materialized by `production-replay` instead of fixture audio.

This is the entry point for "evaluate my last conversation": real audio,
real timing, judge scores written under `evals/runs/<run_id>/`.

Audio judges back onto `StubAudioJudge` by default. Set
`REHEARSE_AUDIO_JUDGE=live` (with `GEMINI_API_KEY`) to score with real
Gemini. Naturalness is always live.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from rehearse.eval.datasets.production_sessions import ProductionSessionsDataset
from rehearse.eval.protocols import BenchmarkExample, Scorer
from rehearse.eval.scorers.affect_perception_judge import AffectPerceptionJudgeScorer
from rehearse.eval.scorers.audio_judge import AudioJudge, StubAudioJudge
from rehearse.eval.scorers.delivery_judge import DeliveryJudgeScorer
from rehearse.eval.scorers.naturalness import NaturalnessScorer


class ProductionVoiceReplayEval:
    name = "production-voice-replay"
    version = "v0"
    supported_environments = frozenset({"production-replay"})
    preferred_environment = "production-replay"

    def __init__(self) -> None:
        self.dataset = ProductionSessionsDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        live = os.environ.get("REHEARSE_AUDIO_JUDGE", "").lower() == "live"
        if live:
            affect_judge: AudioJudge | StubAudioJudge = AudioJudge(model="gemini-2.5-pro")
            delivery_judge: AudioJudge | StubAudioJudge = AudioJudge(model="gemini-2.5-pro")
        else:
            affect_judge = StubAudioJudge(
                response={
                    "affect_perception": {
                        "score": 0.5,
                        "rationale": (
                            "stub: production-voice-replay default. "
                            "Set REHEARSE_AUDIO_JUDGE=live (and GEMINI_API_KEY) "
                            "to run against real Gemini."
                        ),
                    }
                }
            )
            delivery_judge = StubAudioJudge(
                response={
                    "delivery_quality": {
                        "score": 0.5,
                        "rationale": "stub: production-voice-replay default",
                    }
                }
            )
        return [
            AffectPerceptionJudgeScorer(judge=affect_judge),
            DeliveryJudgeScorer(judge=delivery_judge),
            NaturalnessScorer(),
        ]

    def rollout_timeout_s(self) -> int:
        return 120
