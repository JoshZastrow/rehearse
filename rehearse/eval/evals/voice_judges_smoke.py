"""voice-judges-smoke — runnable CI smoke for the audio judges (Spec 2).

Composes:
  - `VoiceJudgesSmokeDataset`: 1 example with fixture transcript + per-turn
    audio durations.
  - `audio-fixture` environment: synthesizes silent per-turn WAVs.
  - Scoring plan: `AffectPerceptionJudgeScorer` + `DeliveryJudgeScorer`,
    backed by `StubAudioJudge` by default (deterministic 0.5 scores);
    set `REHEARSE_AUDIO_JUDGE=live` to use a real Gemini-backed judge
    instead. `live` mode requires `GEMINI_API_KEY` and a funded project.

The point of this eval: prove the audio-judge plumbing works
end-to-end without needing live API keys, real audio files, or
provider credit. For real voice scoring against real audio, use a
sandbox-shaped eval (Spec 2's second half) or wire these scorers into
your own scoring plan.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from rehearse.eval.datasets.voice_judges_smoke import VoiceJudgesSmokeDataset
from rehearse.eval.protocols import BenchmarkExample, Scorer
from rehearse.eval.scorers.affect_perception_judge import AffectPerceptionJudgeScorer
from rehearse.eval.scorers.audio_judge import AudioJudge, StubAudioJudge
from rehearse.eval.scorers.delivery_judge import DeliveryJudgeScorer


class VoiceJudgesSmokeEval:
    name = "voice-judges-smoke"
    version = "v0"
    supported_environments = frozenset({"audio-fixture"})
    preferred_environment = "audio-fixture"

    def __init__(self) -> None:
        self.dataset = VoiceJudgesSmokeDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        live = os.environ.get("REHEARSE_AUDIO_JUDGE", "").lower() == "live"
        if live:
            affect_judge = AudioJudge(model="gemini-2.5-pro")
            delivery_judge = AudioJudge(model="gemini-2.5-pro")
        else:
            affect_judge = StubAudioJudge(  # type: ignore[assignment]
                response={
                    "affect_perception": {
                        "score": 0.5,
                        "rationale": (
                            "stub: smoke eval default. "
                            "Set REHEARSE_AUDIO_JUDGE=live (and GEMINI_API_KEY) "
                            "to run against real Gemini."
                        ),
                    }
                }
            )
            delivery_judge = StubAudioJudge(  # type: ignore[assignment]
                response={
                    "delivery_quality": {
                        "score": 0.5,
                        "rationale": "stub: smoke eval default",
                    }
                }
            )

        return [
            AffectPerceptionJudgeScorer(judge=affect_judge),
            DeliveryJudgeScorer(judge=delivery_judge),
        ]

    def rollout_timeout_s(self) -> int:
        return 60
