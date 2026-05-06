"""voice-judges-smoke — runnable CI smoke for the voice scoring stack.

Exercises the full text + audio + timing scoring pipeline end-to-end
without needing live API keys, real audio files, or provider credit.

Composes:
  - `VoiceJudgesSmokeDataset`: 1 example with fixture transcript +
    per-turn audio durations + `silence_between_turns_s` padding.
  - `audio-fixture` environment: synthesizes silent per-turn WAVs and
    a `timing.jsonl` derived from the durations.
  - Scoring plan:
      * `AffectPerceptionJudgeScorer` (Spec 2)
      * `DeliveryJudgeScorer` (Spec 2)
      * `NaturalnessScorer` (Spec 4) — deterministic, no judge

Audio judges back onto `StubAudioJudge` by default (deterministic 0.5
scores). Set `REHEARSE_AUDIO_JUDGE=live` to use a real Gemini-backed
judge instead — that requires `GEMINI_API_KEY` and a funded project.
The naturalness scorer is always live (deterministic arithmetic).
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from rehearse.eval.datasets.voice_judges_smoke import VoiceJudgesSmokeDataset
from rehearse.eval.protocols import BenchmarkExample, Scorer
from rehearse.eval.scorers.affect_perception_judge import AffectPerceptionJudgeScorer
from rehearse.eval.scorers.audio_judge import AudioJudge, StubAudioJudge
from rehearse.eval.scorers.delivery_judge import DeliveryJudgeScorer
from rehearse.eval.scorers.naturalness import NaturalnessScorer


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
            NaturalnessScorer(),
        ]

    def rollout_timeout_s(self) -> int:
        return 60
